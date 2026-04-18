"""Connect 4 GSPO training — Gemma 4 + Unsloth + TRL 0.28.

Pipeline: SFT (minimal digit-only warmup) -> GSPO (curriculum RL) -> eval -> export.

Usage:
  python connect4_train.py --model e4b-8bit --stage sft  --csv connect4_data.csv --hf-repo Betha/connect4-agent-e4b-8bit
  python connect4_train.py --model e4b-8bit --stage grpo --sft --csv connect4_data.csv --hf-repo Betha/connect4-agent-e4b-8bit
  python connect4_train.py --model e4b-8bit --stage eval --csv connect4_data.csv
"""
import argparse
import csv
import json
import os
import random

# unsloth must be imported before transformers — injects Gemma-4 RL patches.
import unsloth  # noqa: F401

ROWS = 6
COLS = 7


# ============================================================================
# Game — ported from backend/app/engine/game.py (no win-check; training
# positions are guaranteed non-terminal, so drop_piece just stacks).
# ============================================================================
class ConnectFour:
    def __init__(self):
        # Row 0 = top, Row 5 = bottom. 0=empty, 1=P1 (X), 2=P2 (O).
        self.board = [[0] * COLS for _ in range(ROWS)]
        self.current_turn = 1

    def drop_piece(self, col):
        if col < 0 or col >= COLS or self.board[0][col] != 0:
            return False
        for r in range(ROWS - 1, -1, -1):
            if self.board[r][col] == 0:
                self.board[r][col] = self.current_turn
                self.current_turn = 2 if self.current_turn == 1 else 1
                return True
        return False

    def get_valid_moves(self):
        return [c for c in range(COLS) if self.board[0][c] == 0]

    def get_visual_board(self):
        symbols = {0: ".", 1: "X", 2: "O"}
        header = " " + " ".join(str(i) for i in range(COLS))
        rows = ["|" + "|".join(symbols[self.board[r][c]] for c in range(COLS)) + "|"
                for r in range(ROWS)]
        return header + "\n" + "\n".join(rows)

    def get_textual_description(self):
        lines = []
        for c in range(COLS):
            pieces = []
            for r in range(ROWS - 1, -1, -1):
                v = self.board[r][c]
                if v == 0:
                    break
                pieces.append("P1" if v == 1 else "P2")
            desc = ", ".join(pieces) if pieces else "Empty"
            lines.append(f"Column {c}: {desc}")
        return "\n".join(lines)


def reconstruct_board(move_sequence):
    g = ConnectFour()
    for ch in move_sequence:
        g.drop_piece(int(ch))
    return g


def current_player_of(move_sequence):
    """Returns (player_id, symbol). Player 1 = X, Player 2 = O."""
    pid = 1 if len(move_sequence) % 2 == 0 else 2
    return pid, ("X" if pid == 1 else "O")


# ============================================================================
# Data
# ============================================================================
def load_csv_data(path):
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.append({
                "move_sequence": row["move_sequence"].strip(),
                "scores": [float(row[f"col{i}"]) for i in range(7)],
                "best_col": int(row["best_col"]),
            })
    return out


def difficulty_score(entry):
    """Higher = easier. Std-dev of column scores: clear-best positions are easy."""
    s = entry["scores"]
    m = sum(s) / len(s)
    return (sum((x - m) ** 2 for x in s) / len(s)) ** 0.5


def split_data(raw, seed=42):
    """Deterministic shuffle -> last 10k eval, rest train (sorted easiest-first)."""
    rng = random.Random(seed)
    raw = list(raw)
    rng.shuffle(raw)
    eval_data = raw[-10_000:]
    train_data = raw[:-10_000]
    train_sorted = sorted(train_data, key=difficulty_score, reverse=True)
    return train_sorted, eval_data

MODEL_CONFIGS = {
    "e4b-8bit": {
        "model_name": "unsloth/gemma-4-E4B-it",
        "load_in_8bit": True,
        "load_in_4bit": False,
        "max_seq_length": 2560,

        "lora_r": 32,
        "lora_alpha": 64,

        "grpo_lr": 5e-5,
        "grpo_loss_type": "gspo",
        "grpo_epsilon": 3e-4,
        "grpo_epsilon_high": 4e-4,
        "grpo_beta": 0.0,
        "grpo_num_generations": 4,
        "grpo_batch_size": 2,
        "grpo_grad_accum": 2,
        "grpo_temperature": 1.0,
        "grpo_max_completion_length": 2048,
        "grpo_max_prompt_length": 512,
        "grpo_max_steps": 1000,
        "grpo_save_steps": 300,

        "sft_examples": 200,
        "sft_max_steps": 25,
        "sft_lr": 2e-5,
        "sft_batch_size": 8,
        "sft_grad_accum": 2,
    },
}


def get_config(variant):
    cfg = dict(MODEL_CONFIGS[variant])
    cfg["variant"] = variant
    cfg["output_dir"] = f"outputs_grpo_{variant}"
    cfg["sft_output_dir"] = f"outputs_sft_{variant}"
    cfg["final_model_dir"] = f"connect4-agent-{variant}"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_CONFIGS), required=True)
    ap.add_argument("--stage", required=True, choices=[
        "test-data", "test-load", "test-prompt", "test-generate",
        "test-rewards", "test-curriculum",
        "sft", "grpo", "eval", "export", "push",
    ])
    ap.add_argument("--csv", default="connect4_data.csv")
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--sft", action="store_true",
                    help="GRPO: load SFT adapter from {hf-repo}-sft before training")
    args = ap.parse_args()

    config = get_config(args.model)
    config["csv"] = args.csv
    config["hf_repo"] = args.hf_repo

    print(f"=== Connect4 {args.model} / {args.stage} ===")
    print(json.dumps({k: v for k, v in config.items() if "dir" not in k}, indent=2))

    if args.stage == "test-data":
        raw = load_csv_data(args.csv)
        train, eval_ = split_data(raw)
        print(f"\ntotal={len(raw)} train={len(train)} eval={len(eval_)}")
        e = train[0]
        g = reconstruct_board(e["move_sequence"])
        pid, sym = current_player_of(e["move_sequence"])
        print(f"\nEasiest train position:")
        print(f"  move_sequence={e['move_sequence']!r}")
        print(f"  best_col={e['best_col']}  player-to-move={pid} ({sym})")
        print(f"\nVisual:\n{g.get_visual_board()}")
        print(f"\nTextual:\n{g.get_textual_description()}")
        print(f"\nValid cols: {g.get_valid_moves()}")
        return

    if args.stage == "grpo":
        raise NotImplementedError("M11-13")
    elif args.stage == "sft":
        raise NotImplementedError("M10")
    elif args.stage == "eval":
        raise NotImplementedError("M15")
    elif args.stage == "export":
        raise NotImplementedError("M16")
    elif args.stage == "push":
        raise NotImplementedError("M16")
    else:
        raise NotImplementedError(args.stage)


if __name__ == "__main__":
    main()
