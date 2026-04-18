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
import math
import os
import random
import re

# Per Unsloth docs, unsloth must be imported BEFORE transformers/trl. We
# defer that import into the stages that touch the model (load/sft/grpo/
# eval/export) so data-only stages (test-data, test-rewards,
# test-curriculum) run without the torch+unsloth bootstrap.

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


# ============================================================================
# Model + LoRA — Unsloth Gemma-4 RL recipe (no vLLM).
# ============================================================================
def load_model_and_tokenizer(config, adapter_path=None):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        load_in_8bit=config.get("load_in_8bit", False),
        load_in_4bit=config.get("load_in_4bit", False),
        max_seq_length=config["max_seq_length"],
        full_finetuning=False,
        fast_inference=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def apply_lora(model, config):
    from unsloth import FastLanguageModel
    return FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )


# ============================================================================
# Prompt — production-verbatim system+user templates from
# backend/app/engine/ai.py:28-47, with the final "Output valid JSON."
# line replaced by a bare-digit instruction.
# ============================================================================
SYSTEM_TEMPLATE = """
You are an expert Connect Four player engine.
You are Player {player_id} (Symbol: {symbol}).
Opponent is Player {opponent_id} (Symbol: {opp_symbol}).
Board: 6 Rows x 7 Columns.
Goal: Connect 4 pieces in a row (Horizontal, Vertical, Diagonal).
Gravity: Pieces fall to the lowest empty slot.
"""

USER_TEMPLATE = """
Board (Visual):
{visual_board}

Board (Textual):
{textual_board}

Valid Columns: {valid_moves}

Analyze the board state carefully. Output only a single digit 0-6.
"""


# ============================================================================
# Rewards
# ============================================================================
# Last-digit-wins parser. Tolerates native thinking (which may mention
# column numbers mid-reasoning) and both skip_special_tokens modes:
# the final answer is whichever 0-6 digit appears last.
DIGIT_RE = re.compile(r"(?<!\d)([0-6])(?!\d)")


def parse_answer(completion):
    matches = DIGIT_RE.findall(completion)
    return int(matches[-1]) if matches else None


# Module-level ring buffer of recent completions. Populated by
# reward_format (which TRL calls every generation batch) so the
# ThinkingLogger callback can surface them to wandb without reaching
# into TRL internals. Holds up to RING_SIZE most-recent completion
# strings.
_RING_SIZE = 32
_RECENT_COMPLETIONS = []


def reward_format(completion, **kwargs):
    _RECENT_COMPLETIONS.append(str(completion)[:4096])
    if len(_RECENT_COMPLETIONS) > _RING_SIZE:
        del _RECENT_COMPLETIONS[:-_RING_SIZE]
    return 5.0 if parse_answer(completion) is not None else -10.0


def reward_move_quality(completion, scores, valid_cols, **kwargs):
    col = parse_answer(completion)
    if col is None or col not in valid_cols:
        return -10.0
    max_abs = max(abs(s) for s in scores) or 1.0
    return 10.0 * (scores[col] / max_abs)


# ============================================================================
# Curriculum — Gaussian over difficulty buckets. Easy positions (high
# std-dev of column scores, i.e. sharp-best) are bucket 0; hard
# positions (flat scores, i.e. many near-equal moves) are bucket 9.
# The sampler draws from a Gaussian centered on the current difficulty
# level and advances when the EMA reward for the current center bucket
# crosses advance_threshold.
# ============================================================================
class GaussianCurriculumSampler:
    def __init__(self, n_items, num_buckets=10, sigma=1.5,
                 max_reward=15.0, advance_ratio=0.7, ema_alpha=0.1, seed=42):
        self.n_items = n_items
        self.num_buckets = num_buckets
        self.sigma = sigma
        self.center = 0.0
        self.advance_threshold = max_reward * advance_ratio
        self.ema_alpha = ema_alpha
        self.bucket_reward_ema = [0.0] * num_buckets
        self.rng = random.Random(seed)

    def sample(self):
        b = round(self.rng.gauss(self.center, self.sigma))
        b = max(0, min(self.num_buckets - 1, b))
        per = self.n_items // self.num_buckets
        lo = b * per
        hi = (b + 1) * per if b < self.num_buckets - 1 else self.n_items
        return self.rng.randrange(lo, hi), b

    def update(self, bucket, reward):
        a = self.ema_alpha
        self.bucket_reward_ema[bucket] = (1 - a) * self.bucket_reward_ema[bucket] + a * reward
        cur = round(self.center)
        if cur < self.num_buckets - 1 and self.bucket_reward_ema[cur] >= self.advance_threshold:
            self.center = min(self.center + 1.0, self.num_buckets - 1)
            return True
        return False


# ============================================================================
# Thinking logger — surfaces raw completions (with any native-thinking
# delimiters visible as plain text) to wandb every N training steps.
# Reads from the _RECENT_COMPLETIONS ring buffer populated by
# reward_format, so it does not depend on TRL's private attributes.
# ============================================================================
class ThinkingLogger:
    def __init__(self, log_every=25):
        self.log_every = log_every

    def _log(self, step):
        try:
            import wandb
        except ImportError:
            return
        if not _RECENT_COMPLETIONS:
            return
        sample = _RECENT_COMPLETIONS[-1]
        parsed = parse_answer(sample)
        wandb.log({
            "sample/completion": wandb.Html(f"<pre>{sample[:2000]}</pre>"),
            "sample/parsed_answer": parsed if parsed is not None else -1,
            "sample/length_chars": len(sample),
        }, step=step)

    def as_callback(self):
        from transformers import TrainerCallback

        logger = self

        class _CB(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if state.global_step % logger.log_every != 0:
                    return
                logger._log(state.global_step)

        return _CB()


def build_prompt(tokenizer, entry):
    seq = entry["move_sequence"]
    g = reconstruct_board(seq)
    pid, sym = current_player_of(seq)
    opp_id = 2 if pid == 1 else 1
    opp_sym = "O" if sym == "X" else "X"

    system = SYSTEM_TEMPLATE.format(
        player_id=pid, symbol=sym,
        opponent_id=opp_id, opp_symbol=opp_sym,
    )
    user = USER_TEMPLATE.format(
        visual_board=g.get_visual_board(),
        textual_board=g.get_textual_description(),
        valid_moves=g.get_valid_moves(),
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

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

    if args.stage == "test-curriculum":
        # Always-high reward → center should walk to the top bucket.
        s = GaussianCurriculumSampler(n_items=10000)
        advances = 0
        for _ in range(2000):
            _, b = s.sample()
            if s.update(b, 12.0):
                advances += 1
        print(f"after 2000 high-reward samples:")
        print(f"  center={s.center}  advances={advances}")
        print(f"  bucket EMAs: {[f'{x:.2f}' for x in s.bucket_reward_ema]}")
        # Low reward → no advance.
        s2 = GaussianCurriculumSampler(n_items=10000)
        for _ in range(500):
            _, b = s2.sample()
            s2.update(b, 2.0)
        print(f"after 500 low-reward samples: center={s2.center}")
        return

    if args.stage == "test-rewards":
        cases = [
            ("bare-digit", "3",
             [0, 0, 0, 0.9, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6]),
            ("thinking-then-digit",
             "Let me analyze: Player X threatens in col 3.\n<channel|>3",
             [0, 0, 0, 0.9, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6]),
            ("verbose-single-digit", "I think the answer is column 3 obviously",
             [0, 0, 0, 0.9, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6]),
            ("wrong", "0",
             [0, 0, 0, 0.9, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6]),
            ("invalid-digit", "9",
             [0, 0, 0, 0.9, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6]),
            ("no-digit", "the answer",
             [0, 0, 0, 0.9, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6]),
            ("invalid-col-in-list", "4",
             [0.5, 0, 0, 0.9, 0, 0, 0], [0, 1, 2, 3, 5, 6]),
        ]
        for name, comp, scores, valid in cases:
            f = reward_format(comp)
            q = reward_move_quality(comp, scores, valid)
            parsed = parse_answer(comp)
            print(f"[{name:25}] parsed={str(parsed):5} f={f:+.1f} q={q:+.2f}")
        return

    if args.stage == "test-generate":
        from unsloth import FastLanguageModel
        raw = load_csv_data(args.csv)
        train, _ = split_data(raw)
        model, tok = load_model_and_tokenizer(config)
        FastLanguageModel.for_inference(model)
        for entry in train[:3]:
            p = build_prompt(tok, entry)
            inp = tok(p, return_tensors="pt").to("cuda")
            out = model.generate(
                **inp, max_new_tokens=1024,
                temperature=1.0, top_p=0.95, top_k=64, do_sample=True,
            )
            raw_decoded = tok.decode(out[0][inp.input_ids.shape[1]:],
                                     skip_special_tokens=False)
            clean = tok.decode(out[0][inp.input_ids.shape[1]:],
                               skip_special_tokens=True)
            print(f"\n=== seq={entry['move_sequence']!r} best={entry['best_col']} ===")
            print("--- RAW (special tokens visible) ---")
            print(raw_decoded)
            print("--- CLEAN (what rewards see) ---")
            print(clean)
        return

    if args.stage == "test-prompt":
        raw = load_csv_data(args.csv)
        train, _ = split_data(raw)
        _, tok = load_model_and_tokenizer(config)
        p = build_prompt(tok, train[0])
        print("=== PROMPT ===")
        print(p)
        print(f"\ntokens: {len(tok(p).input_ids)}")
        return

    if args.stage == "test-load":
        model, tok = load_model_and_tokenizer(config)
        model = apply_lora(model, config)
        model.print_trainable_parameters()
        hcfg = model.base_model.model.config
        print(f"\nnum_hidden_layers: {hcfg.num_hidden_layers}")
        print(f"vocab_size: {hcfg.vocab_size}")
        print(f"pad/eos: {tok.pad_token_id}/{tok.eos_token_id}")
        return

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
