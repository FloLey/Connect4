"""
Connect Four LLM Training Pipeline — GRPO with Unsloth (Gemma 4)
Usage:
  python connect4_train.py --model {e2b-bf16,e2b-8bit,e4b-bf16,e4b-8bit} --stage {grpo,eval,export,push} --csv connect4_data.csv
  python connect4_train.py --model e4b-bf16 --stage push --hf-repo yourname/connect4-agent-e4b-bf16
"""

import unsloth  # Must be imported first to apply training optimizations

import argparse
import csv
import json
import os
import re
import random

import math

import torch
from datasets import Dataset
from torch.utils.data import Sampler
from transformers import TrainerCallback

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from huggingface_hub import HfApi, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# =============================================================================
# BOARD RECONSTRUCTION (mirrors backend/app/engine/game.py)
# =============================================================================

ROWS = 6
COLS = 7

class ConnectFour:
    """Minimal board engine for replaying move sequences into production format."""

    def __init__(self):
        self.board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
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
        header = " " + " ".join([str(i) for i in range(COLS)])
        rows_str = []
        for r in range(ROWS):
            row_cells = [symbols[self.board[r][c]] for c in range(COLS)]
            rows_str.append("|" + "|".join(row_cells) + "|")
        return header + "\n" + "\n".join(rows_str)

    def get_textual_description(self):
        lines = []
        for c in range(COLS):
            pieces = []
            for r in range(ROWS - 1, -1, -1):
                val = self.board[r][c]
                if val == 0:
                    break
                pieces.append("P1" if val == 1 else "P2")
            desc = ", ".join(pieces) if pieces else "Empty"
            lines.append(f"Column {c}: {desc}")
        return "\n".join(lines)


def reconstruct_board(move_sequence):
    """Replay a move_sequence string into a ConnectFour board."""
    game = ConnectFour()
    for ch in move_sequence:
        game.drop_piece(int(ch))
    return game


# =============================================================================
# PRODUCTION PROMPT TEMPLATES (mirrors backend/app/engine/ai.py)
# =============================================================================

SYSTEM_TEMPLATE = """<|think|>
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

Analyze the board state carefully. Output valid JSON.
"""


def build_prompt(move_sequence):
    """Build (system_msg, user_msg) matching production format exactly."""
    game = reconstruct_board(move_sequence)
    num_moves = len(move_sequence)
    player_id = 1 if num_moves % 2 == 0 else 2
    opponent_id = 2 if player_id == 1 else 1
    symbol = "X" if player_id == 1 else "O"
    opp_symbol = "O" if player_id == 1 else "X"

    system_msg = SYSTEM_TEMPLATE.format(
        player_id=player_id, symbol=symbol,
        opponent_id=opponent_id, opp_symbol=opp_symbol,
    )
    user_msg = USER_TEMPLATE.format(
        visual_board=game.get_visual_board(),
        textual_board=game.get_textual_description(),
        valid_moves=game.get_valid_moves(),
    )
    return system_msg, user_msg


# =============================================================================
# MODEL CONFIGS (reduced num_generations for longer completions)
# =============================================================================

MODEL_CONFIGS = {
    "e2b-bf16": {"model_name": "unsloth/gemma-4-E2B-it", "lora_r": 32, "grpo_num_generations": 4, "grpo_batch_size": 4, "grpo_grad_accum": 2, "load_in_4bit": False, "load_in_8bit": False},
    "e2b-8bit": {"model_name": "unsloth/gemma-4-E2B-it", "lora_r": 32, "grpo_num_generations": 4, "grpo_batch_size": 4, "grpo_grad_accum": 2, "load_in_4bit": False, "load_in_8bit": True},
    "e4b-bf16": {"model_name": "unsloth/gemma-4-E4B-it", "lora_r": 32, "grpo_num_generations": 3, "grpo_batch_size": 2, "grpo_grad_accum": 4, "load_in_4bit": False, "load_in_8bit": False},
    "e4b-8bit": {"model_name": "unsloth/gemma-4-E4B-it", "lora_r": 32, "grpo_num_generations": 3, "grpo_batch_size": 2, "grpo_grad_accum": 4, "load_in_4bit": False, "load_in_8bit": True},
}


def get_config(model_size):
    mc = MODEL_CONFIGS[model_size]
    return {
        "model_name": mc["model_name"],
        "model_size": model_size,
        "max_seq_length": 4096,
        "load_in_4bit": mc.get("load_in_4bit", False),
        "load_in_8bit": mc.get("load_in_8bit", False),
        "lora_r": mc["lora_r"],
        "lora_alpha": mc["lora_r"],
        "lora_dropout": 0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "grpo_learning_rate": 1e-5,
        "grpo_max_steps": 2000,
        "grpo_batch_size": mc["grpo_batch_size"],
        "grpo_grad_accum": mc["grpo_grad_accum"],
        "grpo_num_generations": mc["grpo_num_generations"],
        "grpo_temperature": 0.7,
        "grpo_max_rows": 200_000,
        "csv_path": "connect4_data.csv",
        "grpo_output": f"outputs_grpo_{model_size}",
        "final_model": f"connect4-agent-{model_size}",
        "wandb_project": "connect4-llm",
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_csv_data(csv_path):
    data = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "move_sequence": row["move_sequence"].strip(),
                "scores": [float(row[f"col{i}"]) for i in range(7)],
                "best_col": int(row["best_col"]),
            })
    return data


def build_lookup_table(data):
    return {e["move_sequence"]: e["scores"] for e in data}


def difficulty_score(entry):
    """How hard is this position? Higher std = easier (clearer good vs bad moves).

    Returns negative std so sorted(..., reverse=True) puts easy first.
    """
    scores = entry["scores"]
    mean = sum(scores) / len(scores)
    return -(sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5


def sort_by_difficulty(data):
    """Sort positions from easiest (obvious best move) to hardest (ambiguous)."""
    return sorted(data, key=difficulty_score, reverse=True)


def split_data(csv_path):
    """Single deterministic train/eval split to prevent data leakage."""
    raw_data = load_csv_data(csv_path)
    random.seed(42)
    random.shuffle(raw_data)
    eval_data = raw_data[-10_000:]
    train_data = sort_by_difficulty(raw_data[:-10_000])
    return train_data, eval_data


# =============================================================================
# GRPO DATASET
# =============================================================================

def prepare_grpo_dataset(data, max_rows, tokenizer):
    formatted = []
    for entry in data[:max_rows]:
        system_msg, user_msg = build_prompt(entry["move_sequence"])
        conv = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        best_score = max(entry["scores"])
        formatted.append({
            "prompt": tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True),
            "move_sequence": entry["move_sequence"],
            "max_reward": best_score * 10.0 + 1.0,
        })
    return Dataset.from_list(formatted)


# =============================================================================
# CURRICULUM LEARNING — Gaussian adaptive difficulty (10 levels)
# =============================================================================

NUM_BUCKETS = 10


class GaussianCurriculumSampler(Sampler):
    """Samples from difficulty buckets using a Gaussian distribution.

    Dataset is pre-sorted easy→hard, split into NUM_BUCKETS equal buckets.
    A Gaussian centered on `center` (0=easy, 9=hard) controls sampling
    probability per bucket. The center advances as the model improves.
    """

    def __init__(self, dataset_size, num_buckets=NUM_BUCKETS, sigma=1.5, seed=42):
        self.dataset_size = dataset_size
        self.num_buckets = num_buckets
        self.sigma = sigma
        self.center = 0.0
        self.rng = random.Random(seed)

        # Precompute bucket index ranges
        bucket_size = dataset_size // num_buckets
        self.bucket_ranges = []
        for b in range(num_buckets):
            start = b * bucket_size
            end = start + bucket_size if b < num_buckets - 1 else dataset_size
            self.bucket_ranges.append((start, end))

    def _bucket_probs(self):
        raw = [math.exp(-((i - self.center) ** 2) / (2 * self.sigma ** 2)) for i in range(self.num_buckets)]
        total = sum(raw)
        return [p / total for p in raw]

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        probs = self._bucket_probs()
        indices = []
        for _ in range(self.dataset_size):
            # Pick bucket according to Gaussian probs
            r = self.rng.random()
            cumsum = 0.0
            bucket = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    bucket = i
                    break
            # Pick random index within that bucket
            start, end = self.bucket_ranges[bucket]
            indices.append(self.rng.randint(start, end - 1))
        return iter(indices)

    def advance(self):
        self.center = min(self.center + 1.0, self.num_buckets - 1)


class CurriculumCallback(TrainerCallback):
    """Advances the Gaussian center when the model approaches optimal play.

    Tracks actual rewards vs max possible rewards for positions seen.
    When ratio >= threshold over `check_interval` steps, advances to next level.
    """

    def __init__(self, sampler, reward_calc, threshold=0.7, check_interval=100):
        self.sampler = sampler
        self.reward_calc = reward_calc
        self.threshold = threshold
        self.check_interval = check_interval
        self.last_check_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Log curriculum state
        if logs is not None:
            logs["curriculum_level"] = self.sampler.center
            if self.reward_calc.max_reward_log:
                avg_max = sum(self.reward_calc.max_reward_log) / len(self.reward_calc.max_reward_log)
                avg_got = sum(self.reward_calc.reward_log) / len(self.reward_calc.reward_log)
                logs["curriculum_ratio"] = avg_got / avg_max if avg_max != 0 else 0.0

        if state.global_step - self.last_check_step < self.check_interval:
            return
        if self.sampler.center >= self.sampler.num_buckets - 1:
            return

        # Compute ratio from tracked rewards
        if not self.reward_calc.max_reward_log:
            return
        avg_max = sum(self.reward_calc.max_reward_log) / len(self.reward_calc.max_reward_log)
        avg_got = sum(self.reward_calc.reward_log) / len(self.reward_calc.reward_log)
        ratio = avg_got / avg_max if avg_max != 0 else 0.0

        if ratio >= self.threshold:
            self.sampler.advance()
            probs = self.sampler._bucket_probs()
            top_buckets = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
            print(f"\n>>> CURRICULUM: ratio {ratio:.2f} >= {self.threshold}, advancing to level {self.sampler.center:.0f}/9")
            print(f"    Top buckets: {['L'+str(b) + f'({probs[b]:.0%})' for b in top_buckets]}")

        # Reset logs for next interval
        self.reward_calc.max_reward_log.clear()
        self.reward_calc.reward_log.clear()
        self.last_check_step = state.global_step


# =============================================================================
# OUTPUT PARSING (handles Gemma 4 <|channel>thought blocks + JSON)
# =============================================================================

def strip_thinking(text):
    """Remove Gemma 4 thinking blocks from output."""
    return re.sub(r'<\|channel>thought.*?<channel\|>', '', text, flags=re.DOTALL).strip()


def extract_column_from_response(text):
    """Multi-strategy column extraction: JSON → regex → last digit."""
    cleaned = strip_thinking(text)
    # Strategy 1: full JSON parse
    try:
        data = json.loads(cleaned)
        col = data.get("column")
        if isinstance(col, int) and 0 <= col <= 6:
            return col
    except (json.JSONDecodeError, AttributeError):
        pass
    # Strategy 2: regex for "column": N
    match = re.search(r'"column"\s*:\s*(\d)', cleaned)
    if match:
        col = int(match.group(1))
        if 0 <= col <= 6:
            return col
    # Strategy 3: last digit 0-6
    digits = re.findall(r'[0-6]', cleaned)
    return int(digits[-1]) if digits else None


def classify_format(text):
    """Classify output quality: 'json', 'regex', 'digit', or 'invalid'."""
    cleaned = strip_thinking(text)
    try:
        data = json.loads(cleaned)
        col = data.get("column")
        if isinstance(col, int) and 0 <= col <= 6:
            return "json"
    except (json.JSONDecodeError, AttributeError):
        pass
    if re.search(r'"column"\s*:\s*(\d)', cleaned):
        return "regex"
    if re.findall(r'[0-6]', cleaned):
        return "digit"
    return "invalid"


# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

class RewardCalculator:
    def __init__(self, data):
        self.score_lookup = build_lookup_table(data)
        # Tracked by CurriculumCallback to compute reward/max ratio
        self.reward_log = []
        self.max_reward_log = []

    def reward_format(self, completions, **kwargs):
        """Reward for producing valid structured output."""
        rewards = []
        for c in completions:
            fmt = classify_format(c)
            rewards.append(1.0 if fmt == "json" else -10.0)
        return rewards

    def reward_move_quality(self, completions, move_sequence, max_reward=None, **kwargs):
        """Core reward: oracle tanh score for the chosen column, scaled to [-10, +10]."""
        rewards = []
        for i, (c, seq) in enumerate(zip(completions, move_sequence)):
            col = extract_column_from_response(c)
            if col is None:
                rewards.append(-10.0)
            else:
                scores = self.score_lookup.get(seq)
                rewards.append(scores[col] * 10.0 if scores else 0.0)
            # Track for curriculum: actual reward vs max possible
            self.reward_log.append(rewards[-1])
            if max_reward is not None:
                self.max_reward_log.append(max_reward[i])
        return rewards


# =============================================================================
# GRPO TRAINING
# =============================================================================

def run_grpo(config, train_data):
    from unsloth import FastModel
    from trl import GRPOConfig, GRPOTrainer
    print(f"\n{'='*60}\nGRPO TRAINING -- {config['model_name']}\n{'='*60}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        load_in_8bit=config["load_in_8bit"],
    )
    model = FastModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
    )

    reward_calc = RewardCalculator(train_data)
    print(f"Training on {len(train_data)} positions, {len(reward_calc.score_lookup)} in lookup")
    dataset = prepare_grpo_dataset(train_data, config["grpo_max_rows"], tokenizer)
    run_name = f"grpo-{config['model_size']}"

    # Gaussian curriculum: 10 difficulty levels, advances when reward/max >= threshold
    sampler = GaussianCurriculumSampler(len(dataset))
    curriculum_cb = CurriculumCallback(
        sampler, reward_calc,
        threshold=config.get("curriculum_threshold", 0.7),
        check_interval=100,
    )
    print(f"  Curriculum: Gaussian over {NUM_BUCKETS} levels, threshold={config.get('curriculum_threshold', 0.7)}")

    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        wandb.init(
            project=config["wandb_project"],
            name=run_name,
            tags=[config["model_size"], "grpo"],
            config={
                "model": config["model_name"],
                "stage": "grpo",
                "num_generations": config["grpo_num_generations"],
                "temperature": config["grpo_temperature"],
                "max_seq_length": config["max_seq_length"],
            },
            reinit=True,
        )

    grpo_kwargs = dict(
        output_dir=config["grpo_output"],
        temperature=config["grpo_temperature"],
        num_generations=config["grpo_num_generations"],
        max_completion_length=3072,
        use_vllm=True,
        learning_rate=config["grpo_learning_rate"],
        per_device_train_batch_size=config["grpo_batch_size"],
        gradient_accumulation_steps=config["grpo_grad_accum"],
        max_steps=config["grpo_max_steps"],
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        save_steps=300,
        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
    )
    if config.get("grpo_loss_type"):
        grpo_kwargs["loss_type"] = config["grpo_loss_type"]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_calc.reward_format,
            reward_calc.reward_move_quality,
        ],
        args=GRPOConfig(**grpo_kwargs),
        train_dataset=dataset,
        callbacks=[curriculum_cb],
    )
    trainer.sampler = sampler

    print("\nStarting GRPO... Watch reward climb at https://wandb.ai -> connect4-llm")
    print(f"  max_completion_length=3072 (extended thinking enabled)")
    print(f"  Curriculum: level 0→9 (easy→hard), Gaussian σ=1.5, advances at {config.get('curriculum_threshold', 0.7):.0%} ratio")
    trainer.train()
    if use_wandb:
        wandb.finish()
    model.save_pretrained(config["grpo_output"])
    tokenizer.save_pretrained(config["grpo_output"])
    print(f"GRPO saved to {config['grpo_output']}")


# =============================================================================
# EVALUATION
# =============================================================================

def run_eval(config, eval_data):
    from unsloth import FastModel
    print(f"\n{'='*60}\nEVALUATION -- {config['model_size'].upper()}\n{'='*60}")

    checkpoint_dir = config["grpo_output"]
    if not os.path.exists(checkpoint_dir):
        print("ERROR: No model found")
        return

    model, tokenizer = FastModel.from_pretrained(
        model_name=checkpoint_dir,
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        load_in_8bit=config["load_in_8bit"],
    )
    FastModel.for_inference(model)
    print(f"Evaluating on {len(eval_data)} held-out positions...")

    exact = 0
    top2 = 0
    score_sum = 0.0
    valid = 0
    invalid = 0
    json_count = 0
    phase_stats = {p: {"correct": 0, "total": 0, "score_sum": 0.0} for p in [
        "opening (0-8 moves)", "midgame (9-20 moves)", "endgame (21+ moves)"
    ]}

    for i, entry in enumerate(eval_data):
        if i % 1000 == 0 and i > 0:
            print(f"  ...{i}/10000 (acc: {100*exact/valid:.1f}%)" if valid else f"  ...{i}/10000")

        seq = entry["move_sequence"]
        scores = entry["scores"]
        best_col = entry["best_col"]
        phase = "opening (0-8 moves)" if len(seq) <= 8 else "midgame (9-20 moves)" if len(seq) <= 20 else "endgame (21+ moves)"

        system_msg, user_msg = build_prompt(seq)
        msgs = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        inputs = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(input_ids=inputs, max_new_tokens=3072, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        col = extract_column_from_response(response)

        if col is None:
            invalid += 1
            continue

        valid += 1
        if classify_format(response) == "json":
            json_count += 1
        score_sum += scores[col]
        if col == best_col:
            exact += 1
            phase_stats[phase]["correct"] += 1
        if col in sorted(range(7), key=lambda c: scores[c], reverse=True)[:2]:
            top2 += 1
        phase_stats[phase]["total"] += 1
        phase_stats[phase]["score_sum"] += scores[col]

    total = valid + invalid
    print(f"\nRESULTS -- {config['model_size'].upper()}")
    if total == 0:
        print("  No positions evaluated.")
        return
    print(f"  Valid outputs:     {valid}/{total} ({100*valid/total:.1f}%)")
    if valid == 0:
        print("  No valid outputs to score.")
        return
    print(f"  JSON format:       {json_count}/{valid} ({100*json_count/valid:.1f}%)")
    print(f"  Exact match:       {exact}/{valid} ({100*exact/valid:.1f}%)")
    print(f"  Top-2 match:       {top2}/{valid} ({100*top2/valid:.1f}%)")
    print(f"  Mean oracle score: {score_sum/valid:+.4f}")
    for phase, s in phase_stats.items():
        if s["total"] > 0:
            print(f"    {phase}: {100*s['correct']/s['total']:.1f}% exact, {s['score_sum']/s['total']:+.3f} avg (n={s['total']})")

    results = {
        "model": config["model_name"],
        "model_size": config["model_size"],
        "valid_pct": round(100 * valid / total, 2),
        "json_format_pct": round(100 * json_count / valid, 2),
        "exact_match_pct": round(100 * exact / valid, 2),
        "top2_match_pct": round(100 * top2 / valid, 2),
        "mean_oracle_score": round(score_sum / valid, 4),
    }
    with open(f"eval_results_{config['model_size']}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to eval_results_{config['model_size']}.json")


# =============================================================================
# EXPORT & PUSH
# =============================================================================

def export_model(config):
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name=config["grpo_output"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        load_in_8bit=config["load_in_8bit"],
    )
    model.save_pretrained_merged(config["final_model"], tokenizer, save_method="merged_16bit")
    model.save_pretrained_gguf(config["final_model"] + "-gguf", tokenizer, quantization_method="q4_k_m")
    print(f"Exported to {config['final_model']} and {config['final_model']}-gguf")


def push_to_hub(config):
    if not HF_AVAILABLE:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return
    from huggingface_hub import HfApi, create_repo
    hf_repo = config.get("hf_repo")
    if not hf_repo:
        print("ERROR: --hf-repo is required for push stage (e.g. yourname/connect4-agent-8b)")
        return
    if hf_repo.upper().endswith("-GGUF"):
        print("WARNING: --hf-repo should be the base repo name. Stripping '-GGUF' suffix.")
        hf_repo = hf_repo[:-5]
    api = HfApi()
    size = config["model_size"]
    merged_dir = config["final_model"]
    gguf_dir = config["final_model"] + "-gguf"
    if os.path.exists(merged_dir):
        print(f"\nPushing merged model to https://huggingface.co/{hf_repo}")
        create_repo(hf_repo, exist_ok=True)
        api.upload_folder(folder_path=merged_dir, repo_id=hf_repo, commit_message=f"Upload Connect4 agent ({size}) — merged 16-bit")
        print(f"  -> https://huggingface.co/{hf_repo}")
    else:
        print(f"WARNING: {merged_dir}/ not found — run --stage export first")
    if os.path.exists(gguf_dir):
        gguf_repo = hf_repo + "-GGUF"
        print(f"\nPushing GGUF to https://huggingface.co/{gguf_repo}")
        create_repo(gguf_repo, exist_ok=True)
        api.upload_folder(folder_path=gguf_dir, repo_id=gguf_repo, commit_message=f"Upload Connect4 agent ({size}) — GGUF q4_k_m")
        print(f"  -> https://huggingface.co/{gguf_repo}")
    else:
        print(f"WARNING: {gguf_dir}/ not found — run --stage export first")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Connect Four GRPO Training Pipeline")
    parser.add_argument("--model", choices=["e2b-bf16", "e2b-8bit", "e4b-bf16", "e4b-8bit"], required=True)
    parser.add_argument("--stage", choices=["grpo", "eval", "export", "push"], default="grpo")
    parser.add_argument("--csv", default="connect4_data.csv")
    parser.add_argument("--hf-repo", default=None, help="HuggingFace repo id for push (e.g. yourname/connect4-agent-8b)")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps (default: 2000)")
    parser.add_argument("--curriculum-threshold", type=float, default=0.7, help="Advance difficulty when reward/max >= this (default: 0.7)")
    parser.add_argument("--loss-type", default=None, help="GRPO loss type (e.g. dr_grpo)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    use_wandb = WANDB_AVAILABLE and not args.no_wandb

    config = get_config(args.model)
    config["csv_path"] = args.csv
    config["hf_repo"] = args.hf_repo
    config["grpo_loss_type"] = args.loss_type
    config["curriculum_threshold"] = args.curriculum_threshold
    if args.max_steps:
        config["grpo_max_steps"] = args.max_steps
    config["use_wandb"] = use_wandb

    print(f"\nConnect Four Pipeline | Model: {config['model_name']} | Stage: {args.stage} | Wandb: {'on' if use_wandb else 'off'}")

    train_data, eval_data = split_data(config["csv_path"])
    print(f"Data split: {len(train_data)} train, {len(eval_data)} eval (seed=42, no overlap)")

    if args.stage == "grpo":
        run_grpo(config, train_data)
    elif args.stage == "eval":
        run_eval(config, eval_data)
    elif args.stage == "export":
        export_model(config)
    elif args.stage == "push":
        push_to_hub(config)


if __name__ == "__main__":
    main()
