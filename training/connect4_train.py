"""
Connect Four LLM Training Pipeline — GRPO with trl + vLLM (Ministral 3)
Usage:
  python connect4_train.py --model {3b,8b} --stage {sft,grpo,eval,export,push} --csv connect4_data.csv
  python connect4_train.py --model 8b --stage push --hf-repo yourname/connect4-agent-8b
"""

import argparse
import csv
import json
import math
import os
import re
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*generation_config.*deprecated.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")

import torch
from datasets import Dataset
from torch.utils.data import Sampler
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel

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


def reconstruct_board(move_sequence):
    """Replay a move_sequence string into a ConnectFour board."""
    game = ConnectFour()
    for ch in move_sequence:
        game.drop_piece(int(ch))
    return game


# =============================================================================
# PRODUCTION PROMPT TEMPLATES
# =============================================================================

SYSTEM_TEMPLATE = """You are an expert Connect Four player. Board: 6 rows x 7 columns. Gravity: pieces fall to lowest empty slot. Goal: connect 4 in a row. You are Player {player_id} ({symbol}). Reply with just the column number (0-6)."""

USER_TEMPLATE = """{visual_board}

Valid columns: {valid_moves}"""


def build_prompt(move_sequence):
    """Build (system_msg, user_msg) for a given board position."""
    game = reconstruct_board(move_sequence)
    num_moves = len(move_sequence)
    player_id = 1 if num_moves % 2 == 0 else 2
    symbol = "X" if player_id == 1 else "O"

    system_msg = SYSTEM_TEMPLATE.format(player_id=player_id, symbol=symbol)
    user_msg = USER_TEMPLATE.format(
        visual_board=game.get_visual_board(),
        valid_moves=game.get_valid_moves(),
    )
    return system_msg, user_msg


# =============================================================================
# MODEL CONFIGS
# =============================================================================

MODEL_CONFIGS = {
    "3b": {"model_name": "mistralai/Ministral-3-3B-Instruct-2512-BF16", "lora_r": 32, "grpo_num_generations": 4, "grpo_batch_size": 1, "grpo_grad_accum": 2},
    "8b": {"model_name": "mistralai/Ministral-3-8B-Instruct-2512-BF16", "lora_r": 32, "grpo_num_generations": 3, "grpo_batch_size": 1, "grpo_grad_accum": 4},
}


def get_config(model_size):
    mc = MODEL_CONFIGS[model_size]
    return {
        "model_name": mc["model_name"],
        "model_size": model_size,
        "max_seq_length": 2048,
        "lora_r": mc["lora_r"],
        "lora_alpha": mc["lora_r"] * 2,
        "grpo_max_steps": 2000,
        "grpo_batch_size": mc["grpo_batch_size"],
        "grpo_grad_accum": mc["grpo_grad_accum"],
        "grpo_num_generations": mc["grpo_num_generations"],
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
    """Higher std = easier. Returns negative std so reverse sort puts easy first."""
    scores = entry["scores"]
    mean = sum(scores) / len(scores)
    return -(sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5


def sort_by_difficulty(data):
    return sorted(data, key=difficulty_score, reverse=True)


def split_data(csv_path):
    raw_data = load_csv_data(csv_path)
    random.seed(42)
    random.shuffle(raw_data)
    eval_data = raw_data[-10_000:]
    train_data = sort_by_difficulty(raw_data[:-10_000])
    return train_data, eval_data


# =============================================================================
# DATASETS
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


FILLER_THOUGHTS = [
    "Thinking about this move.", "Let me consider the options.",
    "Hmm, interesting position.", "I should pick carefully.",
    "Looking at the board.", "What's the best play here?",
    "Considering my options.", "Let me analyze this.",
    "I need to decide.", "Evaluating the columns.",
    "Time to make a move.", "Let me see.",
    "Okay, let me think.", "Which column is best?",
    "Tricky position.", "I think I see it.",
    "Let me figure this out.", "Alright, here goes.",
    "Not obvious, but let me try.", "What should I play?",
]


def prepare_sft_dataset(data, max_rows, tokenizer):
    rng = random.Random(42)
    formatted = []
    for entry in data[:max_rows]:
        system_msg, user_msg = build_prompt(entry["move_sequence"])
        thought = rng.choice(FILLER_THOUGHTS)
        best_col = entry["best_col"]
        conv = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"<think>{thought}</think>\n{best_col}"},
        ]
        formatted.append({
            "text": tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False),
        })
    return Dataset.from_list(formatted)


# =============================================================================
# SFT — teach the model the output format before GRPO
# =============================================================================

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer. Auto-detects model type."""
    from transformers import AutoConfig, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Use AutoModel to handle both CausalLM and ConditionalGeneration
    config = AutoConfig.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16,
        )
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch.bfloat16,
        )
    return model, tokenizer


def apply_lora(model, config):
    """Apply LoRA adapters to the model."""
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    return model


def run_sft(config, train_data):
    from trl import SFTTrainer, SFTConfig
    print(f"\n{'='*60}\nSFT FORMAT TRAINING -- {config['model_name']}\n{'='*60}")

    model, tokenizer = load_model_and_tokenizer(config["model_name"])
    model = apply_lora(model, config)

    dataset = prepare_sft_dataset(train_data[:1000], 1000, tokenizer)
    print(f"SFT on {len(dataset)} examples (teaching format only)")

    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        wandb.init(
            project=config["wandb_project"],
            name=f"sft-{config['model_size']}",
            tags=[config["model_size"], "sft"],
            config={"model": config["model_name"], "stage": "sft"},
            reinit=True,
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            warmup_steps=20,
            max_steps=50,
            learning_rate=2e-5,
            logging_steps=1,
            optim="adamw_8bit",
            output_dir=config["grpo_output"] + "_sft",
            save_steps=50,
            max_length=config["max_seq_length"],
            dataset_text_field="text",
            report_to="wandb" if use_wandb else "none",
            run_name=f"sft-{config['model_size']}",
            bf16=True,
        ),
    )

    print("Starting SFT... Teaching the model to output <think>...</think> + digit")
    trainer.train()
    if use_wandb:
        wandb.finish()

    # Save merged model (LoRA merged into base weights)
    sft_dir = config["grpo_output"] + "_sft"
    merged = model.merge_and_unload()
    merged.save_pretrained(sft_dir)
    tokenizer.save_pretrained(sft_dir)
    print(f"SFT merged model saved to {sft_dir}")

    # Verify format compliance
    print("\nVerifying format compliance on 20 samples...")
    merged.eval()
    correct = 0
    total = 20
    for entry in train_data[:total]:
        system_msg, user_msg = build_prompt(entry["move_sequence"])
        msgs = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        encoded = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids.to(merged.device)
        elif isinstance(encoded, dict):
            input_ids = encoded["input_ids"].to(merged.device)
        else:
            input_ids = encoded.to(merged.device)
        with torch.no_grad():
            outputs = merged.generate(input_ids=input_ids, max_new_tokens=128, do_sample=False)
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        if is_clean_output(response):
            correct += 1
    pct = 100 * correct / total
    print(f"  Format compliance: {correct}/{total} ({pct:.0f}%)")

    if pct >= 80:
        print(f"\n>>> Format compliance {pct:.0f}% >= 80%. Starting GRPO automatically...\n")
        del merged
        torch.cuda.empty_cache()
        return True
    else:
        print(f"\n>>> Format compliance {pct:.0f}% < 80%. Consider more SFT steps.")
        return False


# =============================================================================
# CURRICULUM LEARNING — Gaussian adaptive difficulty (10 levels)
# =============================================================================

NUM_BUCKETS = 10


class GaussianCurriculumSampler(Sampler):
    """Samples from difficulty buckets using a Gaussian distribution."""

    def __init__(self, dataset_size, num_buckets=NUM_BUCKETS, sigma=1.5, seed=42):
        self.dataset_size = dataset_size
        self.num_buckets = num_buckets
        self.sigma = sigma
        self.center = 0.0
        self.rng = random.Random(seed)

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
            r = self.rng.random()
            cumsum = 0.0
            bucket = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    bucket = i
                    break
            start, end = self.bucket_ranges[bucket]
            indices.append(self.rng.randint(start, end - 1))
        return iter(indices)

    def advance(self):
        self.center = min(self.center + 1.0, self.num_buckets - 1)


class CurriculumCallback(TrainerCallback):
    """Advances difficulty when the model approaches optimal play."""

    def __init__(self, sampler, reward_calc, threshold=0.7, check_interval=100):
        self.sampler = sampler
        self.reward_calc = reward_calc
        self.threshold = threshold
        self.check_interval = check_interval
        self.last_check_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        logs["curriculum_level"] = self.sampler.center
        if self.reward_calc.max_reward_log:
            avg_max = sum(self.reward_calc.max_reward_log) / len(self.reward_calc.max_reward_log)
            avg_got = sum(self.reward_calc.reward_log) / len(self.reward_calc.reward_log)
            logs["curriculum_ratio"] = avg_got / avg_max if avg_max != 0 else 0.0

        if state.global_step - self.last_check_step < self.check_interval:
            return
        if self.sampler.center >= self.sampler.num_buckets - 1:
            return

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

        self.reward_calc.max_reward_log.clear()
        self.reward_calc.reward_log.clear()
        self.last_check_step = state.global_step


# =============================================================================
# OUTPUT PARSING (handles <think> blocks, expects single digit 0-6)
# =============================================================================

def strip_thinking(text):
    """Remove <think>...</think> blocks from output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def extract_column_from_response(text):
    """Extract column number (0-6) only if output is a clean single digit."""
    cleaned = strip_thinking(text).strip()
    if cleaned in {"0", "1", "2", "3", "4", "5", "6"}:
        return int(cleaned)
    return None


def is_clean_output(text):
    """Check if the output (after thinking) is just a single digit 0-6."""
    cleaned = strip_thinking(text)
    return cleaned.strip() in {"0", "1", "2", "3", "4", "5", "6"}


# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

class RewardCalculator:
    def __init__(self, data):
        self.score_lookup = build_lookup_table(data)
        self.reward_log = []
        self.max_reward_log = []

    def reward_format(self, completions, **kwargs):
        """Reward for outputting just a single digit 0-6."""
        rewards = []
        for c in completions:
            rewards.append(1.0 if is_clean_output(c) else -10.0)
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
            self.reward_log.append(rewards[-1])
            if max_reward is not None:
                self.max_reward_log.append(max_reward[i])
        return rewards


# =============================================================================
# GRPO TRAINING
# =============================================================================

def run_grpo(config, train_data):
    from trl import GRPOConfig, GRPOTrainer
    print(f"\n{'='*60}\nGRPO TRAINING -- {config['model_name']}\n{'='*60}")

    # Load from SFT merged model if available, otherwise base model
    sft_dir = config["grpo_output"] + "_sft"
    if os.path.exists(sft_dir):
        print(f"  Loading merged SFT model: {sft_dir}")
        load_model = sft_dir
    else:
        print("  WARNING: No SFT checkpoint found. Run --stage sft first.")
        load_model = config["model_name"]

    model, tokenizer = load_model_and_tokenizer(load_model)
    model = apply_lora(model, config)

    reward_calc = RewardCalculator(train_data)
    print(f"Training on {len(train_data)} positions, {len(reward_calc.score_lookup)} in lookup")
    dataset = prepare_grpo_dataset(train_data, config["grpo_max_rows"], tokenizer)
    run_name = f"grpo-{config['model_size']}"

    # Gaussian curriculum
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
            },
            reinit=True,
        )

    grpo_config = GRPOConfig(
        output_dir=config["grpo_output"],
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
        temperature=1.0,
        num_generations=config["grpo_num_generations"],
        max_completion_length=512,
        max_prompt_length=512,
        per_device_train_batch_size=config["grpo_batch_size"],
        gradient_accumulation_steps=config["grpo_grad_accum"],
        max_steps=config["grpo_max_steps"],
        save_steps=300,
        learning_rate=5e-5,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        max_grad_norm=0.1,
        logging_steps=1,
        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_calc.reward_format,
            reward_calc.reward_move_quality,
        ],
        args=grpo_config,
        train_dataset=dataset,
        callbacks=[curriculum_cb],
    )

    print("\nStarting GRPO with vLLM colocate mode...")
    print(f"  max_completion_length=512")
    print(f"  Curriculum: level 0→9 (easy→hard), advances at {config.get('curriculum_threshold', 0.7):.0%} ratio")
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
    print(f"\n{'='*60}\nEVALUATION -- {config['model_size'].upper()}\n{'='*60}")

    checkpoint_dir = config["grpo_output"]
    if not os.path.exists(checkpoint_dir):
        print("ERROR: No model found")
        return

    model, tokenizer = load_model_and_tokenizer(config["model_name"])
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()
    print(f"Evaluating on {len(eval_data)} held-out positions...")

    exact = 0
    top2 = 0
    score_sum = 0.0
    valid = 0
    invalid = 0
    phase_stats = {p: {"correct": 0, "total": 0, "score_sum": 0.0} for p in [
        "opening (0-8 moves)", "midgame (9-20 moves)", "endgame (21+ moves)"
    ]}

    for i, entry in enumerate(eval_data):
        if i % 1000 == 0 and i > 0:
            print(f"  ...{i}/{len(eval_data)} (acc: {100*exact/valid:.1f}%)" if valid else f"  ...{i}/{len(eval_data)}")

        seq = entry["move_sequence"]
        scores = entry["scores"]
        best_col = entry["best_col"]
        phase = "opening (0-8 moves)" if len(seq) <= 8 else "midgame (9-20 moves)" if len(seq) <= 20 else "endgame (21+ moves)"

        system_msg, user_msg = build_prompt(seq)
        msgs = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        input_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(model.device)

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, max_new_tokens=128, do_sample=False)
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        col = extract_column_from_response(response)

        if col is None:
            invalid += 1
            continue

        valid += 1
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
    model, tokenizer = load_model_and_tokenizer(config["model_name"])
    model = PeftModel.from_pretrained(model, config["grpo_output"])
    merged = model.merge_and_unload()
    merged.save_pretrained(config["final_model"])
    tokenizer.save_pretrained(config["final_model"])
    print(f"Exported merged model to {config['final_model']}")


def push_to_hub(config):
    if not HF_AVAILABLE:
        print("ERROR: huggingface_hub not installed.")
        return
    hf_repo = config.get("hf_repo")
    if not hf_repo:
        print("ERROR: --hf-repo is required for push stage")
        return
    api = HfApi()
    merged_dir = config["final_model"]
    if os.path.exists(merged_dir):
        print(f"\nPushing to https://huggingface.co/{hf_repo}")
        create_repo(hf_repo, exist_ok=True)
        api.upload_folder(folder_path=merged_dir, repo_id=hf_repo, commit_message=f"Upload Connect4 agent ({config['model_size']})")
        print(f"  -> https://huggingface.co/{hf_repo}")
    else:
        print(f"WARNING: {merged_dir}/ not found — run --stage export first")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Connect Four GRPO Training Pipeline")
    parser.add_argument("--model", choices=["3b", "8b"], required=True)
    parser.add_argument("--stage", choices=["sft", "grpo", "eval", "export", "push"], default="grpo")
    parser.add_argument("--csv", default="connect4_data.csv")
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--curriculum-threshold", type=float, default=0.7)
    parser.add_argument("--loss-type", default=None)
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

    if args.stage == "sft":
        if run_sft(config, train_data):
            run_grpo(config, train_data)
    elif args.stage == "grpo":
        run_grpo(config, train_data)
    elif args.stage == "eval":
        run_eval(config, eval_data)
    elif args.stage == "export":
        export_model(config)
    elif args.stage == "push":
        push_to_hub(config)


if __name__ == "__main__":
    main()
