"""
Connect Four LLM Training Pipeline — GSPO with trl + Unsloth (Gemma 4)

Default stage is `sft`, which runs a 50-step format warmup and then auto-
chains into GSPO when format compliance is ≥ 80%.

Usage:
  python connect4_train.py --model {e2b-bf16,e2b-8bit,e4b-bf16,e4b-8bit} [--stage {sft,grpo,eval,export,push}] --csv connect4_data.csv
  python connect4_train.py --model e4b-8bit --hf-repo yourname/connect4-agent-e4b-8bit  # SFT → GSPO
  python connect4_train.py --model e4b-8bit --stage grpo --hf-repo yourname/connect4-agent-e4b-8bit  # resume GSPO only
  python connect4_train.py --model e4b-8bit --stage push --hf-repo yourname/connect4-agent-e4b-8bit
"""

# Unsloth must be imported before transformers / trl so its patches apply.
from unsloth import FastLanguageModel  # noqa: F401

import argparse
import csv
import gc
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
from transformers import TrainerCallback
from peft import PeftModel

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
    "e2b-bf16": {"model_name": "unsloth/gemma-4-E2B-it", "load_in_4bit": False, "load_in_8bit": False, "lora_r": 16, "grpo_num_generations": 4, "grpo_batch_size": 4, "grpo_grad_accum": 2},
    "e2b-8bit": {"model_name": "unsloth/gemma-4-E2B-it", "load_in_4bit": False, "load_in_8bit": True,  "lora_r": 16, "grpo_num_generations": 4, "grpo_batch_size": 4, "grpo_grad_accum": 2},
    "e4b-bf16": {"model_name": "unsloth/gemma-4-E4B-it", "load_in_4bit": False, "load_in_8bit": False, "lora_r": 16, "grpo_num_generations": 3, "grpo_batch_size": 3, "grpo_grad_accum": 4},
    "e4b-8bit": {"model_name": "unsloth/gemma-4-E4B-it", "load_in_4bit": False, "load_in_8bit": True,  "lora_r": 16, "grpo_num_generations": 3, "grpo_batch_size": 3, "grpo_grad_accum": 4},
}


def get_config(model_size):
    mc = MODEL_CONFIGS[model_size]
    return {
        "model_name": mc["model_name"],
        "model_size": model_size,
        "max_seq_length": 2048,
        "load_in_4bit": mc["load_in_4bit"],
        "load_in_8bit": mc["load_in_8bit"],
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

def load_model_and_tokenizer(config, model_path=None):
    """Load model and tokenizer via Unsloth's FastLanguageModel (text-only)."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path or config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config.get("load_in_4bit", False),
        load_in_8bit=config.get("load_in_8bit", False),
        full_finetuning=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def apply_lora(model, config):
    """Apply LoRA adapters via Unsloth's helper."""
    return FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )


def _sanitize_config_for_save(cfg):
    """Strip non-JSON-serializable callables from a transformers PretrainedConfig.

    Gemma 4 + Unsloth patching leaves function/method objects on the config,
    which crashes config.to_json_string() at save time. We walk the config
    and any nested sub-configs (text_config, vision_config, audio_config)
    and delete attributes whose value is callable but isn't a type/class.
    """
    import types
    if cfg is None:
        return
    bad_types = (types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.BuiltinMethodType, types.LambdaType)
    for attr in list(vars(cfg).keys()):
        try:
            val = getattr(cfg, attr)
        except Exception:
            continue
        if isinstance(val, bad_types):
            try:
                delattr(cfg, attr)
            except Exception:
                pass
    for sub in ("text_config", "vision_config", "audio_config"):
        if hasattr(cfg, sub):
            _sanitize_config_for_save(getattr(cfg, sub))


def run_sft(config, train_data):
    from trl import SFTTrainer, SFTConfig
    print(f"\n{'='*60}\nSFT FORMAT TRAINING -- {config['model_name']}\n{'='*60}")

    model, tokenizer = load_model_and_tokenizer(config)
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

    # Save LoRA adapter only (not the merged base). Merged save hits a
    # Gemma 4 + Unsloth bug where model.config carries a function object
    # that can't be JSON-serialized. PEFT's adapter save only writes
    # adapter_config.json + adapter_model.safetensors and never touches
    # the base-model config, so it's safe.
    sft_dir = config["grpo_output"] + "_sft"
    _sanitize_config_for_save(model.config)  # defensive; noop if already clean
    model.save_pretrained(sft_dir)
    tokenizer.save_pretrained(sft_dir)
    print(f"SFT LoRA adapter saved to {sft_dir}")

    # Merge into a new in-memory model for format compliance verification.
    merged = model.merge_and_unload()

    # Verify format compliance
    print("\nVerifying format compliance on 20 samples...")
    merged.eval()
    correct = 0
    total = 20
    for entry in train_data[:total]:
        system_msg, user_msg = build_prompt(entry["move_sequence"])
        msgs = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        # Two-step render + tokenize to dodge Gemma 4's multimodal Processor path.
        # `text=` must be an explicit kwarg: Unsloth's patched processor __call__
        # has signature (self, images=None, text=None, videos=None, **kwargs), so
        # a positional arg gets routed to `images` and `text` ends up None — which
        # then crashes inside Gemma4Processor doing text[0].
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text=rendered, return_tensors="pt").input_ids.to(merged.device)
        with torch.no_grad():
            outputs = merged.generate(input_ids=input_ids, max_new_tokens=128, do_sample=False)
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        if is_clean_output(response):
            correct += 1
    pct = 100 * correct / total
    print(f"  Format compliance: {correct}/{total} ({pct:.0f}%)")

    # Aggressive cleanup before returning — GSPO will re-load base Gemma 4 in
    # 8-bit on the 24 GB 4090 and needs ALL of SFT's GPU state (8-bit base,
    # dequantized merged bf16 model, trainer + optimizer state, dataset) to
    # be freed first. bnb-8bit refuses to load if accelerate has to offload
    # any layer to CPU, and that's exactly what happens if this memory
    # lingers.
    for _name in ("merged", "model", "trainer", "dataset"):
        if _name in locals():
            del _name
    try:
        del merged
    except NameError:
        pass
    try:
        del model
    except NameError:
        pass
    try:
        del trainer
    except NameError:
        pass
    try:
        del dataset
    except NameError:
        pass
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if pct >= 80:
        print(f"\n>>> Format compliance {pct:.0f}% >= 80%. Starting GRPO automatically...\n")
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
    print(f"\n{'='*60}\nGSPO TRAINING -- {config['model_name']}\n{'='*60}")

    # SFT is saved as a LoRA adapter (to dodge a Gemma 4 + Unsloth config
    # serialization bug). Load base + adapter via Unsloth's FastLanguageModel
    # rather than stock PeftModel.from_pretrained — stock peft doesn't
    # understand Gemma 4's custom Gemma4ClippableLinear wrapper and crashes
    # during adapter re-injection with "Target module ... is not supported".
    #
    # Do NOT merge_and_unload + apply fresh LoRA: merging LoRA into 8-bit
    # base quantizes the delta back to 8-bit with heavy rounding loss, so
    # SFT's learning doesn't survive into GSPO. Instead keep the SFT
    # adapter attached and continue training IT as the GSPO adapter.
    sft_dir = config["grpo_output"] + "_sft"
    sft_adapter_config = os.path.join(sft_dir, "adapter_config.json")
    if os.path.exists(sft_adapter_config):
        print(f"  Loading base + SFT LoRA adapter from {sft_dir} — continuing adapter training for GSPO")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_dir,
            max_seq_length=config["max_seq_length"],
            load_in_4bit=config.get("load_in_4bit", False),
            load_in_8bit=config.get("load_in_8bit", False),
            full_finetuning=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # The adapter is loaded attached and trainable; skip apply_lora.
    else:
        print("  No SFT adapter found — starting GSPO from base Gemma 4.")
        model, tokenizer = load_model_and_tokenizer(config)
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

    # GSPO paper setting: sequence-level importance sampling, vanilla GRPO loss,
    # beta=0, tight clipping range. See https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/gspo-reinforcement-learning
    loss_type = config.get("grpo_loss_type") or "grpo"
    hf_repo = config.get("hf_repo")
    grpo_kwargs = dict(
        output_dir=config["grpo_output"],
        temperature=1.0,
        num_generations=config["grpo_num_generations"],
        max_completion_length=512,
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
        # --- GSPO ---
        importance_sampling_level="sequence",
        loss_type=loss_type,
        beta=0.0,
        epsilon=3e-4,
        epsilon_high=4e-4,
    )
    if hf_repo:
        # Crash-safety: checkpoint the LoRA adapter to HF every save (every 300 steps).
        grpo_kwargs.update(
            push_to_hub=True,
            hub_model_id=hf_repo,
            hub_strategy="every_save",
            hub_private_repo=True,
        )
    grpo_config = GRPOConfig(**grpo_kwargs)

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

    print("\nStarting GSPO (Unsloth inference, sequence-level IS)...")
    print(f"  loss_type={loss_type}  max_completion_length=512")
    print(f"  Curriculum: level 0→9 (easy→hard), advances at {config.get('curriculum_threshold', 0.7):.0%} ratio")
    if hf_repo:
        print(f"  HF auto-push: every save to https://huggingface.co/{hf_repo} (private)")
    trainer.train()
    if use_wandb:
        wandb.finish()
    model.save_pretrained(config["grpo_output"])
    tokenizer.save_pretrained(config["grpo_output"])
    print(f"GSPO adapter saved to {config['grpo_output']}")


# =============================================================================
# EVALUATION
# =============================================================================

def run_eval(config, eval_data):
    print(f"\n{'='*60}\nEVALUATION -- {config['model_size'].upper()}\n{'='*60}")

    checkpoint_dir = config["grpo_output"]
    if not os.path.exists(checkpoint_dir):
        print("ERROR: No model found")
        return

    model, tokenizer = load_model_and_tokenizer(config)
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
        # See run_sft: must pass `text=` as explicit kwarg through Unsloth's
        # patched Gemma 4 Processor call.
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text=rendered, return_tensors="pt").input_ids.to(model.device)

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
    model, tokenizer = load_model_and_tokenizer(config)
    model = PeftModel.from_pretrained(model, config["grpo_output"])
    merged = model.merge_and_unload()
    # Strip any non-serializable callables Unsloth left on the config before
    # transformers tries to json.dumps(config) during save_pretrained.
    _sanitize_config_for_save(merged.config)
    merged.save_pretrained(config["final_model"])
    tokenizer.save_pretrained(config["final_model"])
    print(f"Exported merged model to {config['final_model']}")

    # GGUF (q4_k_m) for llama.cpp / Ollama. save_pretrained_gguf is an Unsloth
    # method patched onto the model at load time.
    gguf_dir = config["final_model"] + "-gguf"
    try:
        merged.save_pretrained_gguf(
            gguf_dir,
            tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"Exported GGUF to {gguf_dir}")
    except Exception as e:
        print(f"WARNING: GGUF export failed ({e}). Merged HF weights still saved to {config['final_model']}.")


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
    gguf_dir = config["final_model"] + "-gguf"

    if os.path.exists(merged_dir):
        print(f"\nPushing merged weights to https://huggingface.co/{hf_repo}")
        create_repo(hf_repo, exist_ok=True)
        api.upload_folder(folder_path=merged_dir, repo_id=hf_repo, commit_message=f"Upload Connect4 agent ({config['model_size']})")
        print(f"  -> https://huggingface.co/{hf_repo}")
    else:
        print(f"WARNING: {merged_dir}/ not found — run --stage export first")

    if os.path.exists(gguf_dir):
        gguf_repo = f"{hf_repo}-GGUF"
        print(f"\nPushing GGUF to https://huggingface.co/{gguf_repo}")
        create_repo(gguf_repo, exist_ok=True)
        api.upload_folder(folder_path=gguf_dir, repo_id=gguf_repo, commit_message=f"Upload Connect4 agent GGUF ({config['model_size']})")
        print(f"  -> https://huggingface.co/{gguf_repo}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Connect Four GSPO Training Pipeline (Gemma 4 + Unsloth)")
    parser.add_argument("--model", choices=["e2b-bf16", "e2b-8bit", "e4b-bf16", "e4b-8bit"], required=True)
    parser.add_argument("--stage", choices=["sft", "grpo", "eval", "export", "push"], default="sft")
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
