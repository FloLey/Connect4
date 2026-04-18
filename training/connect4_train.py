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


# ============================================================================
# Format probe — generate greedy completions on held-out positions and
# count how many parse to a single digit 0-6. Used as the SFT early-stop
# gate.
# ============================================================================
def probe_format(model, tokenizer, eval_data, n=50):
    from unsloth import FastLanguageModel
    import torch

    FastLanguageModel.for_inference(model)
    ok = 0
    try:
        for entry in eval_data[:n]:
            p = build_prompt(tokenizer, entry)
            inp = tokenizer(p, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inp, max_new_tokens=512, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            comp = tokenizer.decode(out[0][inp.input_ids.shape[1]:],
                                    skip_special_tokens=True)
            if parse_answer(comp) is not None:
                ok += 1
    finally:
        FastLanguageModel.for_training(model)
    return ok / max(n, 1)


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


# ============================================================================
# SFT — minimal digit-only warmup.
# ============================================================================
def run_sft(config, train_sorted, eval_data, model, tokenizer, hf_repo=None):
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    rows = []
    for entry in train_sorted[:config["sft_examples"]]:
        rows.append({
            "prompt": build_prompt(tokenizer, entry),
            "completion": str(entry["best_col"]),
        })
    ds = Dataset.from_list(rows)
    print(f"SFT dataset: {len(ds)} examples (target = single digit)")

    class ProbeEarlyStop(TrainerCallback):
        def __init__(self, every=10, n=50, threshold=0.90):
            self.every = every
            self.n = n
            self.threshold = threshold
            self.best = 0.0

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 0 or state.global_step % self.every != 0:
                return
            rate = probe_format(model, tokenizer, eval_data, n=self.n)
            self.best = max(self.best, rate)
            print(f"  [probe step={state.global_step}] compliance={rate:.0%}")
            if rate >= self.threshold:
                print(f"  [probe] gate passed, early-stopping SFT")
                control.should_training_stop = True

    use_wandb = os.environ.get("WANDB_DISABLED", "").lower() != "true"
    sft_args = SFTConfig(
        output_dir=config["sft_output_dir"],
        per_device_train_batch_size=config["sft_batch_size"],
        gradient_accumulation_steps=config["sft_grad_accum"],
        learning_rate=config["sft_lr"],
        max_steps=config["sft_max_steps"],
        warmup_steps=2,
        lr_scheduler_type="constant",
        logging_steps=1,
        save_strategy="no",
        report_to=["wandb"] if use_wandb else "none",
        run_name=f"sft-{config['variant']}",
        bf16=True,
        completion_only_loss=True,
        max_length=config["max_seq_length"],
    )

    probe_cb = ProbeEarlyStop(every=10, n=50, threshold=0.90)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        processing_class=tokenizer,
        callbacks=[probe_cb],
    )
    trainer.train()

    sft_dir = config["sft_output_dir"]
    trainer.save_model(sft_dir)
    tokenizer.save_pretrained(sft_dir)
    print(f"SFT adapter saved to {sft_dir}")

    if hf_repo:
        sft_repo = f"{hf_repo}-sft"
        print(f"Pushing SFT adapter to https://huggingface.co/{sft_repo}")
        model.push_to_hub(sft_repo, private=True)
        tokenizer.push_to_hub(sft_repo, private=True)

    final_rate = probe_format(model, tokenizer, eval_data, n=50)
    print(f"final format compliance: {final_rate:.0%}")
    return final_rate


# ============================================================================
# GSPO — sequence-level RL with curriculum sampling.
# ============================================================================
def run_grpo(config, train_sorted, eval_data, model, tokenizer, hf_repo=None, max_steps=None):
    from datasets import IterableDataset
    from trl import GRPOConfig, GRPOTrainer

    n_steps = max_steps or config["grpo_max_steps"]
    sampler = GaussianCurriculumSampler(n_items=len(train_sorted))
    # Prime the column-score max so advance_threshold (max_reward * 0.7)
    # matches reward_format(+5) + reward_move_quality(up to +10) = +15.
    sampler.advance_threshold = 15.0 * 0.7

    def _gen():
        # Emit enough items for the whole run with slack.
        n_items = n_steps * config["grpo_batch_size"] * config["grpo_grad_accum"] * 4
        for _ in range(n_items):
            idx, bucket = sampler.sample()
            entry = train_sorted[idx]
            g = reconstruct_board(entry["move_sequence"])
            yield {
                "prompt": build_prompt(tokenizer, entry),
                "scores": entry["scores"],
                "valid_cols": g.get_valid_moves(),
                "bucket": bucket,
            }

    ds = IterableDataset.from_generator(_gen)

    def rf_format(completions, **kwargs):
        return [reward_format(c) for c in completions]

    def rf_quality(completions, scores, valid_cols, bucket, **kwargs):
        rewards = []
        for c, s, v, b in zip(completions, scores, valid_cols, bucket):
            r = reward_move_quality(c, s, v)
            rewards.append(r)
            # Pair with a format-reward estimate so curriculum
            # EMA reflects the full reward budget.
            sampler.update(b, r + reward_format(c))
        return rewards

    use_wandb = os.environ.get("WANDB_DISABLED", "").lower() != "true"

    grpo_kwargs = dict(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["grpo_batch_size"],
        gradient_accumulation_steps=config["grpo_grad_accum"],
        num_generations=config["grpo_num_generations"],
        temperature=config["grpo_temperature"],
        max_completion_length=config["grpo_max_completion_length"],
        max_prompt_length=config["grpo_max_prompt_length"],
        learning_rate=config["grpo_lr"],
        loss_type=config["grpo_loss_type"],
        beta=config["grpo_beta"],
        epsilon=config["grpo_epsilon"],
        epsilon_high=config["grpo_epsilon_high"],
        importance_sampling_level="sequence",
        max_steps=n_steps,
        save_steps=config["grpo_save_steps"],
        save_strategy="steps",
        warmup_steps=10,
        lr_scheduler_type="constant",
        logging_steps=1,
        report_to=["wandb"] if use_wandb else "none",
        run_name=f"grpo-{config['variant']}",
        bf16=True,
        use_vllm=False,
        log_completions=True,
        optim="adamw_8bit",
        max_grad_norm=0.1,
    )
    if hf_repo:
        grpo_kwargs.update(
            push_to_hub=True,
            hub_model_id=hf_repo,
            hub_strategy="every_save",
            hub_private_repo=True,
        )
    grpo_args = GRPOConfig(**grpo_kwargs)

    thinking_cb = ThinkingLogger(log_every=25).as_callback()

    class CurriculumMetrics:
        """Push curriculum level + per-bucket EMA to wandb every log step."""
        def __init__(self, sampler):
            self.sampler = sampler

        def as_callback(self):
            from transformers import TrainerCallback
            outer = self

            class _CB(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kw):
                    if logs is None:
                        return
                    logs["curriculum/level"] = outer.sampler.center
                    for i, ema in enumerate(outer.sampler.bucket_reward_ema):
                        logs[f"curriculum/bucket_{i}_ema"] = ema

            return _CB()

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=ds,
        processing_class=tokenizer,
        reward_funcs=[rf_format, rf_quality],
        callbacks=[thinking_cb, CurriculumMetrics(sampler).as_callback()],
    )

    print(f"\nStarting GSPO: max_steps={n_steps}, K={config['grpo_num_generations']}, "
          f"loss={config['grpo_loss_type']}, curriculum advance_threshold={sampler.advance_threshold:.1f}")
    if hf_repo:
        print(f"HF auto-push every {config['grpo_save_steps']} steps -> https://huggingface.co/{hf_repo}")
    trainer.train()
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"GSPO adapter saved to {config['output_dir']}")


# ============================================================================
# Eval — greedy decode on the 10k held-out positions.
# ============================================================================
def run_eval(config, eval_data, model, tokenizer):
    from unsloth import FastLanguageModel
    import torch

    FastLanguageModel.for_inference(model)

    agg = {
        "n": 0, "valid": 0, "exact": 0, "top2": 0,
        "score_sum": 0.0,
        "phase": {
            "opening (0-8)": {"n": 0, "exact": 0, "top2": 0, "score_sum": 0.0},
            "midgame (9-20)": {"n": 0, "exact": 0, "top2": 0, "score_sum": 0.0},
            "endgame (21+)": {"n": 0, "exact": 0, "top2": 0, "score_sum": 0.0},
        },
    }

    for i, entry in enumerate(eval_data):
        if i and i % 1000 == 0:
            rate = agg["exact"] / max(agg["valid"], 1)
            print(f"  ...{i}/{len(eval_data)}  exact={rate:.1%}")
        p = build_prompt(tokenizer, entry)
        inp = tokenizer(p, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=1024, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        comp = tokenizer.decode(out[0][inp.input_ids.shape[1]:],
                                skip_special_tokens=True)
        col = parse_answer(comp)

        scores = entry["scores"]
        best = entry["best_col"]
        top2 = set(sorted(range(7), key=lambda c: -scores[c])[:2])
        nmoves = len(entry["move_sequence"])
        phase = ("opening (0-8)" if nmoves <= 8 else
                 "midgame (9-20)" if nmoves <= 20 else "endgame (21+)")

        agg["n"] += 1
        agg["phase"][phase]["n"] += 1
        if col is None or not (0 <= col < 7):
            continue
        agg["valid"] += 1
        agg["score_sum"] += scores[col]
        agg["phase"][phase]["score_sum"] += scores[col]
        if col == best:
            agg["exact"] += 1
            agg["phase"][phase]["exact"] += 1
        if col in top2:
            agg["top2"] += 1
            agg["phase"][phase]["top2"] += 1

    n = agg["n"]
    v = max(agg["valid"], 1)
    print(f"\n=== EVAL {config['variant']} on n={n} ===")
    print(f"  valid          : {agg['valid']/n:.1%}")
    print(f"  exact (of valid): {agg['exact']/v:.1%}")
    print(f"  top-2 (of valid): {agg['top2']/v:.1%}")
    print(f"  mean oracle    : {agg['score_sum']/v:+.3f}")
    for ph, s in agg["phase"].items():
        if s["n"] == 0:
            continue
        vv = max(s["n"], 1)
        print(f"    {ph:18}: exact={s['exact']/vv:.1%} top2={s['top2']/vv:.1%} "
              f"score={s['score_sum']/vv:+.3f} (n={s['n']})")

    out_path = f"eval_results_{config['variant']}.json"
    with open(out_path, "w") as f:
        json.dump({
            "variant": config["variant"],
            "model": config["model_name"],
            "n": n,
            "valid_pct": round(100 * agg["valid"] / n, 2),
            "exact_pct": round(100 * agg["exact"] / v, 2),
            "top2_pct": round(100 * agg["top2"] / v, 2),
            "mean_oracle": round(agg["score_sum"] / v, 4),
            "phase": agg["phase"],
        }, f, indent=2)
    print(f"wrote {out_path}")


# ============================================================================
# Export — merge the GSPO LoRA into the base weights and produce a GGUF.
# ============================================================================
def run_export(config):
    from peft import PeftModel
    model, tokenizer = load_model_and_tokenizer(config)
    ckpt = config["output_dir"]
    if not os.path.isdir(ckpt):
        raise FileNotFoundError(f"No GSPO adapter at {ckpt}")
    model = PeftModel.from_pretrained(model, ckpt)
    merged_dir = config["final_model_dir"]
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged 16-bit model -> {merged_dir}")

    gguf_dir = merged_dir + "-gguf"
    try:
        merged.save_pretrained_gguf(gguf_dir, tokenizer,
                                    quantization_method="q4_k_m")
        print(f"GGUF (q4_k_m) -> {gguf_dir}")
    except Exception as e:
        print(f"WARNING: GGUF export failed ({e}); merged weights still saved")


def run_push(config, hf_repo):
    from huggingface_hub import HfApi, create_repo
    if not hf_repo:
        raise ValueError("--hf-repo is required for push stage")
    api = HfApi()
    merged_dir = config["final_model_dir"]
    gguf_dir = merged_dir + "-gguf"
    if os.path.isdir(merged_dir):
        create_repo(hf_repo, exist_ok=True, private=True)
        api.upload_folder(folder_path=merged_dir, repo_id=hf_repo,
                          commit_message=f"merged {config['variant']}")
        print(f"-> https://huggingface.co/{hf_repo}")
    else:
        print(f"skip merged: {merged_dir}/ not found (run --stage export first)")
    if os.path.isdir(gguf_dir):
        gguf_repo = f"{hf_repo}-GGUF"
        create_repo(gguf_repo, exist_ok=True, private=True)
        api.upload_folder(folder_path=gguf_dir, repo_id=gguf_repo,
                          commit_message=f"gguf {config['variant']}")
        print(f"-> https://huggingface.co/{gguf_repo}")
    else:
        print(f"skip gguf: {gguf_dir}/ not found")


def _load_sft_adapter(model, hf_repo):
    """Pull SFT adapter from HF and load it onto the current model."""
    from huggingface_hub import snapshot_download
    sft_repo = f"{hf_repo}-sft"
    local = f"outputs_sft_{hf_repo.split('/')[-1]}"
    if not os.path.isdir(local):
        snapshot_download(repo_id=sft_repo, local_dir=local)
    model.load_adapter(local, adapter_name="default")
    print(f"Loaded SFT adapter from {sft_repo}")


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

    if args.stage == "sft":
        raw = load_csv_data(args.csv)
        train, eval_ = split_data(raw)
        model, tok = load_model_and_tokenizer(config)
        model = apply_lora(model, config)
        rate = run_sft(config, train, eval_, model, tok, hf_repo=args.hf_repo)
        if rate >= 0.80:
            print("\n>>> format gate passed — chaining to GSPO\n")
            run_grpo(config, train, eval_, model, tok,
                     hf_repo=args.hf_repo,
                     max_steps=int(os.environ.get("MAX_STEPS", "0")) or None)
        else:
            print(f"\n>>> format gate failed ({rate:.0%} < 80%) — extend SFT or revisit prompt")
        return

    if args.stage == "grpo":
        raw = load_csv_data(args.csv)
        train, eval_ = split_data(raw)
        model, tok = load_model_and_tokenizer(config)
        model = apply_lora(model, config)
        if args.sft and args.hf_repo:
            _load_sft_adapter(model, args.hf_repo)
        run_grpo(config, train, eval_, model, tok,
                 hf_repo=args.hf_repo,
                 max_steps=int(os.environ.get("MAX_STEPS", "0")) or None)
        return

    if args.stage == "eval":
        from peft import PeftModel
        raw = load_csv_data(args.csv)
        _, eval_ = split_data(raw)
        model, tok = load_model_and_tokenizer(config)
        ckpt = config["output_dir"]
        if os.path.isdir(ckpt):
            print(f"Loading GSPO adapter from {ckpt}")
            model = PeftModel.from_pretrained(model, ckpt)
        else:
            print(f"No adapter at {ckpt} — evaluating base model")
        run_eval(config, eval_, model, tok)
        return

    if args.stage == "export":
        run_export(config)
        return
    if args.stage == "push":
        run_push(config, args.hf_repo)
        return
    raise NotImplementedError(args.stage)


if __name__ == "__main__":
    main()
