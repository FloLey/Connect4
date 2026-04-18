"""Connect 4 GSPO training — Gemma 4 + Unsloth + TRL 0.28.

Pipeline: SFT (minimal digit-only warmup) -> GSPO (curriculum RL) -> eval -> export.

Usage:
  python connect4_train.py --model e4b-8bit --stage sft  --csv connect4_data.csv --hf-repo Betha/connect4-agent-e4b-8bit
  python connect4_train.py --model e4b-8bit --stage grpo --sft --csv connect4_data.csv --hf-repo Betha/connect4-agent-e4b-8bit
  python connect4_train.py --model e4b-8bit --stage eval --csv connect4_data.csv
"""
import argparse
import json
import os

# unsloth must be imported before transformers — injects Gemma-4 RL patches.
import unsloth  # noqa: F401

ROWS = 6
COLS = 7

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
