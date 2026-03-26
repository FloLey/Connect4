"""
Connect Four LLM Training Pipeline — SFT + GRPO with Unsloth
Usage:
  python connect4_train.py --model {4b,8b,14b} --stage {sft,grpo,both,eval,export,push} --csv connect4_data.csv
  python connect4_train.py --model 8b --stage push --hf-repo yourname/connect4-agent-8b
"""

import argparse, csv, json, os, re, random
from collections import defaultdict
import torch
from datasets import Dataset

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

MODEL_CONFIGS = {
    "4b": {"model_name": "unsloth/Qwen3-4B", "lora_r": 32, "grpo_num_generations": 6, "grpo_batch_size": 4, "grpo_grad_accum": 2, "sft_batch_size": 8},
    "8b": {"model_name": "unsloth/Qwen3-8B", "lora_r": 32, "grpo_num_generations": 4, "grpo_batch_size": 2, "grpo_grad_accum": 4, "sft_batch_size": 4},
    "14b": {"model_name": "unsloth/Qwen3-14B", "lora_r": 16, "grpo_num_generations": 4, "grpo_batch_size": 1, "grpo_grad_accum": 8, "sft_batch_size": 2},
}

def get_config(model_size):
    mc = MODEL_CONFIGS[model_size]
    return {
        "model_name": mc["model_name"], "model_size": model_size, "max_seq_length": 512, "load_in_4bit": True,
        "lora_r": mc["lora_r"], "lora_alpha": mc["lora_r"], "lora_dropout": 0,
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "sft_learning_rate": 2e-4, "sft_num_epochs": 3, "sft_batch_size": mc["sft_batch_size"], "sft_grad_accum": 4, "sft_max_rows": 80_000,
        "grpo_learning_rate": 5e-6, "grpo_max_steps": 1500, "grpo_batch_size": mc["grpo_batch_size"],
        "grpo_grad_accum": mc["grpo_grad_accum"], "grpo_num_generations": mc["grpo_num_generations"],
        "grpo_temperature": 0.7, "grpo_max_rows": 200_000, "csv_path": "connect4_data.csv",
        "sft_output": f"outputs_sft_{model_size}", "grpo_output": f"outputs_grpo_{model_size}",
        "final_model": f"connect4-agent-{model_size}", "wandb_project": "connect4-llm",
    }

SYSTEM_PROMPT = """You are an expert Connect Four player. You analyze board positions and choose the best column to play.

Rules:
- The board has 7 columns (0-6) and 6 rows.
- Players alternate turns dropping pieces into columns.
- A move sequence like "334" means: Player 1 played col 3, Player 2 played col 3, Player 1 played col 4.
- A column is full after 6 pieces have been dropped in it.

When given a position, respond with ONLY the column number (0-6) you want to play. Think about which move gives the best winning chances."""

def load_csv_data(csv_path):
    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({"move_sequence": row["move_sequence"].strip(), "scores": [float(row[f"col{i}"]) for i in range(7)], "best_col": int(row["best_col"])})
    return data

def build_lookup_table(data):
    return {e["move_sequence"]: e["scores"] for e in data}

def format_prompt(move_sequence):
    if move_sequence == "":
        return "The board is empty. It's the first move. Which column do you play?"
    moves_desc = ", ".join(f"{'P1' if i%2==0 else 'P2'} -> col {m}" for i,m in enumerate(move_sequence))
    return f"Move history: {move_sequence}\nMoves played: {moves_desc}\nIt's your turn. Which column do you play?"

def prepare_sft_dataset(data, max_rows, tokenizer):
    random.shuffle(data)
    formatted = []
    for entry in data[:max_rows]:
        conv = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":format_prompt(entry["move_sequence"])},{"role":"assistant","content":str(entry["best_col"])}]
        formatted.append({"text": tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)})
    return Dataset.from_list(formatted)

def prepare_grpo_dataset(data, max_rows, tokenizer):
    random.shuffle(data)
    formatted = []
    for entry in data[:max_rows]:
        conv = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":format_prompt(entry["move_sequence"])}]
        formatted.append({"prompt": tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True), "move_sequence": entry["move_sequence"]})
    return Dataset.from_list(formatted)

SCORE_LOOKUP = {}

def extract_column_from_response(text):
    digits = re.findall(r'[0-6]', text.strip())
    return int(digits[-1]) if digits else None

def reward_format(completions, **kwargs):
    return [0.5 if extract_column_from_response(c) is not None else -1.0 for c in completions]

def reward_move_quality(completions, move_sequence, **kwargs):
    rewards = []
    for c, seq in zip(completions, move_sequence):
        col = extract_column_from_response(c)
        if col is None: rewards.append(-1.0); continue
        scores = SCORE_LOOKUP.get(seq)
        rewards.append(scores[col] if scores else 0.0)
    return rewards

def reward_is_best_move(completions, move_sequence, **kwargs):
    rewards = []
    for c, seq in zip(completions, move_sequence):
        col = extract_column_from_response(c)
        if col is None: rewards.append(-0.5); continue
        scores = SCORE_LOOKUP.get(seq)
        if not scores: rewards.append(0.0); continue
        rewards.append(1.0 if col == max(range(7), key=lambda x: scores[x]) else 0.0)
    return rewards

def run_sft(config):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    print(f"\n{'='*60}\nSTAGE 1: SFT -- {config['model_name']}\n{'='*60}")
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=config["model_name"], max_seq_length=config["max_seq_length"], load_in_4bit=config["load_in_4bit"])
    model = FastLanguageModel.get_peft_model(model, r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"], target_modules=config["target_modules"])
    raw_data = load_csv_data(config["csv_path"])
    print(f"Loaded {len(raw_data)} positions")
    dataset = prepare_sft_dataset(raw_data, config["sft_max_rows"], tokenizer)
    run_name = f"sft-{config['model_size']}"
    if WANDB_AVAILABLE:
        wandb.init(project=config["wandb_project"], name=run_name, tags=[config["model_size"],"sft"], config={"model":config["model_name"],"stage":"sft","lora_r":config["lora_r"]}, reinit=True)
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=SFTConfig(
        output_dir=config["sft_output"], per_device_train_batch_size=config["sft_batch_size"], gradient_accumulation_steps=config["sft_grad_accum"],
        num_train_epochs=config["sft_num_epochs"], learning_rate=config["sft_learning_rate"], lr_scheduler_type="cosine", warmup_ratio=0.1,
        optim="adamw_8bit", fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10, save_steps=500, seed=42, dataset_text_field="text", max_seq_length=config["max_seq_length"],
        packing=True, report_to="wandb" if WANDB_AVAILABLE else "none", run_name=run_name))
    print("\nStarting SFT... Watch at https://wandb.ai -> connect4-llm")
    trainer.train()
    if WANDB_AVAILABLE: wandb.finish()
    model.save_pretrained(config["sft_output"]); tokenizer.save_pretrained(config["sft_output"])
    print(f"SFT saved to {config['sft_output']}")

def run_grpo(config):
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    print(f"\n{'='*60}\nSTAGE 2: GRPO -- {config['model_name']}\n{'='*60}")
    if os.path.exists(config["sft_output"]):
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=config["sft_output"], max_seq_length=config["max_seq_length"], load_in_4bit=config["load_in_4bit"])
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=config["model_name"], max_seq_length=config["max_seq_length"], load_in_4bit=config["load_in_4bit"])
        model = FastLanguageModel.get_peft_model(model, r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"], target_modules=config["target_modules"])
    raw_data = load_csv_data(config["csv_path"])
    global SCORE_LOOKUP; SCORE_LOOKUP = build_lookup_table(raw_data)
    print(f"Loaded {len(raw_data)} positions, {len(SCORE_LOOKUP)} in lookup")
    dataset = prepare_grpo_dataset(raw_data, config["grpo_max_rows"], tokenizer)
    run_name = f"grpo-{config['model_size']}"
    if WANDB_AVAILABLE:
        wandb.init(project=config["wandb_project"], name=run_name, tags=[config["model_size"],"grpo"], config={"model":config["model_name"],"stage":"grpo","num_generations":config["grpo_num_generations"],"temperature":config["grpo_temperature"]}, reinit=True)
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[reward_format, reward_move_quality, reward_is_best_move],
        args=GRPOConfig(output_dir=config["grpo_output"], temperature=config["grpo_temperature"], num_generations=config["grpo_num_generations"],
            max_prompt_length=256, max_completion_length=256, learning_rate=config["grpo_learning_rate"],
            per_device_train_batch_size=config["grpo_batch_size"], gradient_accumulation_steps=config["grpo_grad_accum"],
            max_steps=config["grpo_max_steps"], weight_decay=0.01, warmup_ratio=0.05, lr_scheduler_type="cosine",
            optim="adamw_8bit", logging_steps=1, save_steps=300, report_to="wandb" if WANDB_AVAILABLE else "none", run_name=run_name),
        train_dataset=dataset)
    print("\nStarting GRPO... Watch reward climb at https://wandb.ai -> connect4-llm")
    trainer.train()
    if WANDB_AVAILABLE: wandb.finish()
    model.save_pretrained(config["grpo_output"]); tokenizer.save_pretrained(config["grpo_output"])
    print(f"GRPO saved to {config['grpo_output']}")

def run_eval(config):
    from unsloth import FastLanguageModel
    print(f"\n{'='*60}\nEVALUATION -- {config['model_size'].upper()}\n{'='*60}")
    checkpoint_dir = None
    for d in [config["grpo_output"], config["sft_output"]]:
        if os.path.exists(d): checkpoint_dir = d; break
    if not checkpoint_dir: print("ERROR: No model found"); return
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=checkpoint_dir, max_seq_length=config["max_seq_length"], load_in_4bit=config["load_in_4bit"])
    FastLanguageModel.for_inference(model)
    raw_data = load_csv_data(config["csv_path"]); random.seed(42); random.shuffle(raw_data); eval_data = raw_data[-10_000:]
    print(f"Evaluating on {len(eval_data)} held-out positions...")
    exact=0; top2=0; score_sum=0.0; valid=0; invalid=0
    phase_stats = {p: {"correct":0,"total":0,"score_sum":0.0} for p in ["opening (0-8 moves)","midgame (9-20 moves)","endgame (21+ moves)"]}
    for i, entry in enumerate(eval_data):
        if i%1000==0 and i>0: print(f"  ...{i}/10000 (acc: {100*exact/valid:.1f}%)" if valid else f"  ...{i}/10000")
        seq, scores, best_col = entry["move_sequence"], entry["scores"], entry["best_col"]
        phase = "opening (0-8 moves)" if len(seq)<=8 else "midgame (9-20 moves)" if len(seq)<=20 else "endgame (21+ moves)"
        msgs = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":format_prompt(seq)}]
        inputs = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad(): outputs = model.generate(input_ids=inputs, max_new_tokens=16, temperature=0.01, do_sample=True)
        col = extract_column_from_response(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
        if col is None: invalid+=1; continue
        valid+=1; score_sum+=scores[col]
        if col==best_col: exact+=1; phase_stats[phase]["correct"]+=1
        if col in sorted(range(7), key=lambda c: scores[c], reverse=True)[:2]: top2+=1
        phase_stats[phase]["total"]+=1; phase_stats[phase]["score_sum"]+=scores[col]
    total=valid+invalid
    print(f"\nRESULTS -- {config['model_size'].upper()}")
    print(f"  Valid: {valid}/{total} ({100*valid/total:.1f}%)")
    print(f"  Exact match: {exact}/{valid} ({100*exact/valid:.1f}%)")
    print(f"  Top-2 match: {top2}/{valid} ({100*top2/valid:.1f}%)")
    print(f"  Mean oracle score: {score_sum/valid:+.4f}")
    for phase, s in phase_stats.items():
        if s["total"]>0: print(f"    {phase}: {100*s['correct']/s['total']:.1f}% exact, {s['score_sum']/s['total']:+.3f} avg (n={s['total']})")
    results = {"model":config["model_name"],"model_size":config["model_size"],"exact_match_pct":round(100*exact/valid,2),"top2_match_pct":round(100*top2/valid,2),"mean_oracle_score":round(score_sum/valid,4)}
    with open(f"eval_results_{config['model_size']}.json","w") as f: json.dump(results,f,indent=2)

def export_model(config):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=config["grpo_output"], max_seq_length=config["max_seq_length"], load_in_4bit=config["load_in_4bit"])
    model.save_pretrained_merged(config["final_model"], tokenizer, save_method="merged_16bit")
    model.save_pretrained_gguf(config["final_model"]+"-gguf", tokenizer, quantization_method="q4_k_m")
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
    api = HfApi()
    size = config["model_size"]
    merged_dir = config["final_model"]
    gguf_dir = config["final_model"] + "-gguf"
    # Push merged 16-bit model
    if os.path.exists(merged_dir):
        repo_id = hf_repo
        print(f"\nPushing merged model to https://huggingface.co/{repo_id}")
        create_repo(repo_id, exist_ok=True)
        api.upload_folder(folder_path=merged_dir, repo_id=repo_id, commit_message=f"Upload Connect4 agent ({size}) — merged 16-bit")
        print(f"  -> https://huggingface.co/{repo_id}")
    else:
        print(f"WARNING: {merged_dir}/ not found — run --stage export first")
    # Push GGUF to a separate repo (or same repo with subfolder)
    if os.path.exists(gguf_dir):
        gguf_repo = hf_repo + "-GGUF"
        print(f"\nPushing GGUF to https://huggingface.co/{gguf_repo}")
        create_repo(gguf_repo, exist_ok=True)
        api.upload_folder(folder_path=gguf_dir, repo_id=gguf_repo, commit_message=f"Upload Connect4 agent ({size}) — GGUF q4_k_m")
        print(f"  -> https://huggingface.co/{gguf_repo}")
    else:
        print(f"WARNING: {gguf_dir}/ not found — run --stage export first")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["4b","8b","14b"], required=True)
    parser.add_argument("--stage", choices=["sft","grpo","both","eval","export","push"], default="both")
    parser.add_argument("--csv", default="connect4_data.csv")
    parser.add_argument("--hf-repo", default=None, help="HuggingFace repo id for push (e.g. yourname/connect4-agent-8b)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.no_wandb: global WANDB_AVAILABLE; WANDB_AVAILABLE = False
    config = get_config(args.model); config["csv_path"] = args.csv; config["hf_repo"] = args.hf_repo
    print(f"\nConnect Four Pipeline | Model: {config['model_name']} | Stage: {args.stage} | Wandb: {'on' if WANDB_AVAILABLE else 'off'}")
    if args.stage in ("sft","both"): run_sft(config)
    if args.stage in ("grpo","both"): run_grpo(config)
    if args.stage == "eval": run_eval(config)
    if args.stage == "export": export_model(config)
    if args.stage == "push": push_to_hub(config)

if __name__ == "__main__":
    main()
