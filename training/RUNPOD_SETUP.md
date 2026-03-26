# RunPod Setup — Connect Four LLM Training

Train 3 model sizes in parallel on RunPod, with live Weights & Biases monitoring.

## Pod Specs

| Pod | Model | GPU | Est. Cost/hr | Est. Time | Est. Total |
|-----|-------|-----|--------------|-----------|------------|
| Pod 1 | Qwen3-4B | RTX 4090 (24 GB) | ~$0.50/hr | ~2h | ~$1.00 |
| Pod 2 | Qwen3-8B | RTX 4090 (24 GB) | ~$0.50/hr | ~3h | ~$1.50 |
| Pod 3 | Qwen3-14B | A100 40GB | ~$1.50/hr | ~3.5h | ~$5.25 |

**Total estimated cost: ~$8**

---

## Step 0: Weights & Biases Setup

1. Sign up at [wandb.ai](https://wandb.ai) (free tier is fine).
2. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize).
3. Keep the key handy — you'll paste it on each pod.

---

## Step 1: Pod Setup

On **each pod**, run:

```bash
# Install dependencies
pip install unsloth wandb diffusers huggingface_hub

# Login to wandb (paste your API key when prompted)
wandb login

# Login to HuggingFace (paste your token when prompted)
huggingface-cli login

# Upload your files (or git clone your repo)
# Make sure these are in the working directory:
#   - connect4_train.py
#   - connect4_data.csv
```

---

## Step 2: Launch Training

Run one command per pod. Each runs SFT then GRPO automatically:

**Pod 1 (4B):**
```bash
python connect4_train.py --model 4b --stage both --csv connect4_data.csv
```

**Pod 2 (8B):**
```bash
python connect4_train.py --model 8b --stage both --csv connect4_data.csv
```

**Pod 3 (14B):**
```bash
python connect4_train.py --model 14b --stage both --csv connect4_data.csv
```

All three will stream metrics to the same wandb project.

---

## Step 3: Watch Live

Go to [wandb.ai](https://wandb.ai) and open project **"connect4-llm"**.

### What to watch

**During SFT** (the boring part):
- `train/loss` should decrease steadily. That's it.

**During GRPO** (the good stuff):
- `reward` — aggregate reward, should climb starting around step 300.
- `reward_move_quality` — the oracle score ([-1, +1]). This is the metric that matters most. Higher = the model is picking better moves.
- `reward_is_best_move` — binary bonus. Tracks how often the model picks the single best column.
- `reward_format` — should quickly saturate near 0.5 (the model learns to output a digit fast).
- `reward_std` — should decrease over time as the model becomes more consistent.

### Compare all 3 models

1. In the wandb dashboard, select all runs (sft-4b, grpo-4b, sft-8b, grpo-8b, etc.).
2. Use the "Group by" → tag to compare by model size.
3. Create a custom chart panel overlaying `reward_move_quality` for grpo-4b, grpo-8b, grpo-14b.
4. The model where this curve is highest and most stable is your winner.

---

## Step 4: Evaluate

After training completes on each pod:

```bash
# Pod 1
python connect4_train.py --model 4b --stage eval --csv connect4_data.csv

# Pod 2
python connect4_train.py --model 8b --stage eval --csv connect4_data.csv

# Pod 3
python connect4_train.py --model 14b --stage eval --csv connect4_data.csv
```

This evaluates on 10K held-out positions and saves results to `eval_results_{size}.json`.

Download the JSON files and compare:
- **Exact match %** — how often it picks the oracle's best move
- **Top-2 match %** — picked one of the two best moves
- **Mean oracle score** — average quality of chosen moves
- **Per-phase breakdown** — opening / midgame / endgame performance

---

## Step 5: Export the Winner

On the pod with the best eval results:

```bash
python connect4_train.py --model 8b --stage export --csv connect4_data.csv
```

This creates:
- `connect4-agent-{size}/` — merged HuggingFace model (16-bit)
- `connect4-agent-{size}-gguf/` — quantized GGUF (q4_k_m) for llama.cpp / Ollama

---

## Step 6: Push to HuggingFace

After exporting, push to HuggingFace Hub so you can reuse the model anywhere:

```bash
python connect4_train.py --model 8b --stage push --hf-repo yourname/connect4-agent-8b
```

This uploads two repos:
- `yourname/connect4-agent-8b` — the merged 16-bit model (use with `transformers` / vLLM)
- `yourname/connect4-agent-8b-GGUF` — the quantized GGUF (use with llama.cpp / Ollama)

To use later:
```python
# With transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("yourname/connect4-agent-8b")

# With Ollama
# Download the GGUF file from the -GGUF repo, then:
# ollama create connect4 -f Modelfile
```

---

## Troubleshooting

### OOM (Out of Memory)

If you get CUDA OOM errors:
- **4B/8B on 4090:** Reduce `grpo_num_generations` (edit the `MODEL_CONFIGS` dict in the script). Try 4 for 4B, 2 for 8B.
- **14B:** Upgrade to an A100 80GB or H100. Or reduce `grpo_num_generations` to 2.

### Reward stays flat during GRPO

- Lower temperature (try 0.5 instead of 0.7) — the model may be generating too randomly.
- Increase `grpo_grad_accum` to 8 or 16 for more stable gradients.
- Try the DR-GRPO loss variant: `--loss-type dr_grpo`.
- Make sure SFT completed successfully first — GRPO needs a good starting point.

### Pod disconnects mid-training

- Checkpoints are saved every 300/500 steps in `outputs_sft_{size}/` and `outputs_grpo_{size}/`.
- The trainers will auto-resume from the latest checkpoint if you re-run the same command.
- For extra safety, use RunPod's persistent storage (network volume) to keep checkpoints across pod restarts.

### Budget saver tip

Skip the 14B model and just race 4B vs 8B:
- **2 pods × RTX 4090 × ~2.5h avg = ~$2.50 total**
- The 4B model often surprises — it's fast to train and iterates quickly.
- If 8B clearly wins, you can always try 14B later.
