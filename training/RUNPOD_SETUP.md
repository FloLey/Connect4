# RunPod Setup — Connect Four GRPO Training (Gemma 4)

Train Gemma 4 models on RunPod with live Weights & Biases monitoring.
No SFT needed — Gemma 4 already knows JSON and has native thinking (`<|think|>`). All budget goes to GRPO.

## Pod Specs

| Pod | Model | Precision | GPU | VRAM Used | Est. Cost/hr | Est. Time | Est. Total |
|-----|-------|-----------|-----|-----------|--------------|-----------|------------|
| Pod 1 | gemma-4-E2B-it | BF16 | RTX 4090 (24 GB) | ~15 GB | ~$0.50/hr | ~2h | ~$1.00 |
| Pod 2 | gemma-4-E2B-it | 8-bit | RTX 4090 (24 GB) | ~9 GB | ~$0.50/hr | ~2h | ~$1.00 |
| Pod 3 | gemma-4-E4B-it | BF16 | RTX 4090 (24 GB) | ~20 GB | ~$0.50/hr | ~3h | ~$1.50 |
| Pod 4 | gemma-4-E4B-it | 8-bit | RTX 4090 (24 GB) | ~12 GB | ~$0.50/hr | ~3h | ~$1.50 |

**Total estimated cost: ~$5**

---

## Step 0: Weights & Biases Setup

1. Sign up at [wandb.ai](https://wandb.ai) (free tier is fine).
2. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize).
3. Keep the key handy — you'll paste it on each pod.

---

## Step 1: Pod Setup

**Template:** Use `unsloth/unsloth:latest` on RunPod (has all dependencies pre-installed).

On **each pod**, run:

```bash
# Clone the repo
git clone https://github.com/FloLey/Connect4.git
cd Connect4/training

# Install wandb (only extra dependency needed)
pip install wandb

# Login to wandb (paste your API key when prompted)
wandb login
```

---

## Step 2: Launch Training

One command per pod — GRPO directly from the base model:

**Pod 1 (E2B BF16):**
```bash
python connect4_train.py --model e2b-bf16 --stage grpo --csv connect4_data.csv
```

**Pod 2 (E2B 8-bit):**
```bash
python connect4_train.py --model e2b-8bit --stage grpo --csv connect4_data.csv
```

**Pod 3 (E4B BF16):**
```bash
python connect4_train.py --model e4b-bf16 --stage grpo --csv connect4_data.csv
```

**Pod 4 (E4B 8-bit):**
```bash
python connect4_train.py --model e4b-8bit --stage grpo --csv connect4_data.csv
```

All four will stream metrics to the same wandb project.

---

## Step 3: Watch Live

Go to [wandb.ai](https://wandb.ai) and open project **"connect4-llm"**.

### What to watch

- `reward` — aggregate reward, should climb starting around step 200-400.
- `reward_move_quality` — the oracle score scaled to **[-10, +10]**. **This is the metric that matters most.** Higher = the model is picking better moves.
- `reward_format` — should climb toward +1.0 as the model learns to output valid JSON. If it stays at -10.0, the model is failing to produce parseable JSON.
- `curriculum_level` �� current difficulty level (0=easy, 9=hard). Watch it climb as the model improves.
- `curriculum_ratio` — reward obtained / max possible. When this hits 0.7, the level advances.

**Expected pattern:** reward climbs → level advances → reward drops (harder positions) → reward climbs back → repeat. This sawtooth pattern is healthy — it means the model is progressively mastering harder positions.

### Compare all 4 models

1. In the wandb dashboard, select all grpo runs.
2. Use the "Group by" -> tag to compare by model variant.
3. Create a custom chart panel overlaying `reward_move_quality` for all 4 runs.
4. The model where this curve is highest and most stable is your winner.

---

## Step 4: Evaluate

After training completes on each pod:

```bash
python connect4_train.py --model e2b-bf16 --stage eval --csv connect4_data.csv
python connect4_train.py --model e2b-8bit --stage eval --csv connect4_data.csv
python connect4_train.py --model e4b-bf16 --stage eval --csv connect4_data.csv
python connect4_train.py --model e4b-8bit --stage eval --csv connect4_data.csv
```

This evaluates on 10K held-out positions and saves results to `eval_results_{variant}.json`.

Download the JSON files and compare:
- **Valid output %** — parsed a column from the response
- **JSON format %** — produced proper `{"column": N}` JSON
- **Exact match %** — how often it picks the oracle's best move
- **Top-2 match %** — picked one of the two best moves
- **Mean oracle score** — average quality of chosen moves
- **Per-phase breakdown** — opening / midgame / endgame performance

---

## Step 5: Export the Winner

On the pod with the best eval results:

```bash
python connect4_train.py --model e4b-bf16 --stage export --csv connect4_data.csv
```

This creates:
- `connect4-agent-{variant}/` — merged HuggingFace model (16-bit)
- `connect4-agent-{variant}-gguf/` — quantized GGUF (q4_k_m) for llama.cpp / Ollama

---

## Step 6: Push to HuggingFace

After exporting, push to HuggingFace Hub so you can reuse the model anywhere:

```bash
python connect4_train.py --model e4b-bf16 --stage push --hf-repo yourname/connect4-agent-e4b-bf16
```

This uploads two repos:
- `yourname/connect4-agent-e4b-bf16` — the merged 16-bit model (use with `transformers` / vLLM)
- `yourname/connect4-agent-e4b-bf16-GGUF` — the quantized GGUF (use with llama.cpp / Ollama)

---

## Troubleshooting

### OOM (Out of Memory)

All 4 variants should fit on a 24 GB RTX 4090. If you get CUDA OOM:
- Reduce `grpo_num_generations` in `MODEL_CONFIGS` (e.g. from 4 to 3 for E2B, from 3 to 2 for E4B).
- Reduce `grpo_batch_size` to 1.
- The 8-bit variants use significantly less VRAM — try those first if BF16 is tight.

### Reward stays flat during GRPO

- Lower temperature (try 0.5 instead of 0.7) — the model may be generating too randomly.
- Increase `grpo_grad_accum` to 8 or 16 for more stable gradients.
- Try the DR-GRPO loss variant: `--loss-type dr_grpo`.

### reward_format stays at -10.0

If the model can't produce parseable JSON after ~100 steps, it may need a lightweight SFT warmup to learn the JSON format. Add a small SFT stage (~5K examples, 1 epoch) before GRPO.

### Pod disconnects mid-training

- Checkpoints are saved every 300 steps in `outputs_grpo_{variant}/`.
- The trainer will auto-resume from the latest checkpoint if you re-run the same command.
- For extra safety, use RunPod's persistent storage (network volume) to keep checkpoints across pod restarts.

### Budget saver tip

Skip the BF16 variants and just race E2B-8bit vs E4B-8bit:
- **2 pods x RTX 4090 x ~2.5h avg = ~$2.50 total**
- If E4B-8bit clearly wins, you can always try BF16 later to see if precision matters.
