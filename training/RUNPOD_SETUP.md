# RunPod Setup — Connect Four GSPO Training (Gemma 4 + Unsloth)

Train Gemma 4 models on RunPod with live Weights & Biases monitoring, plus automatic per-checkpoint pushes to Hugging Face so a pod dying mid-training doesn't cost you anything.
No SFT needed — Gemma 4 has native thinking (`<think>`) and learns the single-digit answer format within a few hundred GSPO steps. All budget goes to GSPO.

## Pod Specs

| Pod | Model | Precision | GPU | VRAM Used | Est. Cost/hr | Est. Time | Est. Total |
|-----|-------|-----------|-----|-----------|--------------|-----------|------------|
| Pod 1 | gemma-4-E2B-it | BF16 | RTX 4090 (24 GB) | ~15 GB | ~$0.50/hr | ~2h | ~$1.00 |
| Pod 2 | gemma-4-E2B-it | 8-bit | RTX 4090 (24 GB) | ~9 GB | ~$0.50/hr | ~2h | ~$1.00 |
| Pod 3 | gemma-4-E4B-it | BF16 | RTX 4090 (24 GB) | ~20 GB | ~$0.50/hr | ~3h | ~$1.50 |
| Pod 4 | gemma-4-E4B-it | 8-bit | RTX 4090 (24 GB) | ~12 GB | ~$0.50/hr | ~3h | ~$1.50 |

**Total estimated cost: ~$5**

---

## Step 0: Weights & Biases + Hugging Face Setup

1. Sign up at [wandb.ai](https://wandb.ai) and grab your key at [wandb.ai/authorize](https://wandb.ai/authorize).
2. Sign up at [huggingface.co](https://huggingface.co) and create a **write** token at [Settings → Access Tokens](https://huggingface.co/settings/tokens). Pick your repo name pattern up front, e.g. `Betha/connect4-agent-e4b-8bit` — one repo per variant.
3. Keep both keys handy — you'll paste them on each pod.

---

## Step 1: Pod Setup

**Template:** Use `unsloth/unsloth:latest` on RunPod (has all dependencies pre-installed).

On **each pod**, run:

```bash
# Clone the repo
git clone https://github.com/FloLey/Connect4.git
cd Connect4/training

# Install the two extras not in the base image.
# Pin huggingface_hub < 1.0 — transformers 4.57 in the unsloth image is
# incompatible with the 1.x release.
pip install wandb hf_transfer 'huggingface_hub<1.0'

# Login to wandb (paste API key when prompted)
wandb login

# Login to HF (paste write token when prompted) — required for auto-push.
# The new CLI is `hf`; `huggingface-cli` is deprecated.
hf auth login
```

---

## Step 2: Launch Training

One command per pod. Passing `--hf-repo` makes the trainer push the LoRA adapter to Hugging Face every 300 steps (private repo by default) — if the pod dies you won't lose more than one checkpoint window of progress.

**Pod 1 (E2B BF16):**
```bash
python connect4_train.py --model e2b-bf16 --stage grpo --csv connect4_data.csv \
    --hf-repo Betha/connect4-agent-e2b-bf16
```

**Pod 2 (E2B 8-bit):**
```bash
python connect4_train.py --model e2b-8bit --stage grpo --csv connect4_data.csv \
    --hf-repo Betha/connect4-agent-e2b-8bit
```

**Pod 3 (E4B BF16):**
```bash
python connect4_train.py --model e4b-bf16 --stage grpo --csv connect4_data.csv \
    --hf-repo Betha/connect4-agent-e4b-bf16
```

**Pod 4 (E4B 8-bit):**
```bash
python connect4_train.py --model e4b-8bit --stage grpo --csv connect4_data.csv \
    --hf-repo Betha/connect4-agent-e4b-8bit
```

All four stream metrics to the same wandb project and their LoRA adapters land in four separate HF repos.

---

## Step 3: Watch Live

Go to [wandb.ai](https://wandb.ai) and open project **"connect4-llm"**.

### What to watch

- `reward` — aggregate reward, should climb starting around step 200-400.
- `reward_move_quality` — the oracle score scaled to **[-10, +10]**. **This is the metric that matters most.** Higher = the model is picking better moves.
- `reward_format` — should climb toward +1.0 as the model learns to output a bare single digit `0-6` (optionally wrapped in `<think>...</think>`). If it stays at -10.0, the model isn't producing a clean digit answer.
- `curriculum_level` — current difficulty level (0=easy, 9=hard). Watch it climb as the model improves.
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
- **Valid output %** — parsed a column from the response (single digit 0-6)
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

After exporting, push the merged weights + GGUF to HuggingFace so you can reuse the model anywhere (the LoRA adapter has been auto-pushed during training already; this publishes the *merged* final model):

```bash
python connect4_train.py --model e4b-bf16 --stage push --hf-repo Betha/connect4-agent-e4b-bf16
```

This uploads two repos:
- `Betha/connect4-agent-e4b-bf16` — the merged 16-bit model (use with `transformers` / vLLM)
- `Betha/connect4-agent-e4b-bf16-GGUF` — the quantized GGUF (use with llama.cpp / Ollama)

---

## Step 7: Resume after pod death

If a pod disconnects mid-training, spin up a fresh RTX 4090 with `unsloth/unsloth:latest`, redo Step 1, then pull the last checkpoint from HF back into the local output directory before relaunching:

```bash
# Example: resume e4b-8bit
huggingface-cli download Betha/connect4-agent-e4b-8bit \
    --local-dir outputs_grpo_e4b-8bit

# Re-run the exact same training command — TRL auto-resumes from the checkpoint
python connect4_train.py --model e4b-8bit --stage grpo --csv connect4_data.csv \
    --hf-repo Betha/connect4-agent-e4b-8bit
```

`outputs_grpo_{variant}/` must contain a `checkpoint-*/` subdirectory for TRL to resume; the HF repo preserves it via `hub_strategy="every_save"`.

---

## Troubleshooting

### OOM (Out of Memory)

All 4 variants should fit on a 24 GB RTX 4090. If you get CUDA OOM:
- Reduce `grpo_num_generations` in `MODEL_CONFIGS` (e.g. from 4 to 3 for E2B, from 3 to 2 for E4B).
- Reduce `grpo_batch_size` to 1.
- The 8-bit variants use significantly less VRAM — try those first if BF16 is tight.

### Reward stays flat during GSPO

- Lower temperature (try 0.5 instead of 0.7) — the model may be generating too randomly.
- Increase `grpo_grad_accum` to 8 or 16 for more stable gradients.
- Try the DR-GRPO loss variant: `--loss-type dr_grpo`.

### reward_format stays at -10.0

If the model can't produce a clean single-digit answer after ~200 steps, run a short SFT warmup (~1K examples, 50 steps) before GSPO:
```bash
python connect4_train.py --model e4b-8bit --stage sft --csv connect4_data.csv \
    --hf-repo Betha/connect4-agent-e4b-8bit
```
`--stage sft` runs SFT, verifies ≥80% format compliance, then kicks off GSPO automatically.

### Pod disconnects mid-training

- Checkpoints are saved every 300 steps in `outputs_grpo_{variant}/` **and** pushed to `Betha/connect4-agent-{variant}` on Hugging Face (when `--hf-repo` is set).
- See **Step 7** below for the exact resume recipe on a fresh pod.
- For extra safety, you can also attach a RunPod network volume so the local `outputs_grpo_{variant}/` persists across pod restarts — but with `--hf-repo` the HF repo is already a durable backup.

### Budget saver tip

Skip the BF16 variants and just race E2B-8bit vs E4B-8bit:
- **2 pods x RTX 4090 x ~2.5h avg = ~$2.50 total**
- If E4B-8bit clearly wins, you can always try BF16 later to see if precision matters.
