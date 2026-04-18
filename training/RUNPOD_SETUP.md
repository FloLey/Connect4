# RunPod Setup — Connect Four GSPO Training (Gemma-4 + Unsloth)

Train Gemma-4 on a single RunPod RTX 4090 with wandb live metrics and
per-checkpoint Hugging Face pushes. No vLLM (Unsloth's reference
Gemma-4 RL recipe uses eager generation). The pipeline is SFT (minimal
format warmup) → GSPO (1000 steps, online Gaussian curriculum) → eval
→ export → push.

## Pod spec

| Item | Value |
|------|-------|
| Template | `unsloth/unsloth:latest` |
| GPU | RTX 4090 (24 GB) |
| Variant | `e4b-8bit` (unsloth/gemma-4-E4B-it, 8-bit) |
| Peak VRAM | ~12 GB |
| Wall time | ~5–7 h for 1000 steps |
| Cost | ~$5 at $0.50/h |

## Step 0 — Accounts

1. **wandb**: sign up at [wandb.ai](https://wandb.ai), grab your key
   from [wandb.ai/authorize](https://wandb.ai/authorize).
2. **HuggingFace**: write token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   Pick the repo name up front — the HF repo (private) receives one
   adapter checkpoint every 300 GSPO steps so a pod crash loses ≤1
   save window. Example: `Betha/connect4-agent-e4b-8bit`.

## Step 1 — Environment

On the pod (user `unsloth`, `/root` is not writable):

```bash
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/home/unsloth/hf_cache
export TORCHINDUCTOR_CACHE_DIR=/home/unsloth/torchinductor_cache
export TRITON_CACHE_DIR=/home/unsloth/triton_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
EOF
source ~/.bashrc
mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

git clone https://github.com/FloLey/Connect4.git
cd Connect4/training

# Pinned to the exact versions Unsloth tests for Gemma-4 RL.
# Do NOT install vllm; do NOT -U trl.
pip install 'transformers==5.5.0' 'trl==0.28.0' wandb hf_transfer

wandb login
hf auth login
```

Sanity check:

```bash
python -c "
import unsloth
from transformers import AutoConfig
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
cfg = AutoConfig.from_pretrained('unsloth/gemma-4-E4B-it')
assert cfg.model_type == 'gemma4'
print('OK')
"
```

Must print `OK`.

## Step 2 — Dry-run each stage locally (no-GPU stages)

You can verify these without a GPU — they only touch data and Python
logic:

```bash
python connect4_train.py --model e4b-8bit --stage test-data      --csv connect4_data.csv
python connect4_train.py --model e4b-8bit --stage test-rewards   --csv connect4_data.csv
python connect4_train.py --model e4b-8bit --stage test-curriculum --csv connect4_data.csv
```

On the pod (GPU):

```bash
python connect4_train.py --model e4b-8bit --stage test-load     --csv connect4_data.csv
python connect4_train.py --model e4b-8bit --stage test-prompt   --csv connect4_data.csv
python connect4_train.py --model e4b-8bit --stage test-generate --csv connect4_data.csv
```

`test-generate` should show `<|channel>thought ...` delimiters in the
RAW output — native thinking is firing.

## Step 3 — Full pipeline

```bash
python connect4_train.py --model e4b-8bit --stage sft \
    --csv connect4_data.csv --hf-repo Betha/connect4-agent-e4b-8bit
```

This runs the minimal SFT warmup (200 examples, ≤25 steps) and
auto-chains into GSPO once the probe clears 80% format compliance.
The SFT adapter is pushed to `Betha/connect4-agent-e4b-8bit-sft`;
GSPO checkpoints land in `Betha/connect4-agent-e4b-8bit` every 300
steps.

If SFT succeeded previously and you just want GSPO:

```bash
python connect4_train.py --model e4b-8bit --stage grpo --sft \
    --csv connect4_data.csv --hf-repo Betha/connect4-agent-e4b-8bit
```

`--sft` tells the trainer to pull `{hf-repo}-sft` and load it as the
starting adapter.

## Step 4 — Monitor in wandb

Project: `connect4-llm` (matches `config["wandb_project"]`).

Key panels:

- `reward` — aggregate, should climb from ~0 to >10 over hundreds of
  steps.
- `rewards/rf_format` — +5 if the model emits a digit, else -10.
  After SFT this should start near +5.
- `rewards/rf_quality` — normalized oracle score in [-10, +10].
- `completions/mean_length` — grows as the model discovers longer
  thinking helps.
- `curriculum/level` — 0–9 difficulty level. Advances when the
  current-center bucket's EMA reward crosses 10.5 (= 15 × 0.7).
- `curriculum/bucket_{i}_ema` — per-bucket reward EMA; shows the
  curriculum shape.
- `sample/completion` — HTML-pre of the most recent model output,
  logged every 25 steps by `ThinkingLogger`.

Healthy pattern: reward climbs → level advances → reward drops (harder
positions) → climbs back → repeat.

## Step 5 — Eval + export + push

After GSPO finishes:

```bash
python connect4_train.py --model e4b-8bit --stage eval    --csv connect4_data.csv
python connect4_train.py --model e4b-8bit --stage export  --csv connect4_data.csv
python connect4_train.py --model e4b-8bit --stage push    --hf-repo Betha/connect4-agent-e4b-8bit
```

`eval` writes `eval_results_e4b-8bit.json` (exact/top-2/mean-oracle +
per-phase breakdown). `export` produces `connect4-agent-e4b-8bit/`
(16-bit merged) and `connect4-agent-e4b-8bit-gguf/` (q4_k_m). `push`
uploads both to HF.

## Step 6 — Resume after pod death

GSPO writes a checkpoint every 300 steps and mirrors it to HF. To
resume on a fresh pod after Step 1:

```bash
hf download Betha/connect4-agent-e4b-8bit --local-dir outputs_grpo_e4b-8bit
python connect4_train.py --model e4b-8bit --stage grpo \
    --csv connect4_data.csv --hf-repo Betha/connect4-agent-e4b-8bit
```

TRL auto-resumes from `outputs_grpo_{variant}/checkpoint-*/`.

## Troubleshooting

| Symptom | First fix | Second fix |
|---------|-----------|------------|
| `rewards/rf_format` stuck at -10 after 100 GSPO steps | Re-run SFT with 500 samples × 50 steps | 1000 samples × 100 steps, lr=5e-5 |
| `rewards/rf_quality` near 0, `completions/mean_length` ~30 | Switch `loss_type='bnpo'` in MODEL_CONFIGS | Raise temperature to 1.2 |
| Reward drops and format collapses | Lower lr to 2e-5 | Set `grpo_beta=0.01` (small KL to ref) |
| `reward_std` near 0 (group collapse) | Raise temperature to 1.2 | Increase K to 6 |
| `completions/clipped_ratio` ≈ 1.0 | Raise `max_completion_length` to 4096, reduce K to 2 | Shrink `max_prompt_length` |
| OOM | K from 4 → 2 | `max_completion_length` 2048 → 1024 |
| Step time > 5 min | Reduce K or completion length | Accept (no-vLLM HF generate is slow) |

Each fix: edit `MODEL_CONFIGS` in `connect4_train.py`, commit, push,
pull on the pod, relaunch — TRL auto-resumes from the latest
checkpoint.
