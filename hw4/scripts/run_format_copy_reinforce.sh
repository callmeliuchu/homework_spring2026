#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/models/Qwen2.5-Math-1.5B-Instruct}"
GPU_ID="${GPU_ID:-0}"

cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" uv run python -m hw4.train \
  --model_name "$MODEL_PATH" \
  --task format_copy \
  --algo reinforce \
  --output_dir runs/format_copy_reinforce \
  --steps 51 \
  --batch_size 8 \
  --group_size 6 \
  --min_new_tokens 1 \
  --max_new_tokens 24 \
  --lr 3e-5 \
  --minibatch_size 8 \
  --grad_accum_steps 6 \
  --kl_coef 0.05 \
  --max_grad_norm 0.5 \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 6 \
  --eval_interval 50 \
  --save_interval 50 \
  --warmup_steps 10 \
  --no-wandb_enabled
