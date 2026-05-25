#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/models/Qwen2.5-Math-1.5B-Instruct}"
GPU_ID="${GPU_ID:-0}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT_DIR/.cache/huggingface/datasets}"

mkdir -p "$HF_DATASETS_CACHE"
cd "$ROOT_DIR"

HF_ENDPOINT="$HF_ENDPOINT" \
HF_HOME="$HF_HOME" \
HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
CUDA_VISIBLE_DEVICES="$GPU_ID" uv run python -m hw4.train \
  --model_name "$MODEL_PATH" \
  --task math_hard \
  --algo reinforce \
  --output_dir runs/math_hard_reinforce \
  --steps 201 \
  --batch_size 8 \
  --group_size 8 \
  --min_new_tokens 8 \
  --max_new_tokens 512 \
  --max_prompt_tokens 512 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 3e-5 \
  --minibatch_size 8 \
  --grad_accum_steps 8 \
  --kl_coef 0.05 \
  --max_grad_norm 0.5 \
  --cuda_empty_cache_interval 50 \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 8 \
  --eval_interval 100 \
  --save_interval 100 \
  --no-wandb_enabled
