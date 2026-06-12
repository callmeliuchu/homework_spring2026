#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-cube-single-play-singletask-task1-v0}"
SEED="${SEED:-0}"
OFFLINE_TRAINING_STEPS="${OFFLINE_TRAINING_STEPS:-500000}"
ONLINE_TRAINING_STEPS="${ONLINE_TRAINING_STEPS:-100000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
NUM_EVAL_TRAJECTORIES="${NUM_EVAL_TRAJECTORIES:-25}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_VISIBLE_DEVICES WANDB_MODE=disabled

run_dsrl() {
  local noise_scale="$1"
  local fixed_alpha="$2"
  echo "[$(date '+%F %T')] DSRL noise_scale=${noise_scale} fixed_alpha=${fixed_alpha}"
  BASE_CONFIG=dsrl \
  RUN_GROUP=fp_sweep_dsrl_stable \
  ENV_NAME="$ENV_NAME" \
  SEED="$SEED" \
  OFFLINE_TRAINING_STEPS="$OFFLINE_TRAINING_STEPS" \
  ONLINE_TRAINING_STEPS="$ONLINE_TRAINING_STEPS" \
  LOG_INTERVAL="$LOG_INTERVAL" \
  EVAL_INTERVAL="$EVAL_INTERVAL" \
  NUM_EVAL_TRAJECTORIES="$NUM_EVAL_TRAJECTORIES" \
  ./remote_run_offline_online.sh --noise_scale="$noise_scale" --bc_pretrain_steps=100000 --fixed_alpha="$fixed_alpha"
}

run_dsrl 0.5 0.01
run_dsrl 0.8 0.01
run_dsrl 0.5 0.05
run_dsrl 0.8 0.05
