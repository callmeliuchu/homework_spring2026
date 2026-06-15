#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-cube-single-play-singletask-task1-v0}"
SEED="${SEED:-0}"
OFFLINE_TRAINING_STEPS="${OFFLINE_TRAINING_STEPS:-500000}"
ONLINE_TRAINING_STEPS="${ONLINE_TRAINING_STEPS:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
NUM_EVAL_TRAJECTORIES="${NUM_EVAL_TRAJECTORIES:-25}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_VISIBLE_DEVICES WANDB_MODE=disabled

run_qsm() {
  local alpha="$1"
  echo "[$(date '+%F %T')] QSM fine alpha=${alpha} inv_temp=50 flow_steps=10"
  BASE_CONFIG=qsm \
  RUN_GROUP=codex_qsm_fine_bc_guidance_offline_best \
  ENV_NAME="$ENV_NAME" \
  SEED="$SEED" \
  OFFLINE_TRAINING_STEPS="$OFFLINE_TRAINING_STEPS" \
  ONLINE_TRAINING_STEPS="$ONLINE_TRAINING_STEPS" \
  LOG_INTERVAL="$LOG_INTERVAL" \
  EVAL_INTERVAL="$EVAL_INTERVAL" \
  NUM_EVAL_TRAJECTORIES="$NUM_EVAL_TRAJECTORIES" \
  ./remote_run_offline_online.sh --alpha="$alpha" --inv_temp=50 --flow_steps=10
}

run_qsm 0.0005
run_qsm 0.0008
run_qsm 0.001
