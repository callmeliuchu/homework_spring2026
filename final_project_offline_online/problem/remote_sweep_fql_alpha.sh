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
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES WANDB_MODE=disabled

run_fql() {
  local alpha="$1"
  local flow_steps="$2"
  echo "[$(date '+%F %T')] FQL alpha=${alpha} flow_steps=${flow_steps}"
  BASE_CONFIG=fql \
  RUN_GROUP=fp_sweep_fql_cube_official \
  ENV_NAME="$ENV_NAME" \
  SEED="$SEED" \
  OFFLINE_TRAINING_STEPS="$OFFLINE_TRAINING_STEPS" \
  ONLINE_TRAINING_STEPS="$ONLINE_TRAINING_STEPS" \
  LOG_INTERVAL="$LOG_INTERVAL" \
  EVAL_INTERVAL="$EVAL_INTERVAL" \
  NUM_EVAL_TRAJECTORIES="$NUM_EVAL_TRAJECTORIES" \
  ./remote_run_offline_online.sh --alpha="$alpha" --flow_steps="$flow_steps"
}

run_fql 100 10
run_fql 300 10
run_fql 1000 10
