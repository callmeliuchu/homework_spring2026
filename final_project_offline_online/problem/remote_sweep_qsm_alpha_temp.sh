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

run_qsm() {
  local inv_temp="$1"
  local flow_steps="$2"
  echo "[$(date '+%F %T')] QSM inv_temp=${inv_temp} flow_steps=${flow_steps}"
  BASE_CONFIG=qsm \
  RUN_GROUP=fp_sweep_qsm_minimal_linear \
  ENV_NAME="$ENV_NAME" \
  SEED="$SEED" \
  OFFLINE_TRAINING_STEPS="$OFFLINE_TRAINING_STEPS" \
  ONLINE_TRAINING_STEPS="$ONLINE_TRAINING_STEPS" \
  LOG_INTERVAL="$LOG_INTERVAL" \
  EVAL_INTERVAL="$EVAL_INTERVAL" \
  NUM_EVAL_TRAJECTORIES="$NUM_EVAL_TRAJECTORIES" \
  ./remote_run_offline_online.sh --inv_temp="$inv_temp" --flow_steps="$flow_steps"
}

run_qsm 50 10
run_qsm 100 10
run_qsm 30 10
run_qsm 50 5
