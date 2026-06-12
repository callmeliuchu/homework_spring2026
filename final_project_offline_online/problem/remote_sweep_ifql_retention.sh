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

run_ifql() {
  local expectile="$1"
  local offline_data="$2"
  local wsrl_steps="$3"
  echo "[$(date '+%F %T')] IFQL expectile=${expectile} offline_data=${offline_data} wsrl_steps=${wsrl_steps}"
  BASE_CONFIG=ifql \
  RUN_GROUP=fp_sweep_ifql_retention \
  ENV_NAME="$ENV_NAME" \
  SEED="$SEED" \
  OFFLINE_TRAINING_STEPS="$OFFLINE_TRAINING_STEPS" \
  ONLINE_TRAINING_STEPS="$ONLINE_TRAINING_STEPS" \
  LOG_INTERVAL="$LOG_INTERVAL" \
  EVAL_INTERVAL="$EVAL_INTERVAL" \
  NUM_EVAL_TRAJECTORIES="$NUM_EVAL_TRAJECTORIES" \
  OFFLINE_DATA="$offline_data" \
  WSRL_STEPS="$wsrl_steps" \
  ./remote_run_offline_online.sh --expectile="$expectile"
}

run_ifql 0.90 100000 0
run_ifql 0.90 200000 0
run_ifql 0.85 100000 0
run_ifql 0.90 100000 10000
