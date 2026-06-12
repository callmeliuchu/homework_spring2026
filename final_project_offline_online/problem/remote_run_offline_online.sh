#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" && -x "../.venv/bin/python" ]]; then
  PYTHON_BIN="../.venv/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo ".venv is missing. Run ./remote_setup_env.sh first." >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

RUN_GROUP="${RUN_GROUP:-fp_oo}"
BASE_CONFIG="${BASE_CONFIG:?BASE_CONFIG is required}"
ENV_NAME="${ENV_NAME:-cube-single-play-singletask-task1-v0}"
SEED="${SEED:-0}"
OFFLINE_TRAINING_STEPS="${OFFLINE_TRAINING_STEPS:-500000}"
ONLINE_TRAINING_STEPS="${ONLINE_TRAINING_STEPS:-100000}"
LOG_INTERVAL="${LOG_INTERVAL:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
NUM_EVAL_TRAJECTORIES="${NUM_EVAL_TRAJECTORIES:-25}"
REPLAY_BUFFER_CAPACITY="${REPLAY_BUFFER_CAPACITY:-1000000}"
OFFLINE_DATA="${OFFLINE_DATA:-0}"
WSRL_STEPS="${WSRL_STEPS:-0}"

exec "$PYTHON_BIN" src/scripts/train_offline_online.py \
  --run_group="${RUN_GROUP}" \
  --base_config="${BASE_CONFIG}" \
  --env_name="${ENV_NAME}" \
  --seed="${SEED}" \
  --offline_training_steps="${OFFLINE_TRAINING_STEPS}" \
  --online_training_steps="${ONLINE_TRAINING_STEPS}" \
  --log_interval="${LOG_INTERVAL}" \
  --eval_interval="${EVAL_INTERVAL}" \
  --num_eval_trajectories="${NUM_EVAL_TRAJECTORIES}" \
  --replay_buffer_capacity="${REPLAY_BUFFER_CAPACITY}" \
  --offline_data="${OFFLINE_DATA}" \
  --wsrl_steps="${WSRL_STEPS}" \
  "$@"
