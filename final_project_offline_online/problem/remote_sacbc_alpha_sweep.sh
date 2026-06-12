#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo ".venv is missing. Run ./remote_setup_env.sh first." >&2
  exit 1
fi

RUN_GROUP="${RUN_GROUP:?RUN_GROUP is required}"
ENV_NAME="${ENV_NAME:?ENV_NAME is required}"
SEED="${SEED:-0}"
TRAINING_STEPS="${TRAINING_STEPS:-300000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
NUM_EVAL_TRAJECTORIES="${NUM_EVAL_TRAJECTORIES:-25}"
ALPHAS="${ALPHAS:-1 3 10 30 100 300}"

export PYTHONPATH="${ROOT_DIR}/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

for alpha in ${ALPHAS}; do
  echo "[$(date '+%F %T')] START env=${ENV_NAME} alpha=${alpha}"
  .venv/bin/python src/scripts/run.py \
    --run_group="${RUN_GROUP}" \
    --base_config=sacbc \
    --env_name="${ENV_NAME}" \
    --seed="${SEED}" \
    --training_steps="${TRAINING_STEPS}" \
    --eval_interval="${EVAL_INTERVAL}" \
    --log_interval="${LOG_INTERVAL}" \
    --num_eval_trajectories="${NUM_EVAL_TRAJECTORIES}" \
    --alpha="${alpha}"
  echo "[$(date '+%F %T')] DONE env=${ENV_NAME} alpha=${alpha}"
done
