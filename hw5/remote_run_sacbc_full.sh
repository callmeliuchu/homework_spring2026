#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo ".venv is missing. Run ./remote_setup_env.sh first." >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec .venv/bin/python src/scripts/run.py \
  --run_group="${RUN_GROUP:-sacbc_full}" \
  --base_config="${BASE_CONFIG:-sacbc}" \
  --env_name="${ENV_NAME:-cube-single-play-singletask-task1-v0}" \
  --seed="${SEED:-0}" \
  --training_steps="${TRAINING_STEPS:-1000000}" \
  --log_interval="${LOG_INTERVAL:-10000}" \
  --eval_interval="${EVAL_INTERVAL:-100000}" \
  --num_eval_trajectories="${NUM_EVAL_TRAJECTORIES:-25}" \
  "$@"
