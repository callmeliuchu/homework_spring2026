#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo ".venv is missing. Run ./remote_setup_env.sh first." >&2
  exit 1
fi

run_env_grid() {
  local env_name="$1"
  local run_group="$2"
  shift 2
  local -a alphas=("$1")
  local -a expectiles=("$2")

  # shellcheck disable=SC2206
  alphas=(${alphas[0]})
  # shellcheck disable=SC2206
  expectiles=(${expectiles[0]})

  for alpha in "${alphas[@]}"; do
    for expectile in "${expectiles[@]}"; do
      echo "[$(date '+%F %T')] START env=${env_name} alpha=${alpha} expectile=${expectile}"
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      RUN_GROUP="${run_group}" \
      BASE_CONFIG=iql \
      ENV_NAME="${env_name}" \
      TRAINING_STEPS="${TRAINING_STEPS:-1000000}" \
      LOG_INTERVAL="${LOG_INTERVAL:-10000}" \
      EVAL_INTERVAL="${EVAL_INTERVAL:-100000}" \
      NUM_EVAL_TRAJECTORIES="${NUM_EVAL_TRAJECTORIES:-25}" \
      SEED="${SEED:-0}" \
      ./remote_run_sacbc_full.sh --alpha="${alpha}" --expectile="${expectile}" --seed="${SEED:-0}"
      echo "[$(date '+%F %T')] DONE env=${env_name} alpha=${alpha} expectile=${expectile}"
    done
  done
}

run_env_grid \
  "cube-single-play-singletask-task1-v0" \
  "iql_cube_sweep" \
  "3 10 30" \
  "0.7 0.9"

run_env_grid \
  "antmaze-medium-navigate-singletask-task1-v0" \
  "iql_antmaze_sweep" \
  "3 10 30" \
  "0.7 0.9"

run_env_grid \
  "antsoccer-arena-navigate-singletask-task1-v0" \
  "iql_antsoccer_sweep" \
  "3 10 30" \
  "0.7 0.9"
