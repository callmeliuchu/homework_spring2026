#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x "${HOME}/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [[ ! -x ".venv/bin/python" ]]; then
  "${HOME}/.local/bin/uv" venv --python 3.11 --seed .venv
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install torch ogbench matplotlib opencv-python ml_collections wandb tqdm

.venv/bin/python - <<'PY'
import ogbench

ogbench.download_datasets([
    "cube-single-play-v0",
    "antsoccer-arena-navigate-v0",
    "antmaze-medium-navigate-v0",
])
PY

export PYTHONPATH="${ROOT_DIR}/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec .venv/bin/python src/scripts/run.py \
  --run_group="${RUN_GROUP:-a100_full}" \
  --base_config="${BASE_CONFIG:-sacbc}" \
  --env_name="${ENV_NAME:-cube-single-play-singletask-task1-v0}" \
  --seed="${SEED:-0}" \
  --training_steps="${TRAINING_STEPS:-1000000}" \
  --log_interval="${LOG_INTERVAL:-10000}" \
  --eval_interval="${EVAL_INTERVAL:-100000}" \
  --num_eval_trajectories="${NUM_EVAL_TRAJECTORIES:-25}" \
  "$@"
