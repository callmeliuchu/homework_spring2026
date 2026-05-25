#!/usr/bin/env bash
set -euo pipefail

# Train PG on CartPole, then launch a human-rendered demo.
# Usage:
#   bash scripts/train_and_demo_cartpole.sh
# Optional env overrides:
#   N_ITER=50 BATCH_SIZE=2000 EPISODES=5 EXP_NAME=demo_cartpole bash scripts/train_and_demo_cartpole.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-CartPole-v0}"
EXP_NAME="${EXP_NAME:-demo_cartpole}"
N_ITER="${N_ITER:-30}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-400}"
EPISODES="${EPISODES:-5}"
SEED="${SEED:-1}"

echo "==> Training policy"
uv run python src/scripts/run.py \
  --env_name "$ENV_NAME" \
  --exp_name "$EXP_NAME" \
  -n "$N_ITER" \
  -b "$BATCH_SIZE" \
  -eb "$EVAL_BATCH_SIZE" \
  --seed "$SEED" \
  --use_reward_to_go \
  --use_baseline \
  --normalize_advantages \
  --video_log_freq -1

RUN_DIR="$(ls -dt exp/* | head -n 1)"
echo "==> Latest run_dir: $RUN_DIR"
echo "==> Starting demo rollout"

uv run python src/scripts/play_trained.py \
  --run_dir "$RUN_DIR" \
  --episodes "$EPISODES"
