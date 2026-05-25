#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/demo_dqn.sh
#   ./scripts/demo_dqn.sh exp/dqn_cartpole_sd1_20260507_120000
#   ./scripts/demo_dqn.sh exp/dqn_cartpole_sd1_20260507_120000 experiments/dqn/cartpole.yaml rgb_array previews/cartpole_demo.gif

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${1:-}"
CONFIG_FILE="${2:-experiments/dqn/cartpole.yaml}"
MODE="${3:-human}"                  # human | rgb_array
VIDEO_PATH="${4:-previews/dqn_demo.gif}"
EPISODES="${EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-500}"

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(ls -td exp/* 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "No valid run_dir found."
  echo "Please pass one explicitly, e.g."
  echo "  ./scripts/demo_dqn.sh exp/dqn_cartpole_sd1_YYYYMMDD_HHMMSS"
  exit 1
fi

if [[ ! -f "$RUN_DIR/agent.pt" ]]; then
  echo "Checkpoint not found: $RUN_DIR/agent.pt"
  exit 1
fi

echo "run_dir: $RUN_DIR"
echo "config:  $CONFIG_FILE"
echo "mode:    $MODE"

if [[ "$MODE" == "rgb_array" ]]; then
  uv run src/scripts/play_trained_dqn.py \
    --run_dir "$RUN_DIR" \
    --config_file "$CONFIG_FILE" \
    --episodes "$EPISODES" \
    --max_steps "$MAX_STEPS" \
    --mode rgb_array \
    --video_path "$VIDEO_PATH"
  echo "saved:   $VIDEO_PATH"
else
  uv run src/scripts/play_trained_dqn.py \
    --run_dir "$RUN_DIR" \
    --config_file "$CONFIG_FILE" \
    --episodes "$EPISODES" \
    --max_steps "$MAX_STEPS" \
    --mode human
fi
