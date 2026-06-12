#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/mnt1/mnt1/nlp/lc/final_project_offline_online/problem"
LOCAL_ROOT="/Users/liuchu/codes/homework_spring2026/final_project_offline_online/problem"
REMOTE_PY="../.venv/bin/python"

FQL_RUN="exp/fp_sweep_fql_cube_official/sd0_20260612_062348_fql_cube-single-play-singletask-task1-v0_a100.0_f10_online_offline"
DSRL_RUN="exp/fp_sweep_dsrl_stable/sd0_20260612_054329_dsrl_cube-single-play-singletask-task1-v0_n0.5_bp100000_fa0.01_online_offline"

mkdir -p "$LOCAL_ROOT/visualizations"

generate_and_sync() {
  local name="$1"
  local base_config="$2"
  local run_dir="$3"
  local remote_gif="visualizations/${name}_cube_best.gif"
  local local_gif="$LOCAL_ROOT/visualizations/${name}_cube_best.gif"

  echo "[$(date '+%F %T')] waiting for ${name} checkpoint"
  until ssh a100 "test -f '${REMOTE_ROOT}/${run_dir}/agent.pt'"; do
    sleep 120
  done

  echo "[$(date '+%F %T')] generating ${name} gif"
  ssh a100 "cd '${REMOTE_ROOT}' && mkdir -p visualizations && WANDB_MODE=disabled ${REMOTE_PY} src/scripts/preview_checkpoint.py --base_config '${base_config}' --env_name cube-single-play-singletask-task1-v0 --checkpoint_path '${run_dir}/agent.pt' --gif_path '${remote_gif}' --episodes 8 --fps 20 --which_gpu 0"

  echo "[$(date '+%F %T')] syncing ${name} gif to local"
  rsync -az "a100:${REMOTE_ROOT}/${remote_gif}" "${local_gif}"
  echo "[$(date '+%F %T')] saved ${local_gif}"
}

generate_and_sync "fql" "fql" "$FQL_RUN" &
generate_and_sync "dsrl" "dsrl" "$DSRL_RUN" &
wait

echo "[$(date '+%F %T')] done"
