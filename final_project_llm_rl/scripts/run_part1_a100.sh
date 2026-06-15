#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt1/mnt1/nlp/lc/final_project_llm_rl}"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"
DATASET="${DATASET:-$PROJECT_DIR/dataset/wildchat_min4_judged_5k_v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
GPU="${GPU:-0}"
MODE="${MODE:-all}"
HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"

export HF_HOME
export HF_HUB_CACHE
export TRANSFORMERS_CACHE
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$LOG_DIR" "$PROJECT_DIR/runs"
cd "$PROJECT_DIR"

run_offline() {
  local algo="$1"
  local beta="$2"
  local max_steps="${3:-500}"
  local out_dir="$PROJECT_DIR/runs/wildchat_min4_judged_5k_${algo}_beta${beta}_step${max_steps}_nogc"
  local log="$LOG_DIR/part1_${algo}_$(date +%Y%m%d_%H%M%S).log"

  if find "$out_dir/checkpoints" -maxdepth 2 -type d -name adapter 2>/dev/null | grep -q .; then
    echo "[$(date)] skip ${algo}: checkpoint already exists under $out_dir"
    return 0
  fi

  echo "[$(date)] start ${algo} on GPU ${GPU}; log=$log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -m llm_rl_final_proj.train \
    --algo "$algo" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET" \
    --train_split train_prefs \
    --eval_split test_prefs \
    --generation_split test_gen \
    --output_dir "$out_dir" \
    --beta "$beta" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --grad_accum_steps 4 \
    --lr 5e-5 \
    --max_steps "$max_steps" \
    --max_prompt_tokens 700 \
    --max_response_tokens 512 \
    --generation_eval_limit 32 \
    --generation_eval_max_new_tokens 256 \
    --generation_eval_every 100 \
    --eval_interval 100 \
    --save_interval 100 \
    --no-wandb_enabled \
    --no-grad_checkpointing \
    >"$log" 2>&1
  echo "[$(date)] done ${algo}"
}

run_reward_model() {
  local max_steps="${1:-200}"
  local out_dir="$PROJECT_DIR/runs/wildchat_min4_judged_5k_reward_model_step${max_steps}_nogc"
  local log="$LOG_DIR/part1_reward_model_$(date +%Y%m%d_%H%M%S).log"
  local adapter="$out_dir/checkpoints/step_$(printf '%06d' "$max_steps")/adapter"

  if [ -f "$adapter/adapter_config.json" ]; then
    REWARD_ADAPTER="$adapter"
    return 0
  fi

  echo "[$(date)] start reward_model on GPU ${GPU}; log=$log" >&2
  if ! CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -m llm_rl_final_proj.reward_model.train \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET" \
    --train_split train_prefs \
    --eval_split test_prefs \
    --output_dir "$out_dir" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --grad_accum_steps 16 \
    --lr 3e-5 \
    --max_steps "$max_steps" \
    --max_prompt_tokens 700 \
    --max_response_tokens 512 \
    --eval_interval 25 \
    --save_interval 50 \
    --no-wandb_enabled \
    >"$log" 2>&1; then
    echo "[$(date)] reward_model failed; see $log" >&2
    return 1
  fi
  if [ ! -f "$adapter/adapter_config.json" ]; then
    echo "[$(date)] reward_model finished but missing $adapter/adapter_config.json" >&2
    return 1
  fi
  REWARD_ADAPTER="$adapter"
}

run_online() {
  local algo="$1"
  local reward_adapter="$2"
  local steps="${3:-25}"
  local out_dir="$PROJECT_DIR/runs/wildchat_min4_judged_5k_${algo}_step${steps}_nogc"
  local log="$LOG_DIR/part1_${algo}_$(date +%Y%m%d_%H%M%S).log"

  if find "$out_dir/checkpoints" -maxdepth 2 -type d -name adapter 2>/dev/null | grep -q .; then
    echo "[$(date)] skip ${algo}: checkpoint already exists under $out_dir"
    return 0
  fi

  echo "[$(date)] start ${algo} on GPU ${GPU}; log=$log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -m llm_rl_final_proj.online.train_rm_grpo \
    --algo "$algo" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET" \
    --train_split train_gen \
    --eval_split test_gen \
    --reward_model_name "$MODEL_NAME" \
    --reward_adapter_path "$reward_adapter" \
    --output_dir "$out_dir" \
    --steps "$steps" \
    --batch_size 16 \
    --group_size 4 \
    --min_new_tokens 32 \
    --max_new_tokens 256 \
    --temperature 0.8 \
    --top_p 0.95 \
    --lr 1e-5 \
    --grad_accum_steps 2 \
    --ppo_epochs 2 \
    --minibatch_size 8 \
    --clip_eps 0.2 \
    --kl_coef 0.01 \
    --max_prompt_tokens 700 \
    --max_response_tokens 256 \
    --eval_limit 32 \
    --eval_interval 25 \
    --save_interval 25 \
    --no-wandb_enabled \
    >"$log" 2>&1
  echo "[$(date)] done ${algo}"
}

case "$MODE" in
  offline)
    run_offline ipo 0.005 500
    run_offline aot 0.005 500
    ;;
  reward_online)
    run_reward_model 200
    run_online grpo "$REWARD_ADAPTER" 25
    run_online dr_grpo "$REWARD_ADAPTER" 25
    run_online gspo "$REWARD_ADAPTER" 25
    ;;
  all)
    run_offline ipo 0.005 500
    run_offline aot 0.005 500
    run_reward_model 200
    run_online grpo "$REWARD_ADAPTER" 25
    run_online dr_grpo "$REWARD_ADAPTER" 25
    run_online gspo "$REWARD_ADAPTER" 25
    ;;
  *)
    echo "Unknown MODE=$MODE; expected all, offline, or reward_online" >&2
    exit 2
    ;;
esac
