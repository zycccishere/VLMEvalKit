#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/vlmevalkit/bin/python}"
CKPT_ROOT="${CKPT_ROOT:-/path/to/LLaVA/checkpoints}"
LLAVA_REPO_ROOT="${LLAVA_REPO_ROOT:-/path/to/LLaVA}"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/runs/llava15_ckpt_native_replay}"
RUN_MODE="${RUN_MODE:-all}"
JUDGE="${JUDGE:-gpt-4o-mini}"
OPENAI_API_KEY_VALUE="${OPENAI_API_KEY_VALUE:-${OPENAI_API_KEY_JUDGE:-${OPENAI_API_KEY:-}}}"
OPENAI_API_BASE_VALUE="${OPENAI_API_BASE_VALUE:-https://api.openai.com/v1}"
NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MODEL_NAME="${MODEL_NAME:-llava_v1.5_13b_replay}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATASETS_RAW="${DATASETS:-AI2D_TEST MathVista_MINI OCRBench SEEDBench2_Plus VisuLogic LogicVista VisualPuzzles DynaMath MathVision}"

read -r -a DATASETS <<< "$DATASETS_RAW"

export PYTHONPATH="${LLAVA_REPO_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OPENAI_API_KEY="${OPENAI_API_KEY_VALUE}"
export OPENAI_API_KEY_JUDGE="${OPENAI_API_KEY_VALUE}"
export OPENAI_API_BASE="${OPENAI_API_BASE_VALUE}"
export OPENAI_API_BASE_JUDGE="${OPENAI_API_BASE_VALUE}"
export LLAVA_USE_VLLM="${LLAVA_USE_VLLM:-0}"
export REPLAY_TIMES="${REPLAY_TIMES:-1}"
export REPLAY_IMAGE_COPY_MODE="${REPLAY_IMAGE_COPY_MODE:-reuse_path}"
export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT:-1}"
export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME:-identity}"
export CUDA_VISIBLE_DEVICES

discover_ckpts() {
  find "$CKPT_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'llava-v1.5-13b-image_text*' | sort
}

mode_from_ckpt() {
  local base
  base="$(basename "$1")"
  echo "${base#llava-v1.5-13b-}"
}

run_one() {
  local ckpt_path="$1"
  local train_mode="$2"
  local eval_mode="$3"
  local tag="${train_mode}_train__${eval_mode}_eval"
  local work_dir="$WORK_ROOT/$(basename "$ckpt_path")/$tag"

  mkdir -p "$work_dir"

  echo "[START] ckpt=$(basename "$ckpt_path") train=$train_mode eval=$eval_mode mode=$RUN_MODE cuda=$CUDA_VISIBLE_DEVICES"

  export MODEL_PATH="$ckpt_path"
  export REPLAY_MODE="$eval_mode"

  local cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/run.py"
    --data "${DATASETS[@]}"
    --model "$MODEL_NAME"
    --work-dir "$work_dir"
    --mode "$RUN_MODE"
    --batch-size "$BATCH_SIZE"
    --verbose
  )

  if [[ "$RUN_MODE" != "infer" ]]; then
    cmd+=(--judge "$JUDGE" --nproc "$NPROC")
  fi

  (
    cd /tmp
    "${cmd[@]}"
  )

  echo "[DONE] ckpt=$(basename "$ckpt_path") train=$train_mode eval=$eval_mode"
}

main() {
  mapfile -t CKPTS < <(discover_ckpts)
  if [[ "${#CKPTS[@]}" -eq 0 ]]; then
    echo "No replay checkpoints found under $CKPT_ROOT" >&2
    exit 1
  fi

  for ckpt_path in "${CKPTS[@]}"; do
    train_mode="$(mode_from_ckpt "$ckpt_path")"
    run_one "$ckpt_path" "$train_mode" "$train_mode"
  done
}

main "$@"
