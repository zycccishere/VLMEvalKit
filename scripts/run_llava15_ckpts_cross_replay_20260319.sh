#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/vlmevalkit/bin/python}"
CKPT_ROOT="${CKPT_ROOT:-/path/to/LLaVA/checkpoints}"
LLAVA_REPO_ROOT="${LLAVA_REPO_ROOT:-/path/to/LLaVA}"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/runs/llava15_ckpt_cross_replay}"
RUN_MODE="${RUN_MODE:-all}"
JUDGE="${JUDGE:-gpt-4o-mini}"
OPENAI_API_KEY_VALUE="${OPENAI_API_KEY_VALUE:-${OPENAI_API_KEY_JUDGE:-${OPENAI_API_KEY:-}}}"
OPENAI_API_BASE_VALUE="${OPENAI_API_BASE_VALUE:-https://api.openai.com/v1}"
NPROC="${NPROC:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MODEL_NAME="${MODEL_NAME:-llava_v1.5_13b_replay}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
DATASETS="${DATASETS:-AI2D_TEST MathVista_MINI OCRBench SEEDBench2_Plus VisuLogic LogicVista VisualPuzzles DynaMath MathVision}"
RESUME_INFER="${RESUME_INFER:-0}"

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

CMD=(
  "$PYTHON_BIN" "$SCRIPT_DIR/run_llava15_ckpts_cross_replay_20260319.py"
  --repo-root "$REPO_ROOT"
  --python-bin "$PYTHON_BIN"
  --ckpt-root "$CKPT_ROOT"
  --work-root "$WORK_ROOT"
  --model-name "$MODEL_NAME"
  --mode "$RUN_MODE"
  --judge "$JUDGE"
  --nproc "$NPROC"
  --eval-workers "$EVAL_WORKERS"
  --batch-size "$BATCH_SIZE"
  --gpu-ids "$GPU_IDS"
  --datasets "$DATASETS"
)

if [[ "$RESUME_INFER" == "1" || "$RESUME_INFER" == "true" || "$RESUME_INFER" == "yes" || "$RESUME_INFER" == "on" ]]; then
  CMD+=(--resume-infer)
fi

if [[ "${1:-}" == "--plan-only" ]]; then
  CMD+=(--plan-only)
fi

exec "${CMD[@]}"
