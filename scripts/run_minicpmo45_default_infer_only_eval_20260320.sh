#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/vlmevalkit/bin/python}"
GPU_IDS="${GPU_IDS:-0}"
OUTER_WORKERS="${OUTER_WORKERS:-4}"
DATASETS="${DATASETS:-AI2D_TEST MathVista_MINI OCRBench SEEDBench2_Plus VisuLogic LogicVista VisualPuzzles DynaMath MathVision}"
MODES="${MODES:-image_text text_image image_text_text image_text_image image_text_image_text image_image_text}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
OPENAI_API_KEY_JUDGE="${OPENAI_API_KEY_JUDGE:-${OPENAI_API_KEY:-}}"
OPENAI_API_BASE_JUDGE="${OPENAI_API_BASE_JUDGE:-https://api.openai.com/v1}"

cd "$REPO_ROOT"

export OPENAI_API_KEY="$OPENAI_API_KEY_JUDGE"
export OPENAI_API_BASE="$OPENAI_API_BASE_JUDGE"
export OPENAI_API_KEY_JUDGE
export OPENAI_API_BASE_JUDGE

"$PYTHON_BIN" scripts/rerun_eval_for_infer_complete_bysetting.py \
  --matrix-config scripts/configs/matrix_minicpm_default_infer_only_fresh_20260317.yaml \
  --model-config scripts/configs/models.yaml \
  --models minicpm_o_45 \
  --policies default \
  --modes "$MODES" \
  --datasets "$DATASETS" \
  --gpu-ids "$GPU_IDS" \
  --outer-workers "$OUTER_WORKERS" \
  --rerun-even-if-acc \
  --drop-answer-format \
  "$@"
