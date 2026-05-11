#!/usr/bin/env bash
set -euo pipefail

# Re-eval only: Qwen2.5-VL-72B Newsets. Reuses infer under runs/standard/20260304/.

export EXP_DATE_TAG="${EXP_DATE_TAG:-20260304}"
export NUM_NODES="${NUM_NODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-1}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-8}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export DATALIST="${DATALIST:-AI2D_TEST DynaMath MathVision MathVista_MINI OCRBench SEEDBench2_Plus VisuLogic LogicVista VisualPuzzles}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen25_72b_newsets_last1_1node8gpu}"

export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-Qwen2.5-VL-72B-Instruct__none__last1,Qwen2.5-VL-72B-Instruct__image_text_text__last1,Qwen2.5-VL-72B-Instruct__image_text_image__last1,Qwen2.5-VL-72B-Instruct__image_text_image_text__last1,Qwen2.5-VL-72B-Instruct__image_image_text__last1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_reeval_qwen25_32b72b_sweep.sh"
