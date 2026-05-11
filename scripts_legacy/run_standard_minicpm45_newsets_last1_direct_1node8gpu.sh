#!/usr/bin/env bash
set -euo pipefail

export NUM_NODES="${NUM_NODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-8}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export DATALIST="${DATALIST:-AI2D_TEST DynaMath MathVista_MINI OCRBench SEEDBench2_Plus VisuLogic LogicVista VisualPuzzles MathVision}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-minicpm45_newsets_last1_1node8gpu_direct_answer}"
export MODEL_PATH_MINICPM45="${MODEL_PATH_MINICPM45:-/models/MiniCPM-V-4_5}"
export MINICPM45_USE_VLLM="${MINICPM45_USE_VLLM:-1}"
export MINICPM45_VLLM_TP_SIZE="${MINICPM45_VLLM_TP_SIZE:-1}"

export REPLAY_PROMPT_TEMPLATE_NAME="directly_answer"
unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE TASK_TAG_ALLOWLIST

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_standard_minicpm45_newsets_last1_2node_8gpu_sweep.sh"
