#!/usr/bin/env bash
set -euo pipefail

export NUM_NODES="${NUM_NODES:-1}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-4}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-2}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export DATALIST="${DATALIST:-AI2D_TEST DynaMath MathVista_MINI OCRBench SEEDBench2_Plus VisuLogic LogicVista VisualPuzzles MathVision}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-missing_qwen25_32b_text_image_default_prompt}"

export MODEL_PATH_QWEN25_32B="${MODEL_PATH_QWEN25_32B:-/models/Qwen2.5-VL-32B-Instruct}"
export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-Qwen2.5-VL-32B-Instruct__text_image__last1}"

unset VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS REPLAY_LIMIT_MM_PER_PROMPT INFER_BATCH_SIZE
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
export VLLM_MAX_MODEL_LEN_32B="${VLLM_MAX_MODEL_LEN_32B:-32768}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"
export VLLM_MAX_NUM_SEQS_32B="${VLLM_MAX_NUM_SEQS_32B:-${INFER_BATCH_SIZE}}"
export REPLAY_LIMIT_MM_PER_PROMPT_32B="${REPLAY_LIMIT_MM_PER_PROMPT_32B:-2}"

export REPLAY_PROMPT_TEMPLATE_NAME="identity"
unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_standard_qwen25_32b72b_newsets_last1_dataset_sweep.sh"
