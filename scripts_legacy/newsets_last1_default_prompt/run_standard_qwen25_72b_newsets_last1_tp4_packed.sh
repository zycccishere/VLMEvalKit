#!/usr/bin/env bash
set -euo pipefail

# Newsets + replay sweep for Qwen2.5-VL-72B-Instruct only.
#
# Packed default topology for a single 8-GPU node:
# - 1 node
# - 2 jobs per node
# - 4 GPUs per job
# - TP=4
# - up to 2 workers in parallel per node
#
# Prompt policy:
# - reuse dataset default prompt construction (identity template)
#
# Notes:
# - This is a lower-TP alternative to the original TP=8 script.
# - Uses dataset-level scheduling so additional nodes remain useful.
# - Override NUM_NODES / JOBS_PER_NODE / GPUS_PER_JOB if your cluster layout differs.

export NUM_NODES="${NUM_NODES:-1}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-2}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export DATALIST="${DATALIST:-AI2D_TEST DynaMath MathVision MathVista_MINI OCRBench SEEDBench2_Plus VisuLogic LogicVista VisualPuzzles}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen25_72b_newsets_last1_1node2tasks_tp4_default_prompt}"

export MODEL_PATH_QWEN25_72B="${MODEL_PATH_QWEN25_72B:-/models/Qwen2.5-VL-72B-Instruct}"
export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-Qwen2.5-VL-72B-Instruct__none__last1,Qwen2.5-VL-72B-Instruct__image_text_text__last1,Qwen2.5-VL-72B-Instruct__image_text_image__last1,Qwen2.5-VL-72B-Instruct__image_text_image_text__last1,Qwen2.5-VL-72B-Instruct__image_image_text__last1}"
export REUSE_FROM_EXP_GROUP_TAG="${REUSE_FROM_EXP_GROUP_TAG:-qwen25_72b_newsets_last1_4node4tasks_default_prompt}"

unset VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS REPLAY_LIMIT_MM_PER_PROMPT INFER_BATCH_SIZE
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"
export VLLM_MAX_MODEL_LEN_72B="${VLLM_MAX_MODEL_LEN_72B:-32768}"
export VLLM_MAX_NUM_SEQS_72B="${VLLM_MAX_NUM_SEQS_72B:-1}"
export REPLAY_LIMIT_MM_PER_PROMPT_72B="${REPLAY_LIMIT_MM_PER_PROMPT_72B:-2}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"

export REPLAY_PROMPT_TEMPLATE_NAME="identity"
unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../run_standard_qwen25_32b72b_newsets_last1_dataset_sweep.sh"
