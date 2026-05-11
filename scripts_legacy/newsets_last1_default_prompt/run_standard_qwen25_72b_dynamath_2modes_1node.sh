#!/usr/bin/env bash
set -euo pipefail

# DynaMath-only sweep for Qwen2.5-VL-72B-Instruct.
# Modes:
# - image_text
# - image_text_image
#
# Default topology:
# - 1 node
# - 2 jobs in parallel
# - 4 GPUs/job
# - TP=4

export NUM_NODES="${NUM_NODES:-1}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-2}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export DATALIST="${DATALIST:-DynaMath}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen25_72b32b7b_dynamath_2node_split_default_prompt}"

export MODEL_PATH_QWEN25_72B="${MODEL_PATH_QWEN25_72B:-/models/Qwen2.5-VL-72B-Instruct}"
export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-Qwen2.5-VL-72B-Instruct__none__last1,Qwen2.5-VL-72B-Instruct__image_text_image__last1}"

unset REUSE_FROM_EXP_GROUP_TAG REUSE_FROM_EXP_DATE_TAG REUSE_FROM_SAVE_ROOT
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
