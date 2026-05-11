#!/usr/bin/env bash
set -euo pipefail

# Newsets only: run one 72B task (image_image_text).

export NUM_NODES="${NUM_NODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-1}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-8}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export DATALIST="${DATALIST:-MathVision OCRBench}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen25_72b_newsets_last1_1node8gpu}"

export MODEL_PATH_QWEN25_72B="${MODEL_PATH_QWEN25_72B:-/models/Qwen2.5-VL-72B-Instruct}"
export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-Qwen2.5-VL-72B-Instruct__image_image_text__last1}"

unset VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS REPLAY_LIMIT_MM_PER_PROMPT INFER_BATCH_SIZE
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
export VLLM_MAX_MODEL_LEN_72B="${VLLM_MAX_MODEL_LEN_72B:-32768}"
export VLLM_MAX_NUM_SEQS_72B="${VLLM_MAX_NUM_SEQS_72B:-1}"
export REPLAY_LIMIT_MM_PER_PROMPT_72B="${REPLAY_LIMIT_MM_PER_PROMPT_72B:-2}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_standard_qwen25_32b72b_newsets_last1_2node_8gpu_sweep.sh"
