#!/usr/bin/env bash
set -euo pipefail

# Newsets only: 3B补跑任务（每脚本2个任务）- part2.
# Tasks:
# - Qwen2.5-VL-3B-Instruct__image_text_image__last1
# - Qwen2.5-VL-3B-Instruct__image_text_image_text__last1

export NUM_NODES="${NUM_NODES:-2}"
export NODE_RANK="${NODE_RANK:-0}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-8}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export DATALIST="${DATALIST:-MathVision MathVista_MINI OCRBench}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen2_qwen25_newsets_last1_2node16gpu}"

export MODEL_PATH_QWEN25_3B="${MODEL_PATH_QWEN25_3B:-/models/Qwen2.5-VL-3B-Instruct}"
export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-Qwen2.5-VL-3B-Instruct__image_text_image__last1,Qwen2.5-VL-3B-Instruct__image_text_image_text__last1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_standard_qwen2_qwen25_newsets_last1_2node_8gpu_sweep.sh"
