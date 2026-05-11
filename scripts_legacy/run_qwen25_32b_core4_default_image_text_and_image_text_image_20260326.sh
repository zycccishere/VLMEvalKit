#!/usr/bin/env bash
set -euo pipefail

export NUM_NODES="${NUM_NODES:-1}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-4}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-2}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export DATALIST="${DATALIST:-OCRBench DynaMath LogicVista SEEDBench2_Plus}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen25_32b_core4_default_image_text_and_image_text_image_20260326}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_API_KEY_JUDGE="${OPENAI_API_KEY_JUDGE:-${OPENAI_API_KEY}}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
export OPENAI_API_BASE_JUDGE="${OPENAI_API_BASE_JUDGE:-${OPENAI_API_BASE}}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
export JUDGE_NPROC="${JUDGE_NPROC:-8}"

export MODEL_PATH_QWEN25_32B="${MODEL_PATH_QWEN25_32B:-/models/Qwen2.5-VL-32B-Instruct}"
export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-Qwen2.5-VL-32B-Instruct__image_text__last1,Qwen2.5-VL-32B-Instruct__image_text_image__last1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_standard_qwen25_32b_newsets_last1_tp2_3node_dataset_default.sh"
