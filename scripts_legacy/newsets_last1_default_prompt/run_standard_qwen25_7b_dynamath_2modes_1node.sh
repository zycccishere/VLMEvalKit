#!/usr/bin/env bash
set -euo pipefail

# DynaMath-only sweep for Qwen2.5-VL-7B-Instruct.
# Modes:
# - image_text
# - image_text_image
#
# Default topology:
# - 1 node
# - 2 jobs in parallel
# - 1 GPU/job

export NUM_NODES="${NUM_NODES:-1}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-2}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1}"

export DATALIST="${DATALIST:-DynaMath}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen25_72b32b7b_dynamath_2node_split_default_prompt}"
export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-Qwen2.5-VL-7B-Instruct__none__last1,Qwen2.5-VL-7B-Instruct__image_text_image__last1}"

export MODEL_PATH_QWEN25_7B="${MODEL_PATH_QWEN25_7B:-/models/Qwen2.5-VL-7B-Instruct}"

unset REUSE_FROM_EXP_GROUP_TAG REUSE_FROM_EXP_DATE_TAG REUSE_FROM_SAVE_ROOT
export REPLAY_PROMPT_TEMPLATE_NAME="identity"
unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../run_standard_qwen2_qwen25_newsets_last1_2node_8gpu_sweep.sh"
