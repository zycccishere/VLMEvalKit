#!/usr/bin/env bash
set -euo pipefail

# Balanced node A:
# - 1 node
# - 4 jobs in parallel
# - TP=2 (2 GPUs/job)
# - 15 tasks total => 15 / 4 = 3.75 waves
#
# Covers missing reasoning raw scores for Qwen2.5-VL-32B-Instruct:
# - VisuLogic
# - LogicVista
# - VisualPuzzles

export NUM_NODES="${NUM_NODES:-1}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-4}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-2}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export DATALIST="${DATALIST:-VisuLogic LogicVista VisualPuzzles}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-repair_qwen25_32b_reasoning_three_sets_1node4workers_tp2_default_prompt}"

export MODEL_PATH_QWEN25_32B="${MODEL_PATH_QWEN25_32B:-/models/Qwen2.5-VL-32B-Instruct}"
export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-Qwen2.5-VL-32B-Instruct__none__last1,Qwen2.5-VL-32B-Instruct__image_text_text__last1,Qwen2.5-VL-32B-Instruct__image_text_image__last1,Qwen2.5-VL-32B-Instruct__image_text_image_text__last1,Qwen2.5-VL-32B-Instruct__image_image_text__last1}"

unset VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS REPLAY_LIMIT_MM_PER_PROMPT INFER_BATCH_SIZE
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
export VLLM_MAX_MODEL_LEN_32B="${VLLM_MAX_MODEL_LEN_32B:-32768}"
export VLLM_MAX_NUM_SEQS_32B="${VLLM_MAX_NUM_SEQS_32B:-1}"
export REPLAY_LIMIT_MM_PER_PROMPT_32B="${REPLAY_LIMIT_MM_PER_PROMPT_32B:-2}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"

export REPLAY_PROMPT_TEMPLATE_NAME="identity"
unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../run_standard_qwen25_32b72b_newsets_last1_dataset_sweep.sh"
