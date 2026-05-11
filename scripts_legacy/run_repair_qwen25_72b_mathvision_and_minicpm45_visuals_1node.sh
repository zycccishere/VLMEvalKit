#!/usr/bin/env bash
set -euo pipefail

# Balanced node B:
# Phase 1: Qwen2.5-VL-72B-Instruct MathVision
# - 1 node
# - 2 jobs in parallel
# - TP=4 (4 GPUs/job)
# - 5 tasks => 5 / 2 = 2.5 waves
#
# Phase 2: MiniCPM-V-4_5 VisualPuzzles + VisuLogic
# - 1 node
# - 8 jobs in parallel
# - 1 GPU/job
# - 10 tasks => 10 / 8 = 1.25 waves
#
# Combined rough load: 3.75 waves

ROOT_DIR="/path/to/vlmevalkit/scripts"

(
    export NUM_NODES="${NUM_NODES:-1}"
    export JOBS_PER_NODE="${JOBS_PER_NODE_72B:-2}"
    export GPUS_PER_JOB="${GPUS_PER_JOB_72B:-4}"
    export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

    export DATALIST="${DATALIST_72B:-MathVision}"
    export EXP_GROUP_TAG="${EXP_GROUP_TAG_72B:-repair_qwen25_72b_no_reasoning_mathvision_1node2workers_tp4_direct}"
    export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST_72B:-Qwen2.5-VL-72B-Instruct__none__last1,Qwen2.5-VL-72B-Instruct__image_text_text__last1,Qwen2.5-VL-72B-Instruct__image_text_image__last1,Qwen2.5-VL-72B-Instruct__image_text_image_text__last1,Qwen2.5-VL-72B-Instruct__image_image_text__last1}"
    export MODEL_PATH_QWEN25_72B="${MODEL_PATH_QWEN25_72B:-/models/Qwen2.5-VL-72B-Instruct}"

    unset VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS REPLAY_LIMIT_MM_PER_PROMPT INFER_BATCH_SIZE
    export VLLM_TP_SIZE="${VLLM_TP_SIZE_72B:-4}"
    export VLLM_MAX_MODEL_LEN_72B="${VLLM_MAX_MODEL_LEN_72B:-32768}"
    export VLLM_MAX_NUM_SEQS_72B="${VLLM_MAX_NUM_SEQS_72B:-1}"
    export REPLAY_LIMIT_MM_PER_PROMPT_72B="${REPLAY_LIMIT_MM_PER_PROMPT_72B:-2}"
    export INFER_BATCH_SIZE="${INFER_BATCH_SIZE_72B:-1}"

    export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME_72B:-directly_answer}"
    unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE

    bash "${ROOT_DIR}/run_standard_qwen25_32b72b_newsets_last1_dataset_sweep.sh"
)

(
    export NUM_NODES="${NUM_NODES:-1}"
    export JOBS_PER_NODE="${JOBS_PER_NODE_MINICPM:-8}"
    export GPUS_PER_JOB="${GPUS_PER_JOB_MINICPM:-1}"
    export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

    export EXP_GROUP_TAG="${EXP_GROUP_TAG_MINICPM:-repair_minicpm45_no_reasoning_visual_sets_1node8tasks_direct}"
    export DATALIST="${DATALIST_MINICPM:-VisualPuzzles VisuLogic}"
    export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST_MINICPM:-MiniCPM-V-4_5__none__last1,MiniCPM-V-4_5__image_text_text__last1,MiniCPM-V-4_5__image_text_image__last1,MiniCPM-V-4_5__image_text_image_text__last1,MiniCPM-V-4_5__image_image_text__last1}"

    export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME_MINICPM:-directly_answer}"
    unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE

    bash "${ROOT_DIR}/run_standard_minicpm45_newsets_last1_2node_8gpu_sweep.sh"
)
