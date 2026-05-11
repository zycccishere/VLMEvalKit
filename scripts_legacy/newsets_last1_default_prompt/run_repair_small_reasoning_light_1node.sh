#!/usr/bin/env bash
set -euo pipefail

# Balanced node D:
# Phase 1: Qwen2-VL-2B-Instruct reasoning three sets => 15 tasks
# Phase 2: Qwen2.5-VL-3B-Instruct reasoning three sets => 15 tasks
# Phase 3: Qwen2.5-VL-3B-Instruct reasoning MathVision single repair => 1 task
#
# Total rough load:
# - 31 tasks on an 8-way 1-GPU scheduler => 31 / 8 = 3.875 waves

ROOT_DIR="/path/to/vlmevalkit/scripts/newsets_last1_default_prompt"

run_phase() {
    local datalist="$1"
    local allowlist="$2"
    local group_tag="$3"
    export NUM_NODES="${NUM_NODES:-1}"
    export JOBS_PER_NODE="${JOBS_PER_NODE:-8}"
    export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
    export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"
    export DATALIST="$datalist"
    export TASK_TAG_ALLOWLIST="$allowlist"
    export EXP_GROUP_TAG="$group_tag"
    export REPLAY_PROMPT_TEMPLATE_NAME="identity"
    unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE
    bash "${ROOT_DIR}/../run_standard_qwen2_qwen25_newsets_last1_2node_8gpu_sweep.sh"
}

run_phase \
    "VisuLogic LogicVista VisualPuzzles" \
    "Qwen2-VL-2B-Instruct__none__last1,Qwen2-VL-2B-Instruct__image_text_text__last1,Qwen2-VL-2B-Instruct__image_text_image__last1,Qwen2-VL-2B-Instruct__image_text_image_text__last1,Qwen2-VL-2B-Instruct__image_image_text__last1" \
    "${EXP_GROUP_TAG_PHASE1:-repair_small_reasoning_light_qwen2_2b_1node8tasks_default_prompt}"

run_phase \
    "VisuLogic LogicVista VisualPuzzles" \
    "Qwen2.5-VL-3B-Instruct__none__last1,Qwen2.5-VL-3B-Instruct__image_text_text__last1,Qwen2.5-VL-3B-Instruct__image_text_image__last1,Qwen2.5-VL-3B-Instruct__image_text_image_text__last1,Qwen2.5-VL-3B-Instruct__image_image_text__last1" \
    "${EXP_GROUP_TAG_PHASE2:-repair_small_reasoning_light_qwen25_3b_1node8tasks_default_prompt}"

run_phase \
    "MathVision" \
    "Qwen2.5-VL-3B-Instruct__image_text_image_text__last1" \
    "${EXP_GROUP_TAG_PHASE3:-repair_small_reasoning_light_qwen25_3b_mathvision_1node8tasks_default_prompt}"
