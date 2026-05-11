#!/usr/bin/env bash
set -euo pipefail

# Balanced node C:
# - 1 node
# - 8 jobs in parallel
# - 1 GPU/job
# - 30 tasks total => 30 / 8 = 3.75 waves
#
# Covers:
# - Qwen2-VL-7B-Instruct reasoning three sets
# - Qwen2.5-VL-7B-Instruct reasoning three sets

export NUM_NODES="${NUM_NODES:-1}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-8}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export EXP_GROUP_TAG="${EXP_GROUP_TAG:-repair_small_reasoning_heavy_1node8tasks_default_prompt}"
export DATALIST="${DATALIST:-VisuLogic LogicVista VisualPuzzles}"
export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-Qwen2-VL-7B-Instruct__none__last1,Qwen2-VL-7B-Instruct__image_text_text__last1,Qwen2-VL-7B-Instruct__image_text_image__last1,Qwen2-VL-7B-Instruct__image_text_image_text__last1,Qwen2-VL-7B-Instruct__image_image_text__last1,Qwen2.5-VL-7B-Instruct__none__last1,Qwen2.5-VL-7B-Instruct__image_text_text__last1,Qwen2.5-VL-7B-Instruct__image_text_image__last1,Qwen2.5-VL-7B-Instruct__image_text_image_text__last1,Qwen2.5-VL-7B-Instruct__image_image_text__last1}"

export REPLAY_PROMPT_TEMPLATE_NAME="identity"
unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../run_standard_qwen2_qwen25_newsets_last1_2node_8gpu_sweep.sh"
