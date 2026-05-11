#!/usr/bin/env bash
set -euo pipefail

# Repair missing reasoning-table raw scores for small models:
# - Qwen2-VL-{2B,7B}-Instruct
# - Qwen2.5-VL-{3B,7B}-Instruct
# Datasets:
# - VisuLogic
# - LogicVista
# - VisualPuzzles
#
# Topology:
# - 4 nodes
# - 8 jobs per node
# - 1 GPU per job
# - 32 task slots total.

export NUM_NODES="${NUM_NODES:-4}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-8}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export EXP_GROUP_TAG="${EXP_GROUP_TAG:-repair_qwen2_qwen25_small_reasoning_three_sets_4node32tasks_default_prompt}"
export DATALIST="${DATALIST:-VisuLogic LogicVista VisualPuzzles}"
export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-Qwen2-VL-2B-Instruct__none__last1,Qwen2-VL-2B-Instruct__image_text_text__last1,Qwen2-VL-2B-Instruct__image_text_image__last1,Qwen2-VL-2B-Instruct__image_text_image_text__last1,Qwen2-VL-2B-Instruct__image_image_text__last1,Qwen2-VL-7B-Instruct__none__last1,Qwen2-VL-7B-Instruct__image_text_text__last1,Qwen2-VL-7B-Instruct__image_text_image__last1,Qwen2-VL-7B-Instruct__image_text_image_text__last1,Qwen2-VL-7B-Instruct__image_image_text__last1,Qwen2.5-VL-3B-Instruct__none__last1,Qwen2.5-VL-3B-Instruct__image_text_text__last1,Qwen2.5-VL-3B-Instruct__image_text_image__last1,Qwen2.5-VL-3B-Instruct__image_text_image_text__last1,Qwen2.5-VL-3B-Instruct__image_image_text__last1,Qwen2.5-VL-7B-Instruct__none__last1,Qwen2.5-VL-7B-Instruct__image_text_text__last1,Qwen2.5-VL-7B-Instruct__image_text_image__last1,Qwen2.5-VL-7B-Instruct__image_text_image_text__last1,Qwen2.5-VL-7B-Instruct__image_image_text__last1}"

export REPLAY_PROMPT_TEMPLATE_NAME="identity"
unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../run_standard_qwen2_qwen25_newsets_last1_2node_8gpu_sweep.sh"
