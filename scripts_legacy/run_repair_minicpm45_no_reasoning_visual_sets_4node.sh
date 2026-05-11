#!/usr/bin/env bash
set -euo pipefail

# Repair missing non-reasoning visual-set raw scores for MiniCPM-V-4_5:
# - VisualPuzzles
# - VisuLogic
#
# This includes a few extra VisuLogic reruns beyond the one missing cell,
# but keeps the script simple and fully dataset-level.

export NUM_NODES="${NUM_NODES:-4}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-8}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export EXP_GROUP_TAG="${EXP_GROUP_TAG:-repair_minicpm45_no_reasoning_visual_sets_4node32tasks_direct}"
export DATALIST="${DATALIST:-VisualPuzzles VisuLogic}"
export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-MiniCPM-V-4_5__none__last1,MiniCPM-V-4_5__image_text_text__last1,MiniCPM-V-4_5__image_text_image__last1,MiniCPM-V-4_5__image_text_image_text__last1,MiniCPM-V-4_5__image_image_text__last1}"

export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME:-directly_answer}"
unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_standard_minicpm45_newsets_last1_2node_8gpu_sweep.sh"
