#!/usr/bin/env bash
set -euo pipefail

# Direct-answer replay sweep for MiniCPM-V-4_5 on the three repaired datasets.
#
# Recommended topology in the 8-node plan:
# - 1 node
# - 8 jobs per node
# - 1 GPU per job

export NUM_NODES="${NUM_NODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-8}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LMUData="${LMUData:-$HOME/LMUData}"
export EXP_DATE_TAG="${EXP_DATE_TAG:-20260307}"

export DATALIST="${DATALIST:-VisuLogic LogicVista VisualPuzzles}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-three_sets_minicpm45_direct_last1_1node8workers}"
export MODEL_PATH_MINICPM45="${MODEL_PATH_MINICPM45:-/models/MiniCPM-V-4_5}"

export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME:-directly_answer}"
export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT:-1}"
unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../run_standard_minicpm45_newsets_last1_2node_8gpu_sweep.sh"
