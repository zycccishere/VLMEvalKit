#!/usr/bin/env bash
set -euo pipefail

# Direct-answer replay sweep for:
# - Qwen2-VL-2B-Instruct
# - Qwen2-VL-7B-Instruct
# - Qwen2.5-VL-3B-Instruct
# - Qwen2.5-VL-7B-Instruct
#
# Recommended topology in the 8-node plan:
# - 2 nodes
# - 8 jobs per node
# - 1 GPU per job
# - 16 workers total

export NUM_NODES="${NUM_NODES:-2}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-8}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LMUData="${LMUData:-$HOME/LMUData}"
export EXP_DATE_TAG="${EXP_DATE_TAG:-20260307}"

export DATALIST="${DATALIST:-VisuLogic LogicVista VisualPuzzles}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-three_sets_qwen_small_direct_last1_2node16workers}"

export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME:-directly_answer}"
export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT:-1}"
unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../run_standard_qwen2_qwen25_newsets_last1_2node_8gpu_sweep.sh"
