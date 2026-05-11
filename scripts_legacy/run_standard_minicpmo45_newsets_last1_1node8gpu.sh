#!/usr/bin/env bash
set -euo pipefail

# Single-node 8-GPU entry for MiniCPM-o-4_5 replay newsets sweep.

export NUM_NODES="${NUM_NODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-8}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export DATALIST="${DATALIST:-AI2D_TEST DynaMath MathVision MathVista_MINI OCRBench SEEDBench2_Plus VisuLogic LogicVista VisualPuzzles}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-minicpmo45_newsets_last1_1node8gpu}"
export MODEL_TAG_MINICPM45="${MODEL_TAG_MINICPM45:-MiniCPM-o-4_5}"
export MODEL_NAME_MINICPM45="${MODEL_NAME_MINICPM45:-MiniCPM-o-4_5-Replay}"
export MODEL_PATH_MINICPM45="${MODEL_PATH_MINICPM45:-/models/MiniCPM-o-4_5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_standard_minicpm45_newsets_last1_2node_8gpu_sweep.sh"
