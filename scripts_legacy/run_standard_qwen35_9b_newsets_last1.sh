#!/usr/bin/env bash
set -euo pipefail

export NUM_NODES="${NUM_NODES:-1}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-8}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export DATALIST="${DATALIST:-AI2D_TEST DynaMath MathVista_MINI OCRBench SEEDBench2_Plus VisuLogic LogicVista VisualPuzzles}"
export POLICY_LIST="${POLICY_LIST:-directly_answer identity}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen35_9b_newsets_last1_1node8workers}"

export MODEL_TAG_ALLOWLIST="${MODEL_TAG_ALLOWLIST:-Qwen3.5-9B-Replay}"
export MODEL_PATH_QWEN35_9B="${MODEL_PATH_QWEN35_9B:-/models/Qwen3.5-9B}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_standard_qwen35_newsets_last1_dataset_sweep.sh"
