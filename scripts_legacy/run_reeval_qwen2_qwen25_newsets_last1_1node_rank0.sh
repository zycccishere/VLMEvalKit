#!/usr/bin/env bash
set -euo pipefail

# Re-eval only: Qwen2/VL + Qwen2.5-VL Newsets, node rank 0.
# Use together with run_reeval_qwen2_qwen25_newsets_last1_1node_rank1.sh.
# Reuses infer under runs/standard/20260304/.

export EXP_DATE_TAG="${EXP_DATE_TAG:-20260304}"
export NUM_NODES="${NUM_NODES:-2}"
export NODE_RANK="${NODE_RANK:-0}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-8}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen2_qwen25_newsets_last1_2node16gpu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_reeval_qwen2_qwen25_sweep.sh"
