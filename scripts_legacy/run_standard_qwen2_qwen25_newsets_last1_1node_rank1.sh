#!/usr/bin/env bash
set -euo pipefail

# Single-node entry for node rank 1.
# Use together with run_standard_qwen2_qwen25_newsets_last1_1node_rank0.sh.

export NUM_NODES="${NUM_NODES:-2}"
export NODE_RANK="${NODE_RANK:-1}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-8}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen2_qwen25_newsets_last1_2node16gpu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_standard_qwen2_qwen25_newsets_last1_2node_8gpu_sweep.sh"
