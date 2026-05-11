#!/usr/bin/env bash
set -euo pipefail

# Part 1b/5: 1-node job, 2 settings.
# Run this script on the single node in this allocation.

export NUM_NODES="${NUM_NODES:-1}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-2}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen2_sweep10_part1b_1node}"

export SETTING_TAG_ALLOWLIST="${SETTING_TAG_ALLOWLIST:-image_text_text__last0 image_text_text__last1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_standard_qwen2_5node_8gpu_sweep.sh"
