#!/usr/bin/env bash
set -euo pipefail

# Run Qwen2.5-VL-{32B,7B}-Instruct on DynaMath only.
# Modes:
# - image_text
# - image_text_image
#
# Default split on one 8-GPU node:
# - 32B uses GPUs 0,1,2,3 with TP=2 and 2 workers
# - 7B uses GPUs 4,5 with 1 GPU/job and 2 workers
# - GPUs 6,7 remain free by default

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen25_72b32b7b_dynamath_2node_split_default_prompt}"
export DATALIST="${DATALIST:-DynaMath}"

PIDS=()

EXP_GROUP_TAG="${EXP_GROUP_TAG}" \
DATALIST="${DATALIST}" \
NUM_NODES=1 \
JOBS_PER_NODE="${JOBS_PER_NODE_32B:-2}" \
GPUS_PER_JOB=2 \
NODE_GPU_IDS="${NODE_GPU_IDS_32B:-0,1,2,3}" \
TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST_32B:-Qwen2.5-VL-32B-Instruct__none__last1,Qwen2.5-VL-32B-Instruct__image_text_image__last1}" \
bash "${SCRIPT_DIR}/run_standard_qwen25_32b_dynamath_2modes_1node.sh" &
PIDS+=("$!")

EXP_GROUP_TAG="${EXP_GROUP_TAG}" \
DATALIST="${DATALIST}" \
NUM_NODES=1 \
JOBS_PER_NODE="${JOBS_PER_NODE_7B:-2}" \
GPUS_PER_JOB=1 \
NODE_GPU_IDS="${NODE_GPU_IDS_7B:-4,5}" \
TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST_7B:-Qwen2.5-VL-7B-Instruct__none__last1,Qwen2.5-VL-7B-Instruct__image_text_image__last1}" \
REUSE_FROM_EXP_GROUP_TAG="" \
bash "${SCRIPT_DIR}/run_standard_qwen25_7b_dynamath_2modes_1node.sh" &
PIDS+=("$!")

rc=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        rc=1
    fi
done

exit "$rc"
