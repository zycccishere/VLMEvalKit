#!/usr/bin/env bash
set -euo pipefail

# Object HalBench sweep: 10 settings, 8 GPU (1 node, 2 jobs x 4 GPU).
# Requires OPENAI_API_KEY and COCO2014_ANNOTATIONS (e.g. /path/to/coco2014/annotations). Download: wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip && unzip -d coco2014 annotations_trainval2014.zip
# ObjHal data: obj_halbench_300_with_image.jsonl in RLHF-V-main/eval/data (already in repo).

export NUM_NODES="${NUM_NODES:-1}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-2}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen2_sweep10_objhal}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_API_BASE_JUDGE="${OPENAI_API_BASE_JUDGE:-${OPENAI_API_BASE:-https://api.openai.com/v1}}"

export DATALIST="${DATALIST:-ObjHal}"
export HAL_EVAL_DATASET="${HAL_EVAL_DATASET:-ObjHal}"
# All 10 settings (no allowlist)
export SETTING_TAG_ALLOWLIST="${SETTING_TAG_ALLOWLIST:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_standard_qwen2_5node_8gpu_sweep.sh" "$@"
