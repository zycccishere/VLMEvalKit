#!/usr/bin/env bash
set -euo pipefail

# MMHal-Bench sweep: 10 settings, 8 GPU (1 node, 2 jobs x 4 GPU).
# Requires OPENAI_API_KEY.
# Optional: OPENAI_API_BASE_JUDGE / OPENAI_API_BASE for proxy-compatible GPT judge calls.
# MMHal data: put mmhal-bench_with_image.jsonl in vlmevalkit/RLHF-V-main/eval/data (download from RLHF-V readme Google Drive link); else fallback uses mmhal-bench_answer_template.json (96 items, image_src URLs).

export RLHFV_DATA_ROOT="/path/to/vlmevalkit"
export NUM_NODES="${NUM_NODES:-1}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-2}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen2_sweep10_mmhal}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_API_BASE_JUDGE="${OPENAI_API_BASE_JUDGE:-${OPENAI_API_BASE:-https://api.openai.com/v1}}"

export DATALIST="${DATALIST:-MMHal}"
export HAL_EVAL_DATASET="${HAL_EVAL_DATASET:-MMHal}"
# All 10 settings (no allowlist)
export SETTING_TAG_ALLOWLIST="${SETTING_TAG_ALLOWLIST:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_standard_qwen2_5node_8gpu_sweep.sh" "$@"
