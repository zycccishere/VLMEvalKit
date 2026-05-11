#!/usr/bin/env bash
set -euo pipefail

# Replay-8 sweep for Qwen2.5-VL-72B-Instruct only (single-node default).

export NUM_NODES="${NUM_NODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-1}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-8}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"

export LMUData="${LMUData:-/path/to/vlmevalkit/exp_debug/replay_8subsets_v1}"
export DATALIST="${DATALIST:-ReplayIconA_L2R ReplayIconA_R2L ReplayIconB_L2R ReplayIconB_R2L ReplayShapeA_L2R ReplayShapeA_R2L ReplayShapeB_L2R ReplayShapeB_R2L}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen25_72b_replay8_1node8gpu}"

export MODEL_PATH_QWEN25_72B="${MODEL_PATH_QWEN25_72B:-/models/Qwen2.5-VL-72B-Instruct}"
export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-Qwen2.5-VL-72B-Instruct__none__last1,Qwen2.5-VL-72B-Instruct__image_text_text__last1,Qwen2.5-VL-72B-Instruct__image_text_image__last1,Qwen2.5-VL-72B-Instruct__image_text_image_text__last1,Qwen2.5-VL-72B-Instruct__image_image_text__last1}"

# Align with validate TP setup for 72B.
unset VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS REPLAY_LIMIT_MM_PER_PROMPT INFER_BATCH_SIZE
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
export VLLM_MAX_MODEL_LEN_72B="${VLLM_MAX_MODEL_LEN_72B:-32768}"
export VLLM_MAX_NUM_SEQS_72B="${VLLM_MAX_NUM_SEQS_72B:-1}"
export REPLAY_LIMIT_MM_PER_PROMPT_72B="${REPLAY_LIMIT_MM_PER_PROMPT_72B:-2}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"

export FORCE_GPT_JUDGE_ALL="${FORCE_GPT_JUDGE_ALL:-1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_API_BASE_JUDGE="${OPENAI_API_BASE_JUDGE:-${OPENAI_API_BASE:-https://api.openai.com/v1}}"
if [[ -z "${OPENAI_API_KEY}" ]]; then
    echo "[FATAL] OPENAI_API_KEY is empty. Please export it before running." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_standard_qwen25_32b72b_newsets_last1_2node_8gpu_sweep.sh"
