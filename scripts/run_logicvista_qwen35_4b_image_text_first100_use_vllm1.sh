#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/path/to/vlmevalkit"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS=("$@")
set --

LIMIT="${LIMIT:-100}"
POLICY="${POLICY:-identity}"
MODE="${MODE:-image_text}"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
API_NPROC="${API_NPROC:-4}"
EVAL_NPROC="${EVAL_NPROC:-4}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/runs/smoke/logicvista_qwen35_4b_${POLICY}_${MODE}_first${LIMIT}_use_vllm1}"
OPENAI_API_KEY_VALUE="${OPENAI_API_KEY_VALUE:-${OPENAI_API_KEY_JUDGE:-${OPENAI_API_KEY:-}}}"
OPENAI_API_BASE_VALUE="${OPENAI_API_BASE_VALUE:-https://api.openai.com/v1}"

source /opt/miniconda3/bin/activate
conda activate /opt/miniconda3/envs/vlmeval_qwen35_vllm

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export VLMEVAL_USE_QWEN_MINIMAL_CONFIG=1
export VLMEVAL_API_MINIMAL_IMPORT=1
export VLMEVAL_VLM_MINIMAL_IMPORT=1
export VLMEVAL_LAZY_INIT=1
export QWEN35_USE_VLLM=1
export QWEN35_VLLM_TP_SIZE="${QWEN35_VLLM_TP_SIZE:-1}"
export QWEN35_VLLM_MAX_NUM_SEQS="${QWEN35_VLLM_MAX_NUM_SEQS:-1}"
export QWEN35_VLLM_MAX_MODEL_LEN="${QWEN35_VLLM_MAX_MODEL_LEN:-32768}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

python "${SCRIPT_DIR}/run_logicvista_subset_smoke.py" \
  --model Qwen3.5-4B-Replay \
  --dataset LogicVista \
  --limit "${LIMIT}" \
  --policy "${POLICY}" \
  --mode "${MODE}" \
  --work-dir "${WORK_DIR}" \
  --judge "${JUDGE_MODEL}" \
  --api-nproc "${API_NPROC}" \
  --eval-nproc "${EVAL_NPROC}" \
  --batch-size "${BATCH_SIZE}" \
  --openai-api-key "${OPENAI_API_KEY_VALUE}" \
  --openai-api-base "${OPENAI_API_BASE_VALUE}" \
  "${EXTRA_ARGS[@]}"
