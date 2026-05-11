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
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/runs/smoke/logicvista_minicpm_v45_${POLICY}_${MODE}_first${LIMIT}_use_vllm1}"
OPENAI_API_KEY_VALUE="${OPENAI_API_KEY_VALUE:-${OPENAI_API_KEY_JUDGE:-${OPENAI_API_KEY:-}}}"
OPENAI_API_BASE_VALUE="${OPENAI_API_BASE_VALUE:-https://api.openai.com/v1}"

source /opt/miniconda3/bin/activate
conda activate /opt/miniconda3/envs/vlmeval_qwen35_vllm

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export VLMEVAL_USE_MINICPM45_MINIMAL_CONFIG=1
export VLMEVAL_API_MINIMAL_IMPORT=1
export VLMEVAL_VLM_MINIMAL_IMPORT=1
export VLMEVAL_LAZY_INIT=1
export MINICPM45_USE_VLLM=1
export MODEL_PATH_MINICPM45="${MODEL_PATH_MINICPM45:-/models/MiniCPM-V-4_5}"
export MINICPM45_MAX_NEW_TOKENS="${MINICPM45_MAX_NEW_TOKENS:-16384}"
export MINICPM45_VLLM_TP_SIZE="${MINICPM45_VLLM_TP_SIZE:-1}"
export MINICPM45_VLLM_MAX_NUM_SEQS="${MINICPM45_VLLM_MAX_NUM_SEQS:-1}"
export MINICPM45_VLLM_MAX_MODEL_LEN="${MINICPM45_VLLM_MAX_MODEL_LEN:-32768}"
export MINICPM45_VLLM_GPU_MEMORY_UTILIZATION="${MINICPM45_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

python "${SCRIPT_DIR}/run_logicvista_subset_smoke.py" \
  --model MiniCPM-V-4_5-Replay \
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
