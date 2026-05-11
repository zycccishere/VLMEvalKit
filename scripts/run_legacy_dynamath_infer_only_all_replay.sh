#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/vlmevalkit/bin/python}"

export NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-1}}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3,4,5,6,7}"
export LMUData="${LMUData:-/datasets/vlmeval}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-$HF_ENDPOINT}"
export SEED_REUSE="${SEED_REUSE:-0}"
export EXCLUDE_STANDARD_20260409="${EXCLUDE_STANDARD_20260409:-0}"

MATRIX_CONFIG="${MATRIX_CONFIG:-${SCRIPT_DIR}/configs/matrix_legacy_dynamath_infer_only_all_replay.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-${SCRIPT_DIR}/configs/models_legacy_dynamath_infer_only.yaml}"

echo "[ENTRY][INFO] NUM_NODES=${NUM_NODES}"
echo "[ENTRY][INFO] NODE_RANK=${NODE_RANK:-auto}"
echo "[ENTRY][INFO] NODE_GPU_IDS=${NODE_GPU_IDS}"
echo "[ENTRY][INFO] LMUData=${LMUData}"
echo "[ENTRY][INFO] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[ENTRY][INFO] MATRIX_CONFIG=${MATRIX_CONFIG}"
echo "[ENTRY][INFO] MODEL_CONFIG=${MODEL_CONFIG}"
echo "[ENTRY][INFO] SEED_REUSE=${SEED_REUSE}"
echo "[ENTRY][INFO] EXCLUDE_STANDARD_20260409=${EXCLUDE_STANDARD_20260409}"

if [[ "${EXCLUDE_STANDARD_20260409}" == "1" ]]; then
  EXCLUDE_20260409_TAGS="qwen25vl_72b__default__image_text__DynaMath,qwen25vl_72b__default__image_text_image__DynaMath,qwen25vl_32b__default__image_text__DynaMath,qwen25vl_32b__default__image_text_image__DynaMath,qwen25vl_7b__default__image_text__DynaMath,qwen25vl_7b__default__image_text_image__DynaMath"
  if [[ -n "${TASK_TAG_EXCLUDELIST:-}" ]]; then
    export TASK_TAG_EXCLUDELIST="${TASK_TAG_EXCLUDELIST},${EXCLUDE_20260409_TAGS}"
  else
    export TASK_TAG_EXCLUDELIST="${EXCLUDE_20260409_TAGS}"
  fi
fi

if [[ -n "${TASK_TAG_EXCLUDELIST:-}" ]]; then
  echo "[ENTRY][INFO] TASK_TAG_EXCLUDELIST=${TASK_TAG_EXCLUDELIST}"
fi

if [[ "${SEED_REUSE}" == "1" ]]; then
  echo "[ENTRY][SEED] seeding reusable DynaMath infer artifacts"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/seed_legacy_dynamath_infer_reuse.py" \
    --matrix-config "${MATRIX_CONFIG}" \
    --model-config "${MODEL_CONFIG}"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_benchmark_task_balanced.py" \
  --matrix-config "${MATRIX_CONFIG}" \
  --model-config "${MODEL_CONFIG}" \
  --nodes "${NUM_NODES}" \
  --gpu-ids "${NODE_GPU_IDS}" \
  "$@"
