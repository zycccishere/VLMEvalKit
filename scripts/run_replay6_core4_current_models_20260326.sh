#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MATRIX_CONFIG="${MATRIX_CONFIG:-${SCRIPT_DIR}/configs/matrix_replay6_core4_current_models_20260326.yaml}"
NUM_NODES="${NUM_NODES:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MODELS="${MODELS:-qwen25vl_72b,qwen25vl_32b,qwen25vl_7b,qwen2vl_7b,minicpm_v_45,minicpm_o_45}"
POLICIES="${POLICIES:-direct,default}"
MODES="${MODES:-image_text,text_image,image_text_text,image_text_image,image_text_image_text,image_image_text}"
DATASETS="${DATASETS:-OCRBench,DynaMath,LogicVista,SEEDBench2_Plus}"

exec bash "${SCRIPT_DIR}/run_benchmark.sh" \
  --matrix-config "${MATRIX_CONFIG}" \
  --models "${MODELS}" \
  --policies "${POLICIES}" \
  --modes "${MODES}" \
  --datasets "${DATASETS}" \
  --nodes "${NUM_NODES}" \
  --gpu-ids "${GPU_IDS}" \
  "$@"
