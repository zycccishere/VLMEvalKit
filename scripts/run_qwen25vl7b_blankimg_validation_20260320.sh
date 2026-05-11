#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NODES="${NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
CONTROL_PYTHON="${CONTROL_PYTHON:-/opt/miniconda3/envs/vlmevalkit/bin/python}"
MATRIX_CONFIG="${MATRIX_CONFIG:-${SCRIPT_DIR}/configs/matrix_qwen25vl7b_blankimg_validation_20260320.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-${SCRIPT_DIR}/configs/models.yaml}"

cd "${REPO_ROOT}"

if [[ ! -x "${CONTROL_PYTHON}" ]]; then
  echo "[FATAL] control python not found: ${CONTROL_PYTHON}" >&2
  exit 1
fi

echo "[LAUNCH] qwen25vl7b blankimg validation"
echo "[LAUNCH] repo_root=${REPO_ROOT}"
echo "[LAUNCH] matrix=${MATRIX_CONFIG}"
echo "[LAUNCH] model_config=${MODEL_CONFIG}"
echo "[LAUNCH] gpu_ids=${GPU_IDS} nodes=${NODES} node_rank=${NODE_RANK}"

exec "${SCRIPT_DIR}/run_benchmark.sh" \
  --matrix-config "${MATRIX_CONFIG}" \
  --model-config "${MODEL_CONFIG}" \
  --gpu-ids "${GPU_IDS}" \
  --nodes "${NODES}" \
  --node-rank "${NODE_RANK}" \
  "$@"
