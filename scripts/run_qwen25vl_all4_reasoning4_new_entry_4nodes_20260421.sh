#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    cat >&2 <<'EOF'
Usage:
  bash scripts/run_qwen25vl_all4_reasoning4_new_entry_4nodes_20260421.sh <node_rank:0..3> [gpu_ids] [extra run_benchmark args...]

Examples:
  bash scripts/run_qwen25vl_all4_reasoning4_new_entry_4nodes_20260421.sh 0
  bash scripts/run_qwen25vl_all4_reasoning4_new_entry_4nodes_20260421.sh 1 0,1,2,3
  bash scripts/run_qwen25vl_all4_reasoning4_new_entry_4nodes_20260421.sh 2 0,1,2,3,4,5,6,7 --plan-only
EOF
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NODE_RANK="$1"
shift
if [[ ! "${NODE_RANK}" =~ ^[0-3]$ ]]; then
    echo "[FATAL] node_rank must be one of 0,1,2,3; got ${NODE_RANK}" >&2
    exit 2
fi

GPU_IDS="${QWEN25VL_REASONING4_GPU_IDS:-0,1,2,3,4,5,6,7}"
if [[ $# -gt 0 && "$1" != --* ]]; then
    GPU_IDS="$1"
    shift
fi

MATRIX_CONFIG="scripts/configs/matrix_qwen25vl_all4_reasoning4_new_entry_20260421.yaml"
MODEL_CONFIG="scripts/configs/models.yaml"
MANIFEST="scripts/configs/task_manifests/qwen25vl_all4_reasoning4_new_entry_20260421/node${NODE_RANK}_tasks.csv"
LOG_DIR="tmp/qwen25vl_all4_reasoning4_new_entry_20260421"
LOG_PATH="${LOG_DIR}/node${NODE_RANK}_$(date +%Y%m%d%H%M%S).log"

if [[ ! -f "${MATRIX_CONFIG}" ]]; then
    echo "[FATAL] missing matrix config: ${MATRIX_CONFIG}" >&2
    exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "[FATAL] missing task manifest: ${MANIFEST}" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

echo "[LAUNCH] node_rank=${NODE_RANK} gpu_ids=${GPU_IDS} manifest=${MANIFEST}"
echo "[LAUNCH] log=${LOG_PATH}"

bash scripts/run_benchmark.sh \
    --matrix-config "${MATRIX_CONFIG}" \
    --model-config "${MODEL_CONFIG}" \
    --nodes 4 \
    --node-rank "${NODE_RANK}" \
    --gpu-ids "${GPU_IDS}" \
    --task-manifest "${MANIFEST}" \
    --manifest-is-node-shard \
    "$@" 2>&1 | tee "${LOG_PATH}"
