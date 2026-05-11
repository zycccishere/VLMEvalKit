#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    cat >&2 <<'EOF'
Usage:
  bash scripts/run_qwen25vl_minicpm45_all4_reasoning_perception4_2nodes_20260422.sh <node_rank:0..1> [gpu_ids] [extra run_benchmark args...]

Examples:
  bash scripts/run_qwen25vl_minicpm45_all4_reasoning_perception4_2nodes_20260422.sh 0
  bash scripts/run_qwen25vl_minicpm45_all4_reasoning_perception4_2nodes_20260422.sh 1 0,1,2,3,4,5,6,7 --plan-only
EOF
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NODE_RANK="$1"
shift
if [[ ! "${NODE_RANK}" =~ ^[0-1]$ ]]; then
    echo "[FATAL] node_rank must be 0 or 1; got ${NODE_RANK}" >&2
    exit 2
fi

GPU_IDS="${QWEN25_MINICPM_ALL4_GPU_IDS:-0,1,2,3,4,5,6,7}"
if [[ $# -gt 0 && "$1" != --* ]]; then
    GPU_IDS="$1"
    shift
fi
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
GPUS_PER_NODE="${#GPU_ID_ARRAY[@]}"

MATRIX_CONFIG="scripts/configs/matrix_qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422.yaml"
MODEL_CONFIG="scripts/configs/models.yaml"
MANIFEST_DIR="scripts/configs/task_manifests/qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422"
MANIFEST="${MANIFEST_DIR}/node${NODE_RANK}_tasks.csv"
LOG_DIR="tmp/qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422"
LOG_PATH="${LOG_DIR}/node${NODE_RANK}_$(date +%Y%m%d%H%M%S).log"

if [[ ! -f "${MATRIX_CONFIG}" ]]; then
    echo "[FATAL] missing matrix config: ${MATRIX_CONFIG}" >&2
    exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "[INFO] missing task manifest; regenerating under ${MANIFEST_DIR}" >&2
    python scripts/prepare_qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422.py \
        --nodes 2 \
        --gpus-per-node "${GPUS_PER_NODE}"
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "[FATAL] missing task manifest after regeneration: ${MANIFEST}" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

echo "[LAUNCH] node_rank=${NODE_RANK} gpu_ids=${GPU_IDS} manifest=${MANIFEST} scheduler=gpu_pool"
echo "[LAUNCH] log=${LOG_PATH}"

bash scripts/run_benchmark.sh \
    --matrix-config "${MATRIX_CONFIG}" \
    --model-config "${MODEL_CONFIG}" \
    --nodes 2 \
    --node-rank "${NODE_RANK}" \
    --gpu-ids "${GPU_IDS}" \
    --task-manifest "${MANIFEST}" \
    --manifest-is-node-shard \
    --scheduler gpu_pool \
    "$@" 2>&1 | tee "${LOG_PATH}"
