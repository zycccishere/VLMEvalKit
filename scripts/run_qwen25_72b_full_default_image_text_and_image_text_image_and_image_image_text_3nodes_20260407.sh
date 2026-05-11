#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_qwen25vl72b_full_3mode_20260407.py"
MANIFEST_DIR="${REPO_ROOT}/scripts/configs/task_manifests/qwen25vl72b_full_3mode_20260407"
MATRIX_CONFIG="scripts/configs/matrix_qwen25vl72b_full_3mode_20260407.yaml"

NODE_IDX="${NODE_IDX:-${1:-}}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

if [[ -z "${NODE_IDX}" ]]; then
  echo "Usage: NODE_IDX=0|1|2 bash scripts/$(basename "$0") [extra run_benchmark args]" >&2
  echo "   or: bash scripts/$(basename "$0") 0 [extra run_benchmark args]" >&2
  exit 1
fi

case "${NODE_IDX}" in
  0|1|2) ;;
  *)
    echo "[FATAL] NODE_IDX must be 0, 1, or 2: ${NODE_IDX}" >&2
    exit 1
    ;;
esac

if [[ "${1:-}" == "${NODE_IDX}" ]]; then
  shift
fi

MANIFEST_PATH="${MANIFEST_DIR}/node${NODE_IDX}_tasks.csv"
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  python3 "${PREPARE_SCRIPT}" --repo-root "${REPO_ROOT}" --nodes 3
fi

cd "${REPO_ROOT}"
NODE_RANK=0 NUM_NODES=1 REPLAY_TRACE_LEVEL=summary REPLAY_TRACE_SAMPLES=1 \
bash scripts/run_benchmark.sh \
  --matrix-config "${MATRIX_CONFIG}" \
  --task-manifest "scripts/configs/task_manifests/qwen25vl72b_full_3mode_20260407/node${NODE_IDX}_tasks.csv" \
  --nodes 1 \
  --node-rank 0 \
  --gpu-ids "${GPU_IDS}" \
  --resume-infer \
  "$@"
