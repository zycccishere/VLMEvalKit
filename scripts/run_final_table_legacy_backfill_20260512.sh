#!/usr/bin/env bash
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SELF_DIR}/.." && pwd)"
CONTROL_PYTHON="${CONTROL_PYTHON:-python}"
GPU_IDS="${1:-${GPU_IDS:-0,1,2,3,4,5,6,7}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${REPO_ROOT}"

exec bash scripts/run_benchmark.sh \
  --matrix-config scripts/configs/matrix_final_table_legacy_backfill_20260512.yaml \
  --model-config scripts/configs/models.yaml \
  --scheduler gpu_pool \
  --gpu-ids "${GPU_IDS}" \
  --task-manifest scripts/configs/task_manifests/final_table_legacy_backfill_20260512/all_tasks.csv \
  ${EXTRA_ARGS}
