#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/vlmevalkit/bin/python}"
GPU_IDS="${GPU_IDS:-0}"
OUTER_WORKERS="${OUTER_WORKERS:-4}"
MANIFEST_PATH="${MANIFEST_PATH:-runs/manifests/bysetting_default_pending_eval_gpt4omini_20260318.json}"
DATASETS="${DATASETS:-AI2D_TEST MathVista_MINI OCRBench SEEDBench2_Plus LogicVista VisualPuzzles DynaMath MathVision}"

cd "$REPO_ROOT"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Missing manifest: $MANIFEST_PATH" >&2
  echo "Run scripts/freeze_default_eval_tasks_20260318.sh first." >&2
  exit 1
fi

"$PYTHON_BIN" scripts/rerun_eval_for_infer_complete_bysetting.py \
  --matrix-config scripts/configs/matrix.yaml \
  --model-config scripts/configs/models.yaml \
  --summary-json "$MANIFEST_PATH" \
  --policies default \
  --datasets "$DATASETS" \
  --gpu-ids "$GPU_IDS" \
  --outer-workers "$OUTER_WORKERS" \
  "$@"
