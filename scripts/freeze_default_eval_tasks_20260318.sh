#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/vlmevalkit/bin/python}"
GPU_IDS="${GPU_IDS:-0}"
MANIFEST_PATH="${MANIFEST_PATH:-runs/manifests/bysetting_default_pending_eval_gpt4omini_20260318.json}"
DATASETS="${DATASETS:-AI2D_TEST MathVista_MINI OCRBench SEEDBench2_Plus LogicVista VisualPuzzles DynaMath MathVision}"

cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/freeze_pending_eval_bysetting.py \
  --matrix-config scripts/configs/matrix.yaml \
  --model-config scripts/configs/models.yaml \
  --policies default \
  --datasets "$DATASETS" \
  --gpu-ids "$GPU_IDS" \
  --out-json "$MANIFEST_PATH"
