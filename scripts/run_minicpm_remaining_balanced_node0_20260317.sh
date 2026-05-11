#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/opt/miniconda3/envs/vlmevalkit/bin/python"
COMMON_ARGS=(
  scripts/run_benchmark.py
  --matrix-config scripts/configs/matrix_minicpm_default_infer_only_fresh_20260317.yaml
  --model-config scripts/configs/models.yaml
  --nodes 1
  --node-rank 0
  --gpu-ids 0,1,2,3,4,5,6,7
)

cd "$REPO_ROOT"

# Remaining MiniCPM-V tasks after the 2026-03-18 cleanup.
"$PYTHON_BIN" "${COMMON_ARGS[@]}" --models minicpm_v_45

# One full MiniCPM-o mode.
"$PYTHON_BIN" "${COMMON_ARGS[@]}" --models minicpm_o_45 --modes image_text

# Add the heavier text_image subsets to balance node 1.
"$PYTHON_BIN" "${COMMON_ARGS[@]}" --models minicpm_o_45 --modes text_image --datasets "VisuLogic,VisualPuzzles,DynaMath,MathVision"
