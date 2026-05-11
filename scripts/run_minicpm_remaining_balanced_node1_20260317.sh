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

# The lighter text_image half for MiniCPM-o after the 2026-03-18 cleanup.
"$PYTHON_BIN" "${COMMON_ARGS[@]}" --models minicpm_o_45 --modes text_image --datasets "AI2D_TEST,MathVista_MINI,OCRBench,SEEDBench2_Plus,LogicVista"

# The remaining four full replay modes for MiniCPM-o.
"$PYTHON_BIN" "${COMMON_ARGS[@]}" --models minicpm_o_45 --modes "image_text_text,image_text_image,image_text_image_text,image_image_text"
