#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/vlmevalkit/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
COMMON_ARGS=(
  scripts/run_benchmark.py
  --matrix-config scripts/configs/matrix.yaml
  --model-config scripts/configs/models.yaml
  --nodes 1
  --node-rank 0
  --gpu-ids "$GPU_IDS"
  --policies default
)
EXTRA_ARGS=("$@")

run_subset() {
  local model="$1"
  local mode="$2"
  local datasets="$3"
  "$PYTHON_BIN" "${COMMON_ARGS[@]}" --models "$model" --modes "$mode" --datasets "$datasets" "${EXTRA_ARGS[@]}"
}

cd "$REPO_ROOT"

# Non-MathVision gaps first.
run_subset qwen35_35b_a3b image_text "MathVista_MINI"
run_subset qwen35_35b_a3b text_image "MathVista_MINI"
run_subset qwen35_35b_a3b image_text_text "MathVista_MINI,VisuLogic"
run_subset qwen35_35b_a3b image_text_image "MathVista_MINI"
run_subset qwen35_35b_a3b image_text_image_text "MathVista_MINI"
run_subset qwen35_35b_a3b image_image_text "MathVista_MINI,DynaMath"

run_subset qwen35_27b image_text "MathVista_MINI"
run_subset qwen35_27b text_image "MathVista_MINI"
run_subset qwen35_27b image_text_text "MathVista_MINI,VisualPuzzles,DynaMath"
run_subset qwen35_27b image_text_image "MathVista_MINI"
run_subset qwen35_27b image_text_image_text "MathVista_MINI,VisuLogic,VisualPuzzles"
run_subset qwen35_27b image_image_text "MathVista_MINI,DynaMath"

run_subset qwen35_9b image_text "MathVista_MINI"
run_subset qwen35_9b text_image "MathVista_MINI"
run_subset qwen35_9b image_text_text "MathVista_MINI"
run_subset qwen35_9b image_text_image "MathVista_MINI"
run_subset qwen35_9b image_text_image_text "MathVista_MINI"
run_subset qwen35_9b image_image_text "MathVista_MINI"

run_subset qwen35_4b image_text "MathVista_MINI,DynaMath"
run_subset qwen35_4b text_image "MathVista_MINI"
run_subset qwen35_4b image_text_text "MathVista_MINI"
run_subset qwen35_4b image_text_image "MathVista_MINI,VisuLogic"
run_subset qwen35_4b image_text_image_text "MathVista_MINI,VisuLogic"
run_subset qwen35_4b image_image_text "MathVista_MINI,VisuLogic"

# MathVision last by construction.
run_subset qwen35_35b_a3b image_text_text "MathVision"
run_subset qwen35_35b_a3b image_text_image_text "MathVision"
run_subset qwen35_35b_a3b image_image_text "MathVision"

run_subset qwen35_27b text_image "MathVision"
run_subset qwen35_27b image_text_text "MathVision"
run_subset qwen35_27b image_text_image "MathVision"
run_subset qwen35_27b image_text_image_text "MathVision"
run_subset qwen35_27b image_image_text "MathVision"

run_subset qwen35_9b image_text_image_text "MathVision"

run_subset qwen35_4b image_text_image "MathVision"
