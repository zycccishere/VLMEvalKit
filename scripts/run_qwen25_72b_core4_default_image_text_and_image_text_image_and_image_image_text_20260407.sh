#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODELS="${MODELS:-qwen25vl_72b}"
export POLICIES="${POLICIES:-default}"
export MODES="${MODES:-image_text,image_text_image,image_image_text}"
export DATASETS="${DATASETS:-OCRBench,DynaMath,LogicVista,SEEDBench2_Plus}"
export NUM_NODES="${NUM_NODES:-1}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

exec bash "${SCRIPT_DIR}/run_replay6_core4_current_models_20260326.sh" "$@"
