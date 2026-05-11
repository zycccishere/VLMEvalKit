#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[BUNDLE] minicpm-v text_image directly_answer"
bash "${SCRIPT_DIR}/run_missing_minicpm45_text_image_direct_1node8gpu.sh"

echo "[BUNDLE] minicpm-v text_image identity"
bash "${SCRIPT_DIR}/run_missing_minicpm45_text_image_default_1node8gpu.sh"

echo "[BUNDLE] minicpm-o text_image directly_answer"
bash "${SCRIPT_DIR}/run_missing_minicpmo45_text_image_direct_1node8gpu.sh"

echo "[BUNDLE] minicpm-o text_image identity"
exec bash "${SCRIPT_DIR}/run_missing_minicpmo45_text_image_default_1node8gpu.sh"
