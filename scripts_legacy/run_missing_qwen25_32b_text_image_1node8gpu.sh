#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[BUNDLE] qwen25-32b text_image directly_answer"
bash "${SCRIPT_DIR}/run_missing_qwen25_32b_text_image_direct_1node4workers_tp2.sh"

echo "[BUNDLE] qwen25-32b text_image identity"
exec bash "${SCRIPT_DIR}/run_missing_qwen25_32b_text_image_default_1node4workers_tp2.sh"
