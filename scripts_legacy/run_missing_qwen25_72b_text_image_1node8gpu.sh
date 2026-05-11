#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[BUNDLE] qwen25-72b text_image directly_answer"
bash "${SCRIPT_DIR}/run_missing_qwen25_72b_text_image_direct_1node2workers_tp4.sh"

echo "[BUNDLE] qwen25-72b text_image identity"
exec bash "${SCRIPT_DIR}/run_missing_qwen25_72b_text_image_default_1node2workers_tp4.sh"
