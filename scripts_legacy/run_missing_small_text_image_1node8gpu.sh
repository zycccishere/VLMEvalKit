#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[BUNDLE] small qwen text_image directly_answer"
bash "${SCRIPT_DIR}/run_missing_qwen2_qwen25_small_text_image_direct_1node8workers.sh"

echo "[BUNDLE] small qwen text_image identity"
exec bash "${SCRIPT_DIR}/run_missing_qwen2_qwen25_small_text_image_default_1node8workers.sh"
