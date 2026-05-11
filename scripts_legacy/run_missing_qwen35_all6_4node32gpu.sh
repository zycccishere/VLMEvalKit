#!/usr/bin/env bash
set -euo pipefail

export REPLAY_MODE_LIST="${REPLAY_MODE_LIST:-image_text,text_image,image_text_text,image_text_image,image_text_image_text,image_image_text}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-missing_qwen35_all6}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_standard_qwen35_newsets_last1_4node32gpu.sh"
