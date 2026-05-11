#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/path/to/vlmevalkit"
PYTHON_BIN="/opt/miniconda3/envs/vlmevalkit/bin/python"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

"$PYTHON_BIN" scripts/launch_rope_align_image_text_image_probe_20260330.py \
  --models qwen25vl_32b \
  "$@"
