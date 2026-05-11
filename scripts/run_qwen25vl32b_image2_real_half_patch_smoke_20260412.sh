#!/usr/bin/env bash
set -euo pipefail

cd /path/to/vlmevalkit
bash scripts/run_benchmark.sh \
  --matrix-config scripts/configs/matrix_qwen25vl32b_image2_real_half_patch_smoke_20260412.yaml \
  "$@"
