#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

export VLMEVAL_API_MINIMAL_IMPORT="${VLMEVAL_API_MINIMAL_IMPORT:-1}"
export VLMEVAL_VLM_MINIMAL_IMPORT="${VLMEVAL_VLM_MINIMAL_IMPORT:-1}"
export VLMEVAL_LAZY_INIT="${VLMEVAL_LAZY_INIT:-1}"
export REPLAY_IMAGE_TRANSFORM_STRICT="${REPLAY_IMAGE_TRANSFORM_STRICT:-1}"
export REPLAY_PROCESSOR_TRACE_VALIDATE="${REPLAY_PROCESSOR_TRACE_VALIDATE:-1}"
export REPLAY_PROCESSOR_TRACE_SAVE_NPZ="${REPLAY_PROCESSOR_TRACE_SAVE_NPZ:-0}"

exec bash scripts/run_benchmark.sh \
  --matrix-config configs/matrix_gemma3_12b_iqiq_image2_shift_fixed_20260629.yaml \
  "$@"
