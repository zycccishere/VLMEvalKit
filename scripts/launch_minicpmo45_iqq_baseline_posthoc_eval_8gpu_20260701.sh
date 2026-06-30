#!/usr/bin/env bash
set -euo pipefail

export VLMEVAL_API_MINIMAL_IMPORT=1
export VLMEVAL_VLM_MINIMAL_IMPORT=1
export VLMEVAL_LAZY_INIT=1
export REPLAY_IMAGE_TRANSFORM_STRICT=1
export MINICPM45_USE_VLLM="${MINICPM45_USE_VLLM:-1}"

exec bash scripts/run_benchmark.sh \
  --matrix-config configs/matrix_minicpmo45_iqq_baseline_posthoc_eval_20260701.yaml \
  "$@"
