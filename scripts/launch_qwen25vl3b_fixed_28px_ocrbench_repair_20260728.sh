#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export LMUData="${LMUData:-/user/zyc1781/LMUData}"
export MODEL_ROOT="${MODEL_ROOT:-/user/zyc1781/models}"
export CONTROL_PYTHON="${CONTROL_PYTHON:-/user/wanzihao/miniconda3/envs/vlmevalkit/bin/python}"
export VLMEVAL_API_MINIMAL_IMPORT=1
export VLMEVAL_VLM_MINIMAL_IMPORT=1
export VLMEVAL_LAZY_INIT=1
export REPLAY_IMAGE_TRANSFORM_STRICT=1

unset http_proxy https_proxy all_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY

exec bash scripts/run_benchmark.sh \
  --matrix-config configs/matrix_qwen25vl3b_iqiq_image2_fixed_28px_ocrbench_repair_20260728.yaml \
  --model-config configs/models_three_model_fixed_28px_20260727.yaml \
  --scheduler gpu_pool \
  --gpu-ids "${GPU_IDS:-7}" \
  "$@"
