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

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

exec bash scripts/run_benchmark.sh \
  --matrix-config configs/matrix_three_model_image2_noise_posthoc_eval_20260805.yaml \
  --model-config configs/models_three_model_image2_noise_eval_20260805.yaml \
  --task-manifest configs/task_manifests/image2_noise_content_valid_conditions_20260805.csv \
  --scheduler gpu_pool \
  --gpu-ids "${GPU_IDS:-0,1,2,3,4,5,6,7}" \
  --resume-infer \
  "$@"
