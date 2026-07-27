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

if [[ ! -s .env ]]; then
  echo "[FATAL] missing non-empty ${REPO_ROOT}/.env" >&2
  exit 1
fi

if [[ "${KEEP_PROXY:-0}" != "1" ]]; then
  unset http_proxy https_proxy all_proxy
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
fi

exec bash scripts/run_benchmark.sh \
  --matrix-config configs/matrix_three_model_iqiq_image2_fixed_28px_posthoc_eval_20260728.yaml \
  --model-config configs/models_three_model_fixed_28px_eval_20260728.yaml \
  --scheduler gpu_pool \
  --gpu-ids "${GPU_IDS:-0,1,2,3,4,5,6}" \
  --task-manifest configs/task_manifests/fixed_28px_posthoc_eval_20_valid_20260728.csv \
  --resume-infer \
  "$@"
