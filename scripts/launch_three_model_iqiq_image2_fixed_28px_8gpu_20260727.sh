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

if [[ "${KEEP_PROXY:-0}" != "1" ]]; then
  unset http_proxy https_proxy all_proxy
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
fi

for path in \
  "${LMUData}" \
  "${MODEL_ROOT}/Qwen2.5-VL-32B-Instruct" \
  "${MODEL_ROOT}/Qwen2.5-VL-3B-Instruct" \
  "${MODEL_ROOT}/MiniCPM-o-4_5"
do
  [[ -e "${path}" ]] || { echo "[FATAL] missing required path: ${path}" >&2; exit 1; }
done

exec bash scripts/run_benchmark.sh \
  --matrix-config configs/matrix_three_model_iqiq_image2_fixed_28px_infer_20260727.yaml \
  --model-config configs/models_three_model_fixed_28px_20260727.yaml \
  --scheduler gpu_pool \
  --gpu-ids "${GPU_IDS:-0,1,2,3,4,5,6,7}" \
  --resume-infer \
  "$@"
