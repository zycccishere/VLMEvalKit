#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TOKEN_ROLL_USER_ROOT="${TOKEN_ROLL_USER_ROOT:-$(dirname "${REPO_ROOT}")}"
TOKEN_ROLL_RUNTIME_ROOT="${TOKEN_ROLL_RUNTIME_ROOT:-${TOKEN_ROLL_USER_ROOT}/.venvs}"
export LMUData="${LMUData:-${TOKEN_ROLL_USER_ROOT}/LMUData}"
export MODEL_ROOT="${MODEL_ROOT:-${TOKEN_ROLL_USER_ROOT}/models}"
export TOKEN_ROLL_PYTHON="${TOKEN_ROLL_PYTHON:-${TOKEN_ROLL_RUNTIME_ROOT}/lmms-engine/bin/python}"
export TOKEN_ROLL_PYDEPS="${TOKEN_ROLL_PYDEPS:-${TOKEN_ROLL_RUNTIME_ROOT}/vlmevalkit-token-roll-pydeps}"
export MINICPM_TOKEN_ROLL_PYDEPS="${MINICPM_TOKEN_ROLL_PYDEPS:-${TOKEN_ROLL_RUNTIME_ROOT}/minicpmo-token-roll-pydeps}"
export QWEN_TOKEN_ROLL_VLLM_PYTHON="${QWEN_TOKEN_ROLL_VLLM_PYTHON:-/user/wanzihao/miniconda3/envs/vlmevalkit/bin/python}"
export MINICPM_TOKEN_ROLL_VLLM_PYTHON="${MINICPM_TOKEN_ROLL_VLLM_PYTHON:-/user/zhangyicheng/miniconda3/envs/vlmeval_qwen35_vllm/bin/python}"
export VLLM_TOKEN_ROLL_PLUGIN_OVERLAY="${VLLM_TOKEN_ROLL_PLUGIN_OVERLAY:-${REPO_ROOT}/.vllm_plugin_overlay}"
export CONTROL_PYTHON="${CONTROL_PYTHON:-${TOKEN_ROLL_PYTHON}}"
export PYTHONPATH="${REPO_ROOT}:${TOKEN_ROLL_PYDEPS}${PYTHONPATH:+:${PYTHONPATH}}"
export REPLAY_VISUAL_TOKEN_SHIFT_STRICT=1
export VLMEVAL_API_MINIMAL_IMPORT=1
export VLMEVAL_VLM_MINIMAL_IMPORT=1
export VLMEVAL_LAZY_INIT=1

unset http_proxy https_proxy all_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY

declare -a labels=(qwen32_mathvision minicpm_mathvision qwen3b_ocrbench)
declare -a configs=(
  configs/matrix_retry_qwen32_mathvision_token_roll_20260722.yaml
  configs/matrix_retry_minicpm_mathvision_token_roll_20260722.yaml
  configs/matrix_retry_qwen3b_ocrbench_token_roll_20260722.yaml
)

if [[ "${1:-run}" == "plan" ]]; then
  for config in "${configs[@]}"; do
    bash scripts/run_benchmark.sh \
      --matrix-config "${config}" \
      --visual-token-shifts roll_right_1 \
      --no-resume-infer \
      --plan-only
  done
  exit 0
fi

SMOKE_SUMMARY="${REPO_ROOT}/runs/vllm_visual_token_shift_real_smoke_20260720_summary.json"
"${CONTROL_PYTHON}" -c \
  'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); assert d.get("all_passed") is True and (d.get("smoke_certificate") or {}).get("valid") is True, d' \
  "${SMOKE_SUMMARY}"
export REPLAY_VISUAL_TOKEN_SHIFT_SMOKE_CERTIFICATE="${SMOKE_SUMMARY}"

"${CONTROL_PYTHON}" scripts/install_vllm_visual_token_shift_plugin.py \
  --target "${VLLM_TOKEN_ROLL_PLUGIN_OVERLAY}"

LOG_ROOT="${REPO_ROOT}/runs/three_model_iqiq_image2_visual_token_roll_retries_20260722/_launcher"
mkdir -p "${LOG_ROOT}"

declare -a pids=()

for i in "${!labels[@]}"; do
  label="${labels[$i]}"
  config="${configs[$i]}"
  echo "[RETRY][START] ${label} config=${config}"
  bash scripts/run_benchmark.sh \
    --matrix-config "${config}" \
    --visual-token-shifts roll_right_1 \
    --no-resume-infer \
    >"${LOG_ROOT}/${label}.log" 2>&1 &
  pids+=("$!")
done

status=0
for i in "${!labels[@]}"; do
  label="${labels[$i]}"
  pid="${pids[$i]}"
  if wait "${pid}"; then
    echo "[RETRY][DONE] ${label}"
  else
    rc=$?
    echo "[RETRY][FAIL] ${label} rc=${rc} log=${LOG_ROOT}/${label}.log" >&2
    status=1
  fi
done

exit "${status}"
