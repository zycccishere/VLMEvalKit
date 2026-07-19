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
export CONTROL_PYTHON="${CONTROL_PYTHON:-${TOKEN_ROLL_PYTHON}}"
export VISUAL_TOKEN_SHIFTS="${VISUAL_TOKEN_SHIFTS:-roll_right_1}"
export PYTHONPATH="${REPO_ROOT}:${TOKEN_ROLL_PYDEPS}${PYTHONPATH:+:${PYTHONPATH}}"
export REPLAY_VISUAL_TOKEN_SHIFT_STRICT=1
export VLMEVAL_API_MINIMAL_IMPORT=1
export VLMEVAL_VLM_MINIMAL_IMPORT=1
export VLMEVAL_LAZY_INIT=1

if [[ "${TOKEN_ROLL_KEEP_PROXY:-0}" != "1" ]]; then
  unset http_proxy https_proxy all_proxy
fi

MODE="${1:-infer}"
if [[ $# -gt 0 ]]; then
  shift
fi

INFER_CONFIG="configs/matrix_three_model_iqiq_image2_visual_token_roll_infer_20260719.yaml"
EVAL_CONFIG="configs/matrix_three_model_iqiq_image2_visual_token_roll_posthoc_eval_20260719.yaml"

check_runtime() {
  [[ -x "${TOKEN_ROLL_PYTHON}" ]] || {
    echo "[FATAL] token-roll Python not found: ${TOKEN_ROLL_PYTHON}" >&2
    exit 1
  }
  [[ -d "${LMUData}" ]] || {
    echo "[FATAL] LMUData not found: ${LMUData}" >&2
    exit 1
  }
  [[ -d "${MINICPM_TOKEN_ROLL_PYDEPS}" ]] || {
    echo "[FATAL] MiniCPM Transformers 4.51 overlay not found: ${MINICPM_TOKEN_ROLL_PYDEPS}" >&2
    exit 1
  }
  "${CONTROL_PYTHON}" scripts/verify_visual_token_shift_runtime_20260719.py \
    --model-root "${MODEL_ROOT}" \
    --minicpm-transformers-pydeps "${MINICPM_TOKEN_ROLL_PYDEPS}" \
    "$@"
}

run_matrix() {
  bash scripts/run_benchmark.sh \
    --matrix-config "$1" \
    --visual-token-shifts "${VISUAL_TOKEN_SHIFTS}" \
    "${@:2}"
}

case "${MODE}" in
  check)
    check_runtime --require-openai-auth
    run_matrix "${INFER_CONFIG}" --plan-only "$@"
    ;;
  infer)
    check_runtime
    run_matrix "${INFER_CONFIG}" "$@"
    ;;
  eval)
    check_runtime --require-openai-auth
    run_matrix "${EVAL_CONFIG}" "$@"
    ;;
  all)
    check_runtime --require-openai-auth
    run_matrix "${INFER_CONFIG}" "$@"
    run_matrix "${EVAL_CONFIG}" "$@"
    ;;
  plan)
    run_matrix "${INFER_CONFIG}" --plan-only "$@"
    ;;
  *)
    echo "Usage: $0 {check|infer|eval|all|plan} [run_benchmark args...]" >&2
    exit 2
    ;;
esac
