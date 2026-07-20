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
  [[ -x "${QWEN_TOKEN_ROLL_VLLM_PYTHON}" ]] || {
    echo "[FATAL] Qwen baseline vLLM Python not found: ${QWEN_TOKEN_ROLL_VLLM_PYTHON}" >&2
    exit 1
  }
  [[ -x "${MINICPM_TOKEN_ROLL_VLLM_PYTHON}" ]] || {
    echo "[FATAL] MiniCPM baseline vLLM Python not found: ${MINICPM_TOKEN_ROLL_VLLM_PYTHON}" >&2
    exit 1
  }
  "${CONTROL_PYTHON}" scripts/verify_visual_token_shift_runtime_20260719.py \
    --model-root "${MODEL_ROOT}" \
    --minicpm-transformers-pydeps "${MINICPM_TOKEN_ROLL_PYDEPS}" \
    "$@"
}

install_plugin_overlay() {
  "${CONTROL_PYTHON}" scripts/install_vllm_visual_token_shift_plugin.py \
    --target "${VLLM_TOKEN_ROLL_PLUGIN_OVERLAY}"
  PYTHONPATH="${REPO_ROOT}:${VLLM_TOKEN_ROLL_PLUGIN_OVERLAY}" \
    VLLM_USE_V1=0 \
    REPLAY_VISUAL_TOKEN_SHIFT=roll_right_1 \
    REPLAY_VLLM_TARGET_FAMILY=qwen2_5_vl \
    "${QWEN_TOKEN_ROLL_VLLM_PYTHON}" -c \
    'import inspect; from importlib.metadata import entry_points; eps=[ep for ep in entry_points(group="vllm.general_plugins") if ep.name == "vlmeval_visual_token_shift"]; assert len(eps) == 1; register=eps[0].load(); assert callable(register); register(); from vlmeval.vlm.replay_vllm_visual_token_models import ReplayShiftQwen2_5VL; sig=inspect.signature(ReplayShiftQwen2_5VL.__init__); assert ReplayShiftQwen2_5VL.replay_model_family == "qwen2_5_vl" and "vllm_config" in sig.parameters and "prefix" in sig.parameters'
  PYTHONPATH="${REPO_ROOT}:${VLLM_TOKEN_ROLL_PLUGIN_OVERLAY}" \
    REPLAY_VISUAL_TOKEN_SHIFT=roll_right_1 \
    REPLAY_VLLM_TARGET_FAMILY=minicpm_o_4_5 \
    "${MINICPM_TOKEN_ROLL_VLLM_PYTHON}" -c \
    'import inspect; from importlib.metadata import entry_points; eps=[ep for ep in entry_points(group="vllm.general_plugins") if ep.name == "vlmeval_visual_token_shift"]; assert len(eps) == 1; register=eps[0].load(); assert callable(register); register(); from vlmeval.vlm.replay_vllm_visual_token_models import ReplayShiftMiniCPMO45; sig=inspect.signature(ReplayShiftMiniCPMO45.__init__); assert ReplayShiftMiniCPMO45.replay_model_family == "minicpm_o_4_5" and "vllm_config" in sig.parameters and "prefix" in sig.parameters'
}

run_matrix() {
  bash scripts/run_benchmark.sh \
    --matrix-config "$1" \
    --visual-token-shifts "${VISUAL_TOKEN_SHIFTS}" \
    "${@:2}"
}

run_real_smoke_gate() {
  local smoke_summary="${REPO_ROOT}/runs/vllm_visual_token_shift_real_smoke_20260720_summary.json"
  GPU_IDS="${TOKEN_ROLL_SMOKE_GPU_IDS:-0,1,2,3}" \
    bash scripts/run_vllm_visual_token_shift_real_smokes_20260720.sh
  "${CONTROL_PYTHON}" -c \
    'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); data=json.loads(p.read_text()); assert data.get("all_passed") is True and (data.get("smoke_certificate") or {}).get("valid") is True, data' \
    "${smoke_summary}"
  export REPLAY_VISUAL_TOKEN_SHIFT_SMOKE_CERTIFICATE="${smoke_summary}"
}

case "${MODE}" in
  check)
    install_plugin_overlay
    check_runtime --require-openai-auth
    run_matrix "${INFER_CONFIG}" --plan-only "$@"
    ;;
  infer)
    install_plugin_overlay
    check_runtime
    run_real_smoke_gate
    run_matrix "${INFER_CONFIG}" "$@"
    ;;
  eval)
    install_plugin_overlay
    check_runtime --require-openai-auth
    run_matrix "${EVAL_CONFIG}" "$@"
    ;;
  all)
    install_plugin_overlay
    check_runtime --require-openai-auth
    run_real_smoke_gate
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
