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
export QWEN_TOKEN_ROLL_VLLM_PYTHON="${QWEN_TOKEN_ROLL_VLLM_PYTHON:-/user/wanzihao/miniconda3/envs/vlmevalkit/bin/python}"
export MINICPM_TOKEN_ROLL_VLLM_PYTHON="${MINICPM_TOKEN_ROLL_VLLM_PYTHON:-/user/zhangyicheng/miniconda3/envs/vlmeval_qwen35_vllm/bin/python}"
export VLLM_TOKEN_ROLL_PLUGIN_OVERLAY="${VLLM_TOKEN_ROLL_PLUGIN_OVERLAY:-${REPO_ROOT}/.vllm_plugin_overlay}"
export CONTROL_PYTHON="${CONTROL_PYTHON:-${TOKEN_ROLL_PYTHON}}"
export PYTHONPATH="${REPO_ROOT}:${TOKEN_ROLL_PYDEPS}${PYTHONPATH:+:${PYTHONPATH}}"
export REPLAY_VISUAL_TOKEN_SHIFT_VALIDATE_VALUES=1
export REPLAY_VISUAL_TOKEN_SHIFT_FULL_VALIDATION=1
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
export WORKER_MONITOR_INTERVAL_SECONDS="${WORKER_MONITOR_INTERVAL_SECONDS:-5}"
export VLMEVAL_API_MINIMAL_IMPORT=1
export VLMEVAL_VLM_MINIMAL_IMPORT=1
export VLMEVAL_LAZY_INIT=1

GPU_IDS="${GPU_IDS:-0,1,2,3}"
SMOKE_ROOT="${REPO_ROOT}/runs/three_model_iqiq_visual_token_roll_vllm_smoke_20260720"
SUMMARY="${REPO_ROOT}/runs/vllm_visual_token_shift_real_smoke_20260720_summary.json"

"${CONTROL_PYTHON}" scripts/install_vllm_visual_token_shift_plugin.py \
  --target "${VLLM_TOKEN_ROLL_PLUGIN_OVERLAY}"

PYTHONPATH="${REPO_ROOT}:${VLLM_TOKEN_ROLL_PLUGIN_OVERLAY}" \
  VLLM_USE_V1=0 \
  REPLAY_VISUAL_TOKEN_SHIFT=roll_right_1 \
  REPLAY_VLLM_TARGET_FAMILY=qwen2_5_vl \
  "${QWEN_TOKEN_ROLL_VLLM_PYTHON}" -c \
  'import inspect; from importlib.metadata import entry_points; ep=[x for x in entry_points(group="vllm.general_plugins") if x.name == "vlmeval_visual_token_shift"]; assert len(ep) == 1; ep[0].load()(); from vlmeval.vlm.replay_vllm_visual_token_models import ReplayShiftQwen2_5VL; sig=inspect.signature(ReplayShiftQwen2_5VL.__init__); assert ReplayShiftQwen2_5VL.replay_model_family == "qwen2_5_vl" and "vllm_config" in sig.parameters and "prefix" in sig.parameters'
PYTHONPATH="${REPO_ROOT}:${VLLM_TOKEN_ROLL_PLUGIN_OVERLAY}" \
  REPLAY_VISUAL_TOKEN_SHIFT=roll_right_1 \
  REPLAY_VLLM_TARGET_FAMILY=minicpm_o_4_5 \
  "${MINICPM_TOKEN_ROLL_VLLM_PYTHON}" -c \
  'import inspect; from importlib.metadata import entry_points; ep=[x for x in entry_points(group="vllm.general_plugins") if x.name == "vlmeval_visual_token_shift"]; assert len(ep) == 1; ep[0].load()(); from vlmeval.vlm.replay_vllm_visual_token_models import ReplayShiftMiniCPMO45; sig=inspect.signature(ReplayShiftMiniCPMO45.__init__); assert ReplayShiftMiniCPMO45.replay_model_family == "minicpm_o_4_5" and "vllm_config" in sig.parameters and "prefix" in sig.parameters'

bash scripts/run_benchmark.sh \
  --matrix-config configs/matrix_three_model_iqiq_visual_token_roll_vllm_smoke_20260720.yaml \
  --gpu-ids "${GPU_IDS}"

PYTHONPATH="${REPO_ROOT}:${TOKEN_ROLL_PYDEPS}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${CONTROL_PYTHON}" scripts/validate_vllm_visual_token_shift_real_dump_20260720.py \
  --root "${SMOKE_ROOT}" \
  --output "${SUMMARY}"
