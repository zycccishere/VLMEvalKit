#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_SET=${MODEL_SET:-all}
RUN_NAME=${RUN_NAME:-"logicvista100_${MODEL_SET}_$(date +%Y%m%d_%H%M%S)"}
SESSION=${SESSION:-"logicvista100-attn-flow-${RUN_NAME}"}
OUTPUT_ROOT=${OUTPUT_ROOT:-/user/zyc1781/logicvista100_attention_flow_runs_20260701}
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/_tmux_logs"
mkdir -p "${LOG_DIR}"
ENV_FILE="${LOG_DIR}/run_env.sh"
TOPIC_LOGICVISTA100_MANIFEST="/Users/zhangyc/Desktop/WorkHub/assets/topics/topic-image-replay/resources/qwen25vl-shift-flow-100-20260629/logicvista_100_seed42_manifest.json"
DEFAULT_MANIFEST="${REPO_ROOT}/tmp/manifests/logicvista_100_seed42_manifest.json"
if [[ -z "${MANIFEST:-}" && -f "${TOPIC_LOGICVISTA100_MANIFEST}" ]]; then
  DEFAULT_MANIFEST="${TOPIC_LOGICVISTA100_MANIFEST}"
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[FATAL] tmux session already exists: ${SESSION}" >&2
  exit 1
fi

quote_export() {
  local key="$1"
  local value="$2"
  printf 'export %s=%q\n' "${key}" "${value}" >> "${ENV_FILE}"
}

: > "${ENV_FILE}"
quote_export MODEL_SET "${MODEL_SET}"
quote_export RUN_NAME "${RUN_NAME}"
quote_export OUTPUT_ROOT "${OUTPUT_ROOT}"
quote_export ALLOW_EXISTING "1"
quote_export WAIT_FOR_GPUS "${WAIT_FOR_GPUS:-1}"
quote_export WAIT_POLL_SECONDS "${WAIT_POLL_SECONDS:-60}"
quote_export GPU_MAX_USED_MB "${GPU_MAX_USED_MB:-2000}"
quote_export GPUS "${GPUS:-0,1,2,3,4,5,6,7}"
quote_export MANIFEST "${MANIFEST:-${DEFAULT_MANIFEST}}"
quote_export ATTN_LAYERS "${ATTN_LAYERS:-last}"
quote_export MODE "${MODE:-image_text_image}"
quote_export POLICY "${POLICY:-identity}"
quote_export TRANSFORMS "${TRANSFORMS:-shift_right_half_vit_token shift_right_one_vit_token shift_right_one_llm_token}"
quote_export SCALAR_RAW_DUMP_LIMIT "${SCALAR_RAW_DUMP_LIMIT:-0}"
quote_export SCALAR_QUERY_CHUNK_SIZE "${SCALAR_QUERY_CHUNK_SIZE:-256}"
quote_export MINICPM_MAX_SLICE_NUMS "${MINICPM_MAX_SLICE_NUMS:-1}"
quote_export QWEN25VL3B_MODEL_PATH "${QWEN25VL3B_MODEL_PATH:-/user/zyc1781/models/Qwen2.5-VL-3B-Instruct}"
quote_export GEMMA3_12B_MODEL_PATH "${GEMMA3_12B_MODEL_PATH:-/user/zyc1781/models/gemma-3-12b-it}"
quote_export MINICPMO45_MODEL_PATH "${MINICPMO45_MODEL_PATH:-/user/zyc1781/models/MiniCPM-o-4_5}"

tmux new-session -d -s "${SESSION}" \
  "source /user/zyc1781/activate_vlmevalkit_release.sh 2>/dev/null || true; source '${ENV_FILE}'; cd '${REPO_ROOT}'; bash scripts/launch_logicvista100_attention_flow_8gpu_20260701.sh 2>&1 | tee '${LOG_DIR}/launcher.log'"

echo "session=${SESSION}"
echo "run_root=${RUN_ROOT}"
echo "log=${LOG_DIR}/launcher.log"
