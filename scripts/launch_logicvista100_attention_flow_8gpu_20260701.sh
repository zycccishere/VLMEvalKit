#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_SET=${MODEL_SET:-all}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
TOPIC_LOGICVISTA100_MANIFEST="/Users/zhangyc/Desktop/WorkHub/assets/topics/topic-image-replay/resources/qwen25vl-shift-flow-100-20260629/logicvista_100_seed42_manifest.json"
if [[ -z "${MANIFEST:-}" ]]; then
  if [[ -f "${TOPIC_LOGICVISTA100_MANIFEST}" ]]; then
    MANIFEST="${TOPIC_LOGICVISTA100_MANIFEST}"
  else
    MANIFEST="${REPO_ROOT}/tmp/manifests/logicvista_100_seed42_manifest.json"
  fi
fi
OUTPUT_ROOT=${OUTPUT_ROOT:-/user/zyc1781/logicvista100_attention_flow_runs_20260701}
RUN_NAME=${RUN_NAME:-"logicvista100_${MODEL_SET}_$(date +%Y%m%d_%H%M%S)"}
ALLOW_EXISTING=${ALLOW_EXISTING:-0}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1}
WAIT_POLL_SECONDS=${WAIT_POLL_SECONDS:-60}
GPU_MAX_USED_MB=${GPU_MAX_USED_MB:-2000}

MODE=${MODE:-image_text_image}
POLICY=${POLICY:-identity}
ATTN_LAYERS=${ATTN_LAYERS:-last}
TRANSFORMS=${TRANSFORMS:-"shift_right_half_vit_token shift_right_one_vit_token shift_right_one_llm_token"}
SCALAR_RAW_DUMP_LIMIT=${SCALAR_RAW_DUMP_LIMIT:-0}
SCALAR_QUERY_CHUNK_SIZE=${SCALAR_QUERY_CHUNK_SIZE:-256}
MINICPM_MAX_SLICE_NUMS=${MINICPM_MAX_SLICE_NUMS:-1}

QWEN25VL3B_MODEL_PATH=${QWEN25VL3B_MODEL_PATH:-/user/zyc1781/models/Qwen2.5-VL-3B-Instruct}
GEMMA3_12B_MODEL_PATH=${GEMMA3_12B_MODEL_PATH:-/user/zyc1781/models/gemma-3-12b-it}
MINICPMO45_MODEL_PATH=${MINICPMO45_MODEL_PATH:-/user/zyc1781/models/MiniCPM-o-4_5}

DRY_RUN=${DRY_RUN:-0}

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[FATAL] LogicVista100 manifest not found: ${MANIFEST}" >&2
  echo "Set MANIFEST=/path/to/logicvista_100_seed42_manifest.json or sync it to the default path." >&2
  exit 1
fi

case "${MODEL_SET}" in
  all)
    MODELS=(gemma3_12b minicpmo45 qwen25vl3b)
    ;;
  gemma3_12b|gemma3)
    MODELS=(gemma3_12b)
    ;;
  minicpmo45|minicpm-o-4_5|minicpm_o_45)
    MODELS=(minicpmo45)
    ;;
  qwen25vl3b|qwen2.5vl3b|qwen3b)
    MODELS=(qwen25vl3b)
    ;;
  *)
    echo "[FATAL] unsupported MODEL_SET=${MODEL_SET}; use all, gemma3_12b, minicpmo45, or qwen25vl3b" >&2
    exit 1
    ;;
esac

RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
if [[ -e "${RUN_ROOT}" && "${ALLOW_EXISTING}" != "1" ]]; then
  if [[ -n "$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "[FATAL] output run root already exists and is non-empty: ${RUN_ROOT}" >&2
    echo "Set RUN_NAME to a fresh value or ALLOW_EXISTING=1 to resume intentionally." >&2
    exit 1
  fi
fi
mkdir -p "${RUN_ROOT}"

wait_for_gpus() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  if [[ "${WAIT_FOR_GPUS}" != "1" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[FATAL] WAIT_FOR_GPUS=1 but nvidia-smi is not available." >&2
    exit 1
  fi
  echo "[WAIT] WAIT_FOR_GPUS=1; waiting for selected GPUs (${GPUS}) to have memory.used <= ${GPU_MAX_USED_MB} MiB"
  while true; do
    busy="$(
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F',' -v selected=",${GPUS}," -v threshold="${GPU_MAX_USED_MB}" '
            {
              gsub(/ /, "", $1); gsub(/ /, "", $2);
              idx=$1; mem=$2 + 0;
              if (index(selected, "," idx ",") && mem > threshold) {
                printf("%s:%d ", idx, mem);
              }
            }'
    )"
    if [[ -z "${busy}" ]]; then
      echo "[WAIT] GPUs are free enough; launching."
      return
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] busy GPUs: ${busy}; sleep ${WAIT_POLL_SECONDS}s"
    sleep "${WAIT_POLL_SECONDS}"
  done
}

write_run_metadata() {
  cat > "${RUN_ROOT}/run_config.json" <<JSON
{
  "manifest": "${MANIFEST}",
  "output_root": "${OUTPUT_ROOT}",
  "run_name": "${RUN_NAME}",
  "run_root": "${RUN_ROOT}",
  "model_set": "${MODEL_SET}",
  "models": [$(printf '"%s",' "${MODELS[@]}" | sed 's/,$//')],
  "gpus": "${GPUS}",
  "mode": "${MODE}",
  "policy": "${POLICY}",
  "attn_layers": "${ATTN_LAYERS}",
  "transforms": "${TRANSFORMS}",
  "scalar_raw_dump_limit": ${SCALAR_RAW_DUMP_LIMIT},
  "scalar_query_chunk_size": ${SCALAR_QUERY_CHUNK_SIZE},
  "minicpm_max_slice_nums": ${MINICPM_MAX_SLICE_NUMS},
  "qwen25vl3b_model_path": "${QWEN25VL3B_MODEL_PATH}",
  "gemma3_12b_model_path": "${GEMMA3_12B_MODEL_PATH}",
  "minicpmo45_model_path": "${MINICPMO45_MODEL_PATH}",
  "started_at": "$(date -Iseconds)"
}
JSON
}

run_gemma3_12b() {
  local out="${RUN_ROOT}/gemma3_12b"
  echo "[RUN] gemma3_12b -> ${out}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  MODEL_FAMILY=gemma3 \
  MODEL_PATH="${GEMMA3_12B_MODEL_PATH}" \
  GPUS="${GPUS}" \
  ATTN_LAYERS="${ATTN_LAYERS}" \
  MODE="${MODE}" \
  POLICY="${POLICY}" \
  TRANSFORMS="${TRANSFORMS}" \
  SCALAR_RAW_DUMP_LIMIT="${SCALAR_RAW_DUMP_LIMIT}" \
  SCALAR_QUERY_CHUNK_SIZE="${SCALAR_QUERY_CHUNK_SIZE}" \
    bash scripts/launch_hf_shift_flow_8gpu_20260701.sh "${MANIFEST}" "${out}"
}

run_minicpmo45() {
  local out="${RUN_ROOT}/minicpmo45"
  echo "[RUN] minicpmo45 -> ${out}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  MODEL_FAMILY=minicpm-o-4_5 \
  MODEL_PATH="${MINICPMO45_MODEL_PATH}" \
  GPUS="${GPUS}" \
  ATTN_LAYERS="${ATTN_LAYERS}" \
  MODE="${MODE}" \
  POLICY="${POLICY}" \
  TRANSFORMS="${TRANSFORMS}" \
  SCALAR_RAW_DUMP_LIMIT="${SCALAR_RAW_DUMP_LIMIT}" \
  SCALAR_QUERY_CHUNK_SIZE="${SCALAR_QUERY_CHUNK_SIZE}" \
  MINICPM_MAX_SLICE_NUMS="${MINICPM_MAX_SLICE_NUMS}" \
    bash scripts/launch_hf_shift_flow_8gpu_20260701.sh "${MANIFEST}" "${out}"
}

run_qwen25vl3b() {
  local out="${RUN_ROOT}/qwen25vl3b"
  echo "[RUN] qwen25vl3b -> ${out}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  MODEL_PATH="${QWEN25VL3B_MODEL_PATH}" \
  GPUS="${GPUS}" \
  ATTN_LAYERS="${ATTN_LAYERS}" \
  MODE="${MODE}" \
  POLICY="${POLICY}" \
  TRANSFORMS="${TRANSFORMS}" \
  DUMP_MODE=scalar \
  SCALAR_RAW_DUMP_LIMIT="${SCALAR_RAW_DUMP_LIMIT}" \
  SCALAR_QUERY_CHUNK_SIZE="${SCALAR_QUERY_CHUNK_SIZE}" \
    bash scripts/launch_qwen25vl_shift_flow_8gpu_20260629.sh "${MANIFEST}" "${out}"
}

write_run_metadata
wait_for_gpus

for model in "${MODELS[@]}"; do
  case "${model}" in
    gemma3_12b) run_gemma3_12b ;;
    minicpmo45) run_minicpmo45 ;;
    qwen25vl3b) run_qwen25vl3b ;;
  esac
done

echo "[DONE] LogicVista100 attention-flow run root: ${RUN_ROOT}"
