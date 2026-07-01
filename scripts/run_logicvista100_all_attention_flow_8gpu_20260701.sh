#!/usr/bin/env bash
set -euo pipefail

# One-command LogicVista100 attention-flow run.
# Runs Gemma3-12B, MiniCPM-o-4.5, and Qwen2.5-VL-3B sequentially; each model
# uses all selected GPUs through the underlying 8-GPU launchers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-/user/zyc1781/activate_vlmevalkit_release.sh}
if [[ -f "${ACTIVATE_SCRIPT}" ]]; then
  # Activation scripts in this workspace may change cwd, so cd back afterward.
  # shellcheck disable=SC1090
  source "${ACTIVATE_SCRIPT}"
fi
cd "${REPO_ROOT}"

MODEL_SET=${MODEL_SET:-all}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-0}
ATTN_LAYERS=${ATTN_LAYERS:-last}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${HOME}/logicvista100_attention_flow_runs_20260701"}
RUN_NAME=${RUN_NAME:-"logicvista100_all_$(date +%Y%m%d_%H%M%S)"}

MODE=${MODE:-image_text_image}
POLICY=${POLICY:-identity}
TRANSFORMS=${TRANSFORMS:-"shift_right_half_vit_token shift_right_one_vit_token shift_right_one_llm_token"}
SCALAR_RAW_DUMP_LIMIT=${SCALAR_RAW_DUMP_LIMIT:-0}
SCALAR_QUERY_CHUNK_SIZE=${SCALAR_QUERY_CHUNK_SIZE:-256}
MINICPM_MAX_SLICE_NUMS=${MINICPM_MAX_SLICE_NUMS:-1}

QWEN25VL3B_MODEL_PATH=${QWEN25VL3B_MODEL_PATH:-/user/zyc1781/models/Qwen2.5-VL-3B-Instruct}
GEMMA3_12B_MODEL_PATH=${GEMMA3_12B_MODEL_PATH:-/user/zyc1781/models/gemma-3-12b-it}
MINICPMO45_MODEL_PATH=${MINICPMO45_MODEL_PATH:-/user/zyc1781/models/MiniCPM-o-4_5}

if [[ -z "${MANIFEST:-}" ]]; then
  for candidate in \
    "${REPO_ROOT}/tmp/manifests/logicvista_100_seed42_manifest.json" \
    "/Users/zhangyc/Desktop/WorkHub/assets/topics/topic-image-replay/resources/qwen25vl-shift-flow-100-20260629/logicvista_100_seed42_manifest.json"
  do
    if [[ -f "${candidate}" ]]; then
      MANIFEST="${candidate}"
      break
    fi
  done
fi

if [[ "${MODEL_SET}" != "all" ]]; then
  echo "[FATAL] this convenience script is intended to run all three models; MODEL_SET must be all." >&2
  echo "For a single model, use scripts/launch_logicvista100_attention_flow_8gpu_20260701.sh directly." >&2
  exit 1
fi

if [[ -z "${MANIFEST:-}" || ! -f "${MANIFEST}" ]]; then
  echo "[FATAL] LogicVista100 manifest not found." >&2
  echo "Expected one of:" >&2
  echo "  ${REPO_ROOT}/tmp/manifests/logicvista_100_seed42_manifest.json" >&2
  echo "  /Users/zhangyc/Desktop/WorkHub/assets/topics/topic-image-replay/resources/qwen25vl-shift-flow-100-20260629/logicvista_100_seed42_manifest.json" >&2
  echo "Or set MANIFEST=/path/to/logicvista_100_seed42_manifest.json." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
LOG_FILE="${OUTPUT_ROOT}/${RUN_NAME}.launcher.log"

export MODEL_SET
export GPUS
export WAIT_FOR_GPUS
export ATTN_LAYERS
export OUTPUT_ROOT
export RUN_NAME
export MODE
export POLICY
export TRANSFORMS
export SCALAR_RAW_DUMP_LIMIT
export SCALAR_QUERY_CHUNK_SIZE
export MINICPM_MAX_SLICE_NUMS
export QWEN25VL3B_MODEL_PATH
export GEMMA3_12B_MODEL_PATH
export MINICPMO45_MODEL_PATH
export MANIFEST

cat <<EOF
[CONFIG] LogicVista100 attention-flow all-model run
  repo:        ${REPO_ROOT}
  manifest:    ${MANIFEST}
  output_root: ${OUTPUT_ROOT}
  run_name:    ${RUN_NAME}
  gpus:        ${GPUS}
  wait_gpus:   ${WAIT_FOR_GPUS}
  layers:      ${ATTN_LAYERS}
  transforms:  ${TRANSFORMS}
  log:         ${LOG_FILE}
EOF

bash scripts/launch_logicvista100_attention_flow_8gpu_20260701.sh 2>&1 | tee "${LOG_FILE}"
