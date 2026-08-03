#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/user/zyc1781/vlmevalkit-release-readout-carriers}"
QWEN_PYTHON="${QWEN_PYTHON:-/user/wanzihao/miniconda3/envs/vlmevalkit/bin/python}"
MINICPM_PYTHON="${MINICPM_PYTHON:-/user/zhangyicheng/miniconda3/envs/duplex_mm_eval310/bin/python}"
MINICPM_PYDEPS="${MINICPM_PYDEPS:-/user/zyc1781/.venvs/minicpmo-token-roll-pydeps}"
QWEN3B_PATH="${QWEN3B_PATH:-/user/zyc1781/models/Qwen2.5-VL-3B-Instruct}"
QWEN7B_PATH="${QWEN7B_PATH:-/user/zyc1781/models/Qwen2.5-VL-7B-Instruct}"
MINICPM_PATH="${MINICPM_PATH:-/user/zyc1781/models/MiniCPM-o-4_5}"
LMU_DATA="${LMUData:-/user/zyc1781/LMUData}"
MATRIX_CONFIG="${MATRIX_CONFIG:-${ROOT}/configs/matrix.yaml}"
RUN_ID="${RUN_ID:-readout_carrier_smoke_v3_20260803}"
OUT_ROOT="${OUT_ROOT:-/user/zyc1781/outputs/readout_carriers/${RUN_ID}}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${OUT_ROOT}/manifests}"
PREBUILT_MANIFESTS="${PREBUILT_MANIFESTS:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-10}"

ALL_SINGLE_CHOICE_MANIFEST="${ALL_SINGLE_CHOICE_MANIFEST:-/user/zyc1781/outputs/readout_v2/qwen25vl3b_readout_v2_all_single_choice_20260729/manifest_all_single_choice.json}"
FIXED_CHOICE_MANIFEST="${FIXED_CHOICE_MANIFEST:-/user/zyc1781/outputs/readout_v2/qwen25vl3b_readout_v2_fixed_choice_20260728_v2/manifest.json}"
MMSTAR_AI2D_MANIFEST="${MMSTAR_AI2D_MANIFEST:-/user/zyc1781/outputs/readout_v2/qwen25vl3b_mmstar_ai2d_new_baseline_full_v1_20260729/manifest.json}"
ALL_DATASETS="DynaMath,WeMath,MMBench_DEV_EN_V11,MMStar,AI2D_TEST"

IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 8 ]]; then
  echo "Smoke launcher requires exactly eight GPUs: ${GPU_IDS}" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
available_gpu_ids="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)"
for gpu in "${GPU_ARRAY[@]}"; do
  if ! [[ "${gpu}" =~ ^[0-9]+$ ]] || ! grep -qx "${gpu}" <<<"${available_gpu_ids}"; then
    echo "Invalid or unavailable GPU id: ${gpu}" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
    echo "GPU ids must be unique: ${GPU_IDS}" >&2
    exit 2
  fi
  SEEN_GPUS["${gpu}"]=1
done
OUTPUT_BASE="$(realpath -m /user/zyc1781/outputs/readout_carriers)"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
MANIFEST_ROOT="$(realpath -m "${MANIFEST_ROOT}")"
case "${OUT_ROOT}" in
  "${OUTPUT_BASE}"/*) ;;
  *)
    echo "Unsafe smoke OUT_ROOT: ${OUT_ROOT}" >&2
    exit 2
    ;;
esac
case "${MANIFEST_ROOT}" in
  "${OUT_ROOT}"/*) ;;
  *)
    echo "Unsafe smoke MANIFEST_ROOT: ${MANIFEST_ROOT}" >&2
    exit 2
    ;;
esac

export PYTHONPATH="${ROOT}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export LMUData="${LMU_DATA}"
export REPLAY_TRACE_LEVEL=off
export VLMEVAL_API_MINIMAL_IMPORT=1
export VLMEVAL_VLM_MINIMAL_IMPORT=1
export VLMEVAL_LAZY_INIT=1

cd "${ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to smoke a dirty repository: ${ROOT}" >&2
  exit 2
fi
required_paths=( \
  "${QWEN_PYTHON}" "${MINICPM_PYTHON}" \
  "${MINICPM_PYDEPS}" \
  "${QWEN3B_PATH}" "${QWEN7B_PATH}" "${MINICPM_PATH}" \
  "${MATRIX_CONFIG}" \
)
if [[ "${PREBUILT_MANIFESTS}" != "1" ]]; then
  required_paths+=(
    "${ALL_SINGLE_CHOICE_MANIFEST}"
    "${FIXED_CHOICE_MANIFEST}"
    "${MMSTAR_AI2D_MANIFEST}"
  )
fi
for required in "${required_paths[@]}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path is missing: ${required}" >&2
    exit 2
  fi
done
PYTHONPATH="${ROOT}:${MINICPM_PYDEPS}" "${MINICPM_PYTHON}" -c \
  'import librosa, torch, torchaudio, transformers; print(transformers.__version__, torch.__version__, torchaudio.__version__, librosa.__version__)'

if [[ "${PREBUILT_MANIFESTS}" == "1" ]]; then
  rm -rf \
    "${OUT_ROOT}/logs" "${OUT_ROOT}/runtime" \
    "${OUT_ROOT}/qwen25vl_3b" "${OUT_ROOT}/qwen25vl_7b" \
    "${OUT_ROOT}/minicpm_o_45" "${OUT_ROOT}/gpu_monitor.csv"
else
  rm -rf "${OUT_ROOT}"
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/runtime" "${MANIFEST_ROOT}"
for model in qwen25vl_3b qwen25vl_7b minicpm_o_45; do
  mkdir -p "${OUT_ROOT}/${model}/raw"
done

build_manifest() {
  local model_key="$1"
  local model_path="$2"
  "${QWEN_PYTHON}" -m vlmeval.probes.readout_carriers manifest \
    --repo-root "${ROOT}" \
    --output "${MANIFEST_ROOT}/${model_key}.json" \
    --all-single-choice-manifest "${ALL_SINGLE_CHOICE_MANIFEST}" \
    --fixed-choice-manifest "${FIXED_CHOICE_MANIFEST}" \
    --mmstar-ai2d-manifest "${MMSTAR_AI2D_MANIFEST}" \
    --lmu-data "${LMU_DATA}" \
    --matrix-config "${MATRIX_CONFIG}" \
    --datasets "${ALL_DATASETS}" \
    --model-key "${model_key}" \
    --model-path "${model_path}" \
    --num-shards 8
}

if [[ "${PREBUILT_MANIFESTS}" == "1" ]]; then
  for model in qwen25vl_3b qwen25vl_7b minicpm_o_45; do
    if [[ ! -f "${MANIFEST_ROOT}/${model}.json" ]]; then
      echo "Prebuilt smoke manifest is missing: ${MANIFEST_ROOT}/${model}.json" >&2
      exit 2
    fi
  done
else
  build_manifest qwen25vl_3b "${QWEN3B_PATH}"
  build_manifest qwen25vl_7b "${QWEN7B_PATH}"
  build_manifest minicpm_o_45 "${MINICPM_PATH}"
fi

verify_contract() {
  local model_key="$1"
  local model_path="$2"
  "${QWEN_PYTHON}" -m vlmeval.probes.readout_carriers verify-run-contract \
    --repo-root "${ROOT}" \
    --manifest "${MANIFEST_ROOT}/${model_key}.json" \
    --model-path "${model_path}" \
    --lmu-data "${LMU_DATA}" \
    --matrix-config "${MATRIX_CONFIG}" \
    --output "${MANIFEST_ROOT}/${model_key}.run_contract.json"
}
verify_contract qwen25vl_3b "${QWEN3B_PATH}"
verify_contract qwen25vl_7b "${QWEN7B_PATH}"
verify_contract minicpm_o_45 "${MINICPM_PATH}"

monitor_pid=""
cleanup_monitor() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" >/dev/null 2>&1 || true
    wait "${monitor_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_monitor EXIT
(
  while true; do
    timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits | sed "s/^/${timestamp},/" || true
    sleep "${GPU_MONITOR_INTERVAL}"
  done
) >"${OUT_ROOT}/gpu_monitor.csv" 2>&1 &
monitor_pid="$!"

launch_probe() {
  local gpu="$1"
  local model_key="$2"
  local model_path="$3"
  local python_bin="$4"
  local datasets="$5"
  local tag="$6"
  local diagnostics="$7"
  local worker_pythonpath="${ROOT}"
  if [[ "${model_key}" == "minicpm_o_45" ]]; then
    worker_pythonpath="${ROOT}:${MINICPM_PYDEPS}"
  fi
  local diagnostic_args=()
  if [[ "${diagnostics}" == "1" ]]; then
    diagnostic_args=(--diagnostics --diagnostics-limit 1)
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONPATH="${worker_pythonpath}"
    "${python_bin}" -m vlmeval.probes.readout_carriers run \
      --repo-root "${ROOT}" \
      --manifest "${MANIFEST_ROOT}/${model_key}.json" \
      --output-jsonl "${OUT_ROOT}/${model_key}/${tag}.jsonl" \
      --runtime-root "${OUT_ROOT}/runtime/${model_key}_${tag}" \
      --datasets "${datasets}" \
      --model-key "${model_key}" \
      --model-path "${model_path}" \
      --lmu-data "${LMU_DATA}" \
      --matrix-config "${MATRIX_CONFIG}" \
      --run-contract-attestation "${MANIFEST_ROOT}/${model_key}.run_contract.json" \
      --gpu-id "${gpu}" \
      --one-per-dataset \
      --dump-raw-root "${OUT_ROOT}/${model_key}/raw" \
      "${diagnostic_args[@]}"
  ) >"${OUT_ROOT}/logs/${model_key}_${tag}_gpu${gpu}.log" 2>&1 &
  LAUNCHED_PID="$!"
  echo "launched model=${model_key} tag=${tag} gpu=${gpu} pid=${LAUNCHED_PID}"
}

pids=()
launch_probe "${GPU_ARRAY[0]}" qwen25vl_3b "${QWEN3B_PATH}" "${QWEN_PYTHON}" "DynaMath,WeMath,MMBench_DEV_EN_V11" part_a 1
pids+=("${LAUNCHED_PID}")
launch_probe "${GPU_ARRAY[1]}" qwen25vl_3b "${QWEN3B_PATH}" "${QWEN_PYTHON}" "MMStar,AI2D_TEST" part_b 0
pids+=("${LAUNCHED_PID}")
launch_probe "${GPU_ARRAY[2]}" qwen25vl_7b "${QWEN7B_PATH}" "${QWEN_PYTHON}" "DynaMath,WeMath" part_a 1
pids+=("${LAUNCHED_PID}")
launch_probe "${GPU_ARRAY[3]}" qwen25vl_7b "${QWEN7B_PATH}" "${QWEN_PYTHON}" "MMBench_DEV_EN_V11,MMStar" part_b 0
pids+=("${LAUNCHED_PID}")
launch_probe "${GPU_ARRAY[4]}" qwen25vl_7b "${QWEN7B_PATH}" "${QWEN_PYTHON}" "AI2D_TEST" part_c 0
pids+=("${LAUNCHED_PID}")
launch_probe "${GPU_ARRAY[5]}" minicpm_o_45 "${MINICPM_PATH}" "${MINICPM_PYTHON}" "DynaMath,WeMath" part_a 1
pids+=("${LAUNCHED_PID}")
launch_probe "${GPU_ARRAY[6]}" minicpm_o_45 "${MINICPM_PATH}" "${MINICPM_PYTHON}" "MMBench_DEV_EN_V11,MMStar" part_b 0
pids+=("${LAUNCHED_PID}")
launch_probe "${GPU_ARRAY[7]}" minicpm_o_45 "${MINICPM_PATH}" "${MINICPM_PYTHON}" "AI2D_TEST" part_c 0
pids+=("${LAUNCHED_PID}")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
cleanup_monitor
monitor_pid=""
trap - EXIT
if [[ "${failed}" != "0" ]]; then
  echo "At least one real smoke worker failed" >&2
  exit 2
fi

for model in qwen25vl_3b qwen25vl_7b minicpm_o_45; do
  "${QWEN_PYTHON}" -m vlmeval.probes.readout_carriers validate-smoke \
    --manifest "${MANIFEST_ROOT}/${model}.json" \
    --raw-root "${OUT_ROOT}/${model}/raw" \
    --output "${OUT_ROOT}/${model}/validation.json" \
    --expected-artifacts 5 \
    --require-diagnostics 1
done

echo "Smoke suite passed: ${OUT_ROOT}"
