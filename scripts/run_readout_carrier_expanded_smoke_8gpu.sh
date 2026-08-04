#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/user/zyc1781/vlmevalkit-release-readout-expanded-models}"
QWEN_PYTHON="${QWEN_PYTHON:-/user/wanzihao/miniconda3/envs/vlmevalkit/bin/python}"
MINICPM_PYTHON="${MINICPM_PYTHON:-/user/zhangyicheng/miniconda3/envs/duplex_mm_eval310/bin/python}"
MINICPM_PYDEPS="${MINICPM_PYDEPS:-/user/zyc1781/.venvs/minicpmo-token-roll-pydeps}"
QWEN32B_PATH="${QWEN32B_PATH:-/user/zyc1781/models/Qwen2.5-VL-32B-Instruct}"
MINICPMV_PATH="${MINICPMV_PATH:-/user/zyc1781/models/MiniCPM-V-4_5}"
LMU_DATA="${LMUData:-/user/zyc1781/LMUData}"
MATRIX_CONFIG="${MATRIX_CONFIG:-${ROOT}/configs/matrix.yaml}"
OUT_ROOT="${OUT_ROOT:-/user/zyc1781/outputs/readout_random_carriers/readout_carrier_expanded_smoke_20260805}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${OUT_ROOT}/manifests}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-128}"
SELECTION_SEED="${SELECTION_SEED:-20260804}"
ALL_DATASETS="DynaMath,WeMath,MMBench_DEV_EN_V11,MMStar,AI2D_TEST"

ALL_SINGLE_CHOICE_MANIFEST="${ALL_SINGLE_CHOICE_MANIFEST:-/user/zyc1781/outputs/readout_v2/qwen25vl3b_readout_v2_all_single_choice_20260729/manifest_all_single_choice.json}"
FIXED_CHOICE_MANIFEST="${FIXED_CHOICE_MANIFEST:-/user/zyc1781/outputs/readout_v2/qwen25vl3b_readout_v2_fixed_choice_20260728_v2/manifest.json}"
MMSTAR_AI2D_MANIFEST="${MMSTAR_AI2D_MANIFEST:-/user/zyc1781/outputs/readout_v2/qwen25vl3b_mmstar_ai2d_new_baseline_full_v1_20260729/manifest.json}"

IFS=',' read -r -a GPUS <<<"${GPU_IDS}"
if [[ "${#GPUS[@]}" -ne 8 ]]; then
  echo "Smoke requires exactly eight GPUs: ${GPU_IDS}" >&2
  exit 2
fi

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
for path in \
  "${QWEN_PYTHON}" "${MINICPM_PYTHON}" "${MINICPM_PYDEPS}" \
  "${QWEN32B_PATH}" "${MINICPMV_PATH}" "${MATRIX_CONFIG}" \
  "${ALL_SINGLE_CHOICE_MANIFEST}" "${FIXED_CHOICE_MANIFEST}" \
  "${MMSTAR_AI2D_MANIFEST}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path is missing: ${path}" >&2
    exit 2
  fi
done

rm -rf "${OUT_ROOT}"
mkdir -p "${MANIFEST_ROOT}" "${OUT_ROOT}/logs" "${OUT_ROOT}/runtime"
for model in qwen25vl_32b minicpm_v_45; do
  mkdir -p "${OUT_ROOT}/${model}/raw"
done

build_manifest() {
  local model_key="$1"
  local model_path="$2"
  local shards="$3"
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
    --num-shards "${shards}" \
    --samples-per-dataset "${SAMPLES_PER_DATASET}" \
    --selection-seed "${SELECTION_SEED}"
  "${QWEN_PYTHON}" -m vlmeval.probes.readout_carriers verify-run-contract \
    --repo-root "${ROOT}" \
    --manifest "${MANIFEST_ROOT}/${model_key}.json" \
    --model-path "${model_path}" \
    --lmu-data "${LMU_DATA}" \
    --matrix-config "${MATRIX_CONFIG}" \
    --output "${MANIFEST_ROOT}/${model_key}.run_contract.json"
}
build_manifest qwen25vl_32b "${QWEN32B_PATH}" 8
build_manifest minicpm_v_45 "${MINICPMV_PATH}" 32

monitor_pid=""
cleanup() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" >/dev/null 2>&1 || true
    wait "${monitor_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
(
  while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits | sed "s/^/$(date '+%F %T %Z'),/"
    sleep 10
  done
) >"${OUT_ROOT}/gpu_monitor.csv" 2>&1 &
monitor_pid="$!"

launch() {
  local gpu="$1"
  local model_key="$2"
  local model_path="$3"
  local python_bin="$4"
  local datasets="$5"
  local tag="$6"
  local diagnostics="$7"
  local worker_pythonpath="${ROOT}"
  local diagnostic_args=()
  if [[ "${model_key}" == minicpm_v_45 ]]; then
    worker_pythonpath="${ROOT}:${MINICPM_PYDEPS}"
  fi
  if [[ "${diagnostics}" == 1 ]]; then
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
qwen_datasets=(DynaMath WeMath MMBench_DEV_EN_V11 MMStar AI2D_TEST)
for offset in 0 1 2 3 4; do
  diagnostics=0
  [[ "${offset}" == 0 ]] && diagnostics=1
  launch "${GPUS[$offset]}" qwen25vl_32b "${QWEN32B_PATH}" "${QWEN_PYTHON}" \
    "${qwen_datasets[$offset]}" "dataset_${offset}" "${diagnostics}"
  pids+=("${LAUNCHED_PID}")
done
launch "${GPUS[5]}" minicpm_v_45 "${MINICPMV_PATH}" "${MINICPM_PYTHON}" \
  "DynaMath,WeMath" part_a 1
pids+=("${LAUNCHED_PID}")
launch "${GPUS[6]}" minicpm_v_45 "${MINICPMV_PATH}" "${MINICPM_PYTHON}" \
  "MMBench_DEV_EN_V11,MMStar" part_b 0
pids+=("${LAUNCHED_PID}")
launch "${GPUS[7]}" minicpm_v_45 "${MINICPMV_PATH}" "${MINICPM_PYTHON}" \
  "AI2D_TEST" part_c 0
pids+=("${LAUNCHED_PID}")

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
cleanup
monitor_pid=""
trap - EXIT
if [[ "${failed}" != 0 ]]; then
  echo "At least one real smoke worker failed" >&2
  exit 2
fi

for model in qwen25vl_32b minicpm_v_45; do
  "${QWEN_PYTHON}" -m vlmeval.probes.readout_carriers validate-smoke \
    --manifest "${MANIFEST_ROOT}/${model}.json" \
    --raw-root "${OUT_ROOT}/${model}/raw" \
    --output "${OUT_ROOT}/${model}/validation.json" \
    --expected-artifacts 5 \
    --require-diagnostics 1
done
echo "Smoke suite passed: ${OUT_ROOT}"
