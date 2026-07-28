#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/user/zyc1781/vlmevalkit-release-readout-v2}"
PYTHON_BIN="${PYTHON_BIN:-/user/wanzihao/miniconda3/envs/vlmevalkit/bin/python}"
MODEL_PATH="${MODEL_PATH:-/user/zyc1781/models/Qwen2.5-VL-3B-Instruct}"
LMU_DATA="${LMUData:-/user/zyc1781/LMUData}"
RUN_ID="${RUN_ID:-qwen25vl3b_readout_v2_all_single_choice_20260729}"
OUT_ROOT="${OUT_ROOT:-/user/zyc1781/outputs/readout_v2/${RUN_ID}}"
DEFAULT_REUSE_RUN_ID="qwen25vl3b_readout_v2_fixed_choice_20260728_v2"
DEFAULT_REUSE_ROOT="/user/zyc1781/outputs/readout_v2/${DEFAULT_REUSE_RUN_ID}"
REUSE_ROOT="${REUSE_ROOT:-${DEFAULT_REUSE_ROOT}}"
REUSE_MANIFEST="${REUSE_MANIFEST:-${REUSE_ROOT}/manifest.json}"
REUSE_PREDICTIONS="${REUSE_PREDICTIONS:-${REUSE_ROOT}/predictions}"
REUSE_LOCK="${REUSE_LOCK:-${REUSE_ROOT}/paired_statistics_clustered.json}"
DEFAULT_REUSE_LOCK_SHA256="c323795126a62aba4c299e718a5f94cb27c43b50bd145c0526fdf995a6bcc0e2"
REUSE_LOCK_SHA256="${REUSE_LOCK_SHA256:-${DEFAULT_REUSE_LOCK_SHA256}}"
REUSE_CODE_REF="${REUSE_CODE_REF:-}"
MATRIX_CONFIG="${MATRIX_CONFIG:-${ROOT}/configs/matrix.yaml}"
DATASETS="${DATASETS:-DynaMath,WeMath,SEEDBench2_Plus}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
RESUME="${RESUME:-0}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-30}"

if [[ "${SKIP_SMOKE}" == "1" && "${SMOKE_ONLY}" == "1" ]]; then
  echo "SKIP_SMOKE=1 and SMOKE_ONLY=1 are mutually exclusive" >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 8 ]]; then
  echo "This launcher requires exactly eight GPU ids; got ${GPU_IDS}" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id: ${gpu}" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
    echo "Duplicate GPU id: ${gpu}" >&2
    exit 2
  fi
  SEEN_GPUS[${gpu}]=1
done
NUM_SHARDS=8
ALL_MANIFEST="${OUT_ROOT}/manifest_all_single_choice.json"
MISSING_MANIFEST="${OUT_ROOT}/manifest_missing_only.json"
RUNTIME_ATTESTATION="${OUT_ROOT}/runtime_attestation.json"

export PYTHONPATH="${ROOT}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export LMUData="${LMU_DATA}"
export MODEL_ROOT="$(dirname "${MODEL_PATH}")"
export VLMEVAL_USE_QWEN_MINIMAL_CONFIG=1
export VLMEVAL_VLM_MINIMAL_IMPORT=1
export VLMEVAL_API_MINIMAL_IMPORT=1
export VLMEVAL_LAZY_INIT=1
export REPLAY_TRACE_LEVEL=off

cd "${ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to derive reusable results from a dirty repository: ${ROOT}" >&2
  exit 2
fi
for required in \
  "${PYTHON_BIN}" \
  "${MODEL_PATH}" \
  "${MATRIX_CONFIG}" \
  "${REUSE_MANIFEST}" \
  "${REUSE_LOCK}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path is missing: ${required}" >&2
    exit 2
  fi
done
if ! compgen -G "${REUSE_PREDICTIONS}/shard*.jsonl" >/dev/null; then
  echo "No accepted reuse predictions found under ${REUSE_PREDICTIONS}" >&2
  exit 2
fi
available_gpu_ids="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)"
for gpu in "${GPU_ARRAY[@]}"; do
  if ! grep -qx "${gpu}" <<<"${available_gpu_ids}"; then
    echo "Requested GPU ${gpu} is not visible to nvidia-smi" >&2
    exit 2
  fi
done

mkdir -p \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/predictions" \
  "${OUT_ROOT}/smoke/raw" \
  "${OUT_ROOT}/runtime"
if [[ "${SMOKE_ONLY}" != "1" ]]; then
  rm -f "${OUT_ROOT}/summary.json" "${OUT_ROOT}/accuracy.csv"
  rm -rf "${OUT_ROOT}/missing_aggregate"
fi

"${PYTHON_BIN}" -m vlmeval.probes.step8_readout_v2 manifest \
  --repo-root "${ROOT}" \
  --output "${ALL_MANIFEST}" \
  --datasets "${DATASETS}" \
  --selection-profile all_single_choice \
  --model-path "${MODEL_PATH}" \
  --lmu-data "${LMU_DATA}" \
  --matrix-config "${MATRIX_CONFIG}" \
  --num-shards "${NUM_SHARDS}"

reuse_code_args=()
if [[ -n "${REUSE_CODE_REF}" ]]; then
  reuse_code_args+=(--reuse-code-ref "${REUSE_CODE_REF}")
fi
"${PYTHON_BIN}" -m vlmeval.probes.step8_readout_v2 derive-missing \
  --all-manifest "${ALL_MANIFEST}" \
  --reuse-manifest "${REUSE_MANIFEST}" \
  --reuse-input-root "${REUSE_PREDICTIONS}" \
  --reuse-lock "${REUSE_LOCK}" \
  --reuse-lock-sha256 "${REUSE_LOCK_SHA256}" \
  --output "${MISSING_MANIFEST}" \
  --num-shards "${NUM_SHARDS}" \
  "${reuse_code_args[@]}"

"${PYTHON_BIN}" -m vlmeval.probes.step8_readout_v2 validate-selection \
  --all-manifest "${ALL_MANIFEST}" \
  --missing-manifest "${MISSING_MANIFEST}" \
  --output "${OUT_ROOT}/selection_validation.json"

"${PYTHON_BIN}" -m vlmeval.probes.step8_readout_v2 attest-runtime \
  --repo-root "${ROOT}" \
  --manifest "${ALL_MANIFEST}" \
  --output "${RUNTIME_ATTESTATION}" \
  --datasets "${DATASETS}" \
  --model-path "${MODEL_PATH}" \
  --lmu-data "${LMU_DATA}" \
  --matrix-config "${MATRIX_CONFIG}"

if [[ "${SKIP_SMOKE}" != "1" ]]; then
  if [[ "${RESUME}" != "1" ]]; then
    rm -rf "${OUT_ROOT}/smoke"
    mkdir -p "${OUT_ROOT}/smoke/raw"
  fi
  smoke_gpu="${GPU_ARRAY[0]}"
  CUDA_VISIBLE_DEVICES="${smoke_gpu}" "${PYTHON_BIN}" -m vlmeval.probes.step8_readout_v2 run \
    --repo-root "${ROOT}" \
    --manifest "${ALL_MANIFEST}" \
    --output-jsonl "${OUT_ROOT}/smoke/smoke.jsonl" \
    --runtime-root "${OUT_ROOT}/runtime/smoke" \
    --runtime-attestation "${RUNTIME_ATTESTATION}" \
    --datasets "${DATASETS}" \
    --model-path "${MODEL_PATH}" \
    --lmu-data "${LMU_DATA}" \
    --gpu-id "${smoke_gpu}" \
    --smoke-one-per-choice-count \
    --dump-raw-root "${OUT_ROOT}/smoke/raw" \
    --diagnostics \
    >"${OUT_ROOT}/logs/smoke.log" 2>&1

  "${PYTHON_BIN}" -m vlmeval.probes.step8_readout_v2 validate-smoke \
    --raw-root "${OUT_ROOT}/smoke/raw" \
    --manifest "${ALL_MANIFEST}" \
    --smoke-jsonl "${OUT_ROOT}/smoke/smoke.jsonl" \
    --reuse-reference-manifest "${REUSE_MANIFEST}" \
    --reuse-reference-root "${REUSE_PREDICTIONS}" \
    --expected-reuse-overlap 2 \
    --datasets "${DATASETS}" \
    --output "${OUT_ROOT}/smoke/validation.json"
fi

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  echo "Smoke passed: ${OUT_ROOT}/smoke/validation.json"
  exit 0
fi

if [[ "${RESUME}" != "1" ]]; then
  rm -f "${OUT_ROOT}"/predictions/shard*.jsonl
fi

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
    date '+%Y-%m-%d %H:%M:%S %Z'
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits || true
    sleep "${GPU_MONITOR_INTERVAL}"
  done
) >"${OUT_ROOT}/gpu_monitor.log" 2>&1 &
monitor_pid="$!"

pids=()
for rank in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$rank]}"
  resume_arg=()
  if [[ "${RESUME}" == "1" ]]; then
    resume_arg+=(--resume)
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_BIN}" -m vlmeval.probes.step8_readout_v2 run \
      --repo-root "${ROOT}" \
      --manifest "${MISSING_MANIFEST}" \
      --output-jsonl "${OUT_ROOT}/predictions/shard${rank}.jsonl" \
      --runtime-root "${OUT_ROOT}/runtime/shard${rank}" \
      --runtime-attestation "${RUNTIME_ATTESTATION}" \
      --datasets "${DATASETS}" \
      --model-path "${MODEL_PATH}" \
      --lmu-data "${LMU_DATA}" \
      --gpu-id "${gpu}" \
      --shard-rank "${rank}" \
      "${resume_arg[@]}"
  ) >"${OUT_ROOT}/logs/shard${rank}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "launched rank=${rank} gpu=${gpu} pid=$!"
done

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
  "${PYTHON_BIN}" -m vlmeval.probes.step8_readout_v2 aggregate \
    --manifest "${MISSING_MANIFEST}" \
    --input-root "${OUT_ROOT}/predictions" \
    --output-root "${OUT_ROOT}/missing_aggregate" || true
  echo "One or more inference shards failed; rerun with RESUME=1" >&2
  exit 2
fi

"${PYTHON_BIN}" -m vlmeval.probes.step8_readout_v2 aggregate \
  --manifest "${MISSING_MANIFEST}" \
  --input-root "${OUT_ROOT}/predictions" \
  --output-root "${OUT_ROOT}/missing_aggregate" \
  --require-complete

"${PYTHON_BIN}" -m vlmeval.probes.step8_readout_v2 aggregate-combined \
  --all-manifest "${ALL_MANIFEST}" \
  --reuse-manifest "${REUSE_MANIFEST}" \
  --missing-manifest "${MISSING_MANIFEST}" \
  --reuse-input-root "${REUSE_PREDICTIONS}" \
  --reuse-lock "${REUSE_LOCK}" \
  --missing-input-root "${OUT_ROOT}/predictions" \
  --output-root "${OUT_ROOT}" \
  --require-complete

echo "Done: ${OUT_ROOT}/accuracy.csv"
