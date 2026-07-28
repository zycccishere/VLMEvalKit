#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/user/zyc1781/vlmevalkit-release-readout-v2}"
PYTHON_BIN="${PYTHON_BIN:-/user/wanzihao/miniconda3/envs/vlmevalkit/bin/python}"
MODEL_PATH="${MODEL_PATH:-/user/zyc1781/models/Qwen2.5-VL-3B-Instruct}"
LMU_DATA="${LMUData:-/user/zyc1781/LMUData}"
RUN_ID="${RUN_ID:-qwen25vl3b_readout_v2_fixed_choice_20260728_v2}"
OUT_ROOT="${OUT_ROOT:-/user/zyc1781/outputs/readout_v2/${RUN_ID}}"
MATRIX_CONFIG="${MATRIX_CONFIG:-${ROOT}/configs/matrix.yaml}"
DATASETS="${DATASETS:-DynaMath,WeMath,MMBench_DEV_EN_V11}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
RESUME="${RESUME:-0}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-30}"

IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
NUM_SHARDS="${#GPU_ARRAY[@]}"
MANIFEST="${OUT_ROOT}/manifest.json"

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
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/predictions" "${OUT_ROOT}/smoke/raw" "${OUT_ROOT}/runtime"

"${PYTHON_BIN}" -m vlmeval.probes.step8_readout_v2 manifest \
  --repo-root "${ROOT}" \
  --output "${MANIFEST}" \
  --datasets "${DATASETS}" \
  --model-path "${MODEL_PATH}" \
  --lmu-data "${LMU_DATA}" \
  --matrix-config "${MATRIX_CONFIG}" \
  --num-shards "${NUM_SHARDS}"

if [[ "${SKIP_SMOKE}" != "1" ]]; then
  if [[ "${RESUME}" != "1" ]]; then
    rm -rf "${OUT_ROOT}/smoke"
    mkdir -p "${OUT_ROOT}/smoke/raw"
  fi
  smoke_gpu="${GPU_ARRAY[0]}"
  CUDA_VISIBLE_DEVICES="${smoke_gpu}" "${PYTHON_BIN}" -m vlmeval.probes.step8_readout_v2 run \
    --repo-root "${ROOT}" \
    --manifest "${MANIFEST}" \
    --output-jsonl "${OUT_ROOT}/smoke/smoke.jsonl" \
    --runtime-root "${OUT_ROOT}/runtime/smoke" \
    --datasets "${DATASETS}" \
    --model-path "${MODEL_PATH}" \
    --lmu-data "${LMU_DATA}" \
    --gpu-id "${smoke_gpu}" \
    --smoke-one-per-dataset \
    --dump-raw-root "${OUT_ROOT}/smoke/raw" \
    --diagnostics \
    >"${OUT_ROOT}/logs/smoke.log" 2>&1

  "${PYTHON_BIN}" -m vlmeval.probes.step8_readout_v2 validate-smoke \
    --raw-root "${OUT_ROOT}/smoke/raw" \
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
      --manifest "${MANIFEST}" \
      --output-jsonl "${OUT_ROOT}/predictions/shard${rank}.jsonl" \
      --runtime-root "${OUT_ROOT}/runtime/shard${rank}" \
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
kill "${monitor_pid}" >/dev/null 2>&1 || true
wait "${monitor_pid}" >/dev/null 2>&1 || true

"${PYTHON_BIN}" -m vlmeval.probes.step8_readout_v2 aggregate \
  --manifest "${MANIFEST}" \
  --input-root "${OUT_ROOT}/predictions" \
  --output-root "${OUT_ROOT}" \
  --require-complete

if [[ "${failed}" != "0" ]]; then
  echo "One or more inference shards failed" >&2
  exit 2
fi

echo "Done: ${OUT_ROOT}/accuracy.csv"
