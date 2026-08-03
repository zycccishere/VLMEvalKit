#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/user/zyc1781/vlmevalkit-release-readout-carriers}"
MODEL_KEY="${MODEL_KEY:?Set MODEL_KEY to qwen25vl_3b, qwen25vl_7b, or minicpm_o_45}"
DATASETS="${DATASETS:-DynaMath,WeMath,MMBench_DEV_EN_V11,MMStar,AI2D_TEST}"
PARTITION_TAG="${PARTITION_TAG:-all}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
LMU_DATA="${LMUData:-/user/zyc1781/LMUData}"
MATRIX_CONFIG="${MATRIX_CONFIG:-${ROOT}/configs/matrix.yaml}"
RUN_ID="${RUN_ID:-readout_carriers_20260803}"
OUT_ROOT="${OUT_ROOT:-/user/zyc1781/outputs/readout_carriers/${RUN_ID}/${MODEL_KEY}/${PARTITION_TAG}}"
RESUME="${RESUME:-0}"
REPAIR_TORN_JSONL="${REPAIR_TORN_JSONL:-0}"
ALLOW_MULTI_MODEL_PER_GPU="${ALLOW_MULTI_MODEL_PER_GPU:-0}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-20}"
WAVE_DELAY_SECONDS="${WAVE_DELAY_SECONDS:-30}"

case "${MODEL_KEY}" in
  qwen25vl_3b)
    PYTHON_BIN="${PYTHON_BIN:-/user/wanzihao/miniconda3/envs/vlmevalkit/bin/python}"
    MODEL_PATH="${MODEL_PATH:-/user/zyc1781/models/Qwen2.5-VL-3B-Instruct}"
    SMOKE_VALIDATION="${SMOKE_VALIDATION:-/user/zyc1781/outputs/readout_carriers/readout_carrier_smoke_20260803/qwen25vl_3b/validation.json}"
    ;;
  qwen25vl_7b)
    PYTHON_BIN="${PYTHON_BIN:-/user/wanzihao/miniconda3/envs/vlmevalkit/bin/python}"
    MODEL_PATH="${MODEL_PATH:-/user/zyc1781/models/Qwen2.5-VL-7B-Instruct}"
    SMOKE_VALIDATION="${SMOKE_VALIDATION:-/user/zyc1781/outputs/readout_carriers/readout_carrier_smoke_20260803/qwen25vl_7b/validation.json}"
    ;;
  minicpm_o_45)
    PYTHON_BIN="${PYTHON_BIN:-/user/wanzihao/miniconda3/envs/vlmevalkit_minicpmv/bin/python}"
    MODEL_PATH="${MODEL_PATH:-/user/zyc1781/models/MiniCPM-o-4_5}"
    SMOKE_VALIDATION="${SMOKE_VALIDATION:-/user/zyc1781/outputs/readout_carriers/readout_carrier_smoke_20260803/minicpm_o_45/validation.json}"
    ;;
  *)
    echo "Unsupported MODEL_KEY: ${MODEL_KEY}" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 8 ]]; then
  echo "Full launcher requires exactly eight GPUs: ${GPU_IDS}" >&2
  exit 2
fi
if ! [[ "${WORKERS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS_PER_GPU must be a positive integer" >&2
  exit 2
fi
if ((WORKERS_PER_GPU > 1)) && [[ "${ALLOW_MULTI_MODEL_PER_GPU}" != "1" ]]; then
  echo "WORKERS_PER_GPU>1 requires ALLOW_MULTI_MODEL_PER_GPU=1 after a measured smoke" >&2
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
TOTAL_WORKERS=$((8 * WORKERS_PER_GPU))
DEFAULT_SMOKE_MANIFEST="/user/zyc1781/outputs/readout_carriers/readout_carrier_smoke_20260803/manifests/${MODEL_KEY}.json"
MANIFEST="${MANIFEST_PATH:-${DEFAULT_SMOKE_MANIFEST}}"
OUTPUT_BASE="$(realpath -m /user/zyc1781/outputs/readout_carriers)"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
case "${OUT_ROOT}" in
  "${OUTPUT_BASE}"/*) ;;
  *)
    echo "Unsafe full-run OUT_ROOT: ${OUT_ROOT}" >&2
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
  echo "Refusing to run a dirty repository: ${ROOT}" >&2
  exit 2
fi
for required in \
  "${PYTHON_BIN}" "${MODEL_PATH}" "${MATRIX_CONFIG}" \
  "${SMOKE_VALIDATION}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path is missing: ${required}" >&2
    exit 2
  fi
done
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/predictions" "${OUT_ROOT}/runtime"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Accepted smoke manifest is missing: ${MANIFEST}" >&2
  exit 2
fi
RUN_CONTRACT="${OUT_ROOT}/run_contract_attestation.json"
"${PYTHON_BIN}" -m vlmeval.probes.readout_carriers verify-run-contract \
  --repo-root "${ROOT}" \
  --manifest "${MANIFEST}" \
  --model-path "${MODEL_PATH}" \
  --lmu-data "${LMU_DATA}" \
  --matrix-config "${MATRIX_CONFIG}" \
  --output "${RUN_CONTRACT}"
"${PYTHON_BIN}" - "${MANIFEST}" "${MODEL_KEY}" "${TOTAL_WORKERS}" "${DATASETS}" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected_datasets = [item for item in sys.argv[4].split(",") if item]
actual_datasets = [item["dataset"] for item in manifest["datasets"]]
if manifest["model_key"] != sys.argv[2]:
    raise SystemExit("Prebuilt manifest model mismatch")
if int(manifest["num_shards"]) != int(sys.argv[3]):
    raise SystemExit("Prebuilt manifest shard-count mismatch")
if any(item not in actual_datasets for item in expected_datasets):
    raise SystemExit(
        f"Prebuilt manifest does not cover requested datasets: "
        f"requested={expected_datasets} actual={actual_datasets}"
    )
PY
"${PYTHON_BIN}" - "${SMOKE_VALIDATION}" "${MANIFEST}" <<'PY'
import hashlib
import json
import sys

validation = json.load(open(sys.argv[1], encoding="utf-8"))
manifest_path = sys.argv[2]
manifest = json.load(open(manifest_path, encoding="utf-8"))
manifest_sha = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
if validation.get("passed") is not True:
    raise SystemExit(f"Smoke gate did not pass: {sys.argv[1]}")
if validation.get("manifest_sha256") != manifest_sha:
    raise SystemExit("Smoke gate belongs to a different manifest")
if validation.get("model_key") != manifest.get("model_key"):
    raise SystemExit("Smoke gate belongs to a different model key")
if validation.get("model_identity") != manifest.get("model_identity"):
    raise SystemExit("Smoke gate belongs to a different checkpoint")
expected = {
    "manifest_sha256": manifest_sha,
    "manifest_schema": manifest["schema"],
    "manifest_records_sha256": manifest["records_sha256"],
    "implementation_sha256": manifest["implementation_sha256"],
    "repo_commit": manifest["repo_commit"],
    "model_key": manifest["model_key"],
    "model_family": manifest["model_family"],
    "model_identity_sha256": manifest["model_identity_sha256"],
    "source_data_sha256": manifest["source_data_sha256"],
    "matrix_config_sha256": manifest["matrix_config"]["sha256"],
    "models_config_sha256": manifest["models_config"]["sha256"],
}
if validation.get("provenance") != expected:
    raise SystemExit("Smoke provenance does not match the current run contract")
PY

if [[ "${RESUME}" != "1" ]]; then
  rm -f "${OUT_ROOT}"/predictions/worker*.jsonl
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
    timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits | sed "s/^/${timestamp},/" || true
    sleep "${GPU_MONITOR_INTERVAL}"
  done
) >"${OUT_ROOT}/gpu_monitor.csv" 2>&1 &
monitor_pid="$!"

pids=()
for ((slot = 0; slot < WORKERS_PER_GPU; slot++)); do
  for gpu_offset in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$gpu_offset]}"
    rank=$((slot * 8 + gpu_offset))
    resume_args=()
    if [[ "${RESUME}" == "1" ]]; then
      resume_args=(--resume)
    fi
    if [[ "${REPAIR_TORN_JSONL}" == "1" ]]; then
      resume_args+=(--repair-torn-jsonl)
    fi
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      "${PYTHON_BIN}" -m vlmeval.probes.readout_carriers run \
        --repo-root "${ROOT}" \
        --manifest "${MANIFEST}" \
        --output-jsonl "${OUT_ROOT}/predictions/worker${rank}.jsonl" \
        --runtime-root "${OUT_ROOT}/runtime/worker${rank}" \
        --datasets "${DATASETS}" \
        --model-key "${MODEL_KEY}" \
        --model-path "${MODEL_PATH}" \
        --lmu-data "${LMU_DATA}" \
        --matrix-config "${MATRIX_CONFIG}" \
        --run-contract-attestation "${RUN_CONTRACT}" \
        --expected-runtime-validation "${SMOKE_VALIDATION}" \
        --gpu-id "${gpu}" \
        --shard-rank "${rank}" \
        "${resume_args[@]}"
    ) >"${OUT_ROOT}/logs/worker${rank}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
    echo "launched model=${MODEL_KEY} rank=${rank} gpu=${gpu} pid=$!"
  done
  if ((slot + 1 < WORKERS_PER_GPU)); then
    sleep "${WAVE_DELAY_SECONDS}"
  fi
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

complete_args=()
if [[ "${failed}" == "0" ]]; then
  complete_args=(--require-complete)
fi
"${PYTHON_BIN}" -m vlmeval.probes.readout_carriers aggregate \
  --manifest "${MANIFEST}" \
  --input-root "${OUT_ROOT}/predictions" \
  --output-root "${OUT_ROOT}" \
  --datasets "${DATASETS}" \
  "${complete_args[@]}"

if [[ "${failed}" != "0" ]]; then
  echo "At least one worker failed; inspect logs and rerun with RESUME=1" >&2
  exit 2
fi
echo "Completed: ${OUT_ROOT}/accuracy.csv"
