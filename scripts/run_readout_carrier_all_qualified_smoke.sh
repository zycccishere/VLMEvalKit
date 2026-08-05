#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/user/zyc1781/vlmevalkit-release-readout-all-qualified}"
MODEL_KEY="${MODEL_KEY:?Set MODEL_KEY to qwen25vl_32b, minicpm_v_45, or gemma3_12b}"
LMU_DATA="${LMUData:-/user/zyc1781/LMUData}"
MATRIX_CONFIG="${MATRIX_CONFIG:-${ROOT}/configs/matrix.yaml}"
GPU_ID="${GPU_ID:-0}"
OUT_ROOT="${OUT_ROOT:-/user/zyc1781/outputs/readout_random_carriers/readout_carriers_all_qualified_smoke_20260805/${MODEL_KEY}}"
MANIFEST="${OUT_ROOT}/manifest.json"
RUN_CONTRACT="${OUT_ROOT}/run_contract_attestation.json"
ALL_DATASETS="DynaMath,WeMath,MMBench_DEV_EN_V11,MMStar,AI2D_TEST"

ALL_SINGLE_CHOICE_MANIFEST="${ALL_SINGLE_CHOICE_MANIFEST:-/user/zyc1781/outputs/readout_v2/qwen25vl3b_readout_v2_all_single_choice_20260729/manifest_all_single_choice.json}"
FIXED_CHOICE_MANIFEST="${FIXED_CHOICE_MANIFEST:-/user/zyc1781/outputs/readout_v2/qwen25vl3b_readout_v2_fixed_choice_20260728_v2/manifest.json}"
MMSTAR_AI2D_MANIFEST="${MMSTAR_AI2D_MANIFEST:-/user/zyc1781/outputs/readout_v2/qwen25vl3b_mmstar_ai2d_new_baseline_full_v1_20260729/manifest.json}"
MINICPM_PYDEPS="${MINICPM_PYDEPS:-/user/zyc1781/.venvs/minicpmo-token-roll-pydeps}"

case "${MODEL_KEY}" in
  qwen25vl_32b)
    PYTHON_BIN="${PYTHON_BIN:-/user/wanzihao/miniconda3/envs/vlmevalkit/bin/python}"
    MODEL_PATH="${MODEL_PATH:-/user/zyc1781/models/Qwen2.5-VL-32B-Instruct}"
    NUM_SHARDS=8
    WORKER_PYTHONPATH="${ROOT}"
    ;;
  minicpm_v_45)
    PYTHON_BIN="${PYTHON_BIN:-/user/zhangyicheng/miniconda3/envs/duplex_mm_eval310/bin/python}"
    MODEL_PATH="${MODEL_PATH:-/user/zyc1781/models/MiniCPM-V-4_5}"
    NUM_SHARDS=32
    WORKER_PYTHONPATH="${ROOT}:${MINICPM_PYDEPS}"
    ;;
  gemma3_12b)
    PYTHON_BIN="${PYTHON_BIN:-/user/zhangyicheng/miniconda3/envs/duplex_mm_eval310/bin/python}"
    MODEL_PATH="${MODEL_PATH:-/user/zhangyicheng/models/gemma-3-12b-it}"
    NUM_SHARDS=8
    WORKER_PYTHONPATH="${ROOT}"
    ;;
  *)
    echo "Unsupported MODEL_KEY: ${MODEL_KEY}" >&2
    exit 2
    ;;
esac

export PYTHONPATH="${WORKER_PYTHONPATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export LMUData="${LMU_DATA}"
export REPLAY_TRACE_LEVEL=off
export VLMEVAL_API_MINIMAL_IMPORT=1
export VLMEVAL_VLM_MINIMAL_IMPORT=1
export VLMEVAL_LAZY_INIT=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "${ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to smoke a dirty repository: ${ROOT}" >&2
  exit 2
fi
for required in \
  "${PYTHON_BIN}" "${MODEL_PATH}" "${MATRIX_CONFIG}" \
  "${ALL_SINGLE_CHOICE_MANIFEST}" "${FIXED_CHOICE_MANIFEST}" \
  "${MMSTAR_AI2D_MANIFEST}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path is missing: ${required}" >&2
    exit 2
  fi
done
if [[ "${MODEL_KEY}" == "minicpm_v_45" && ! -d "${MINICPM_PYDEPS}" ]]; then
  echo "Required MiniCPM dependency overlay is missing: ${MINICPM_PYDEPS}" >&2
  exit 2
fi

rm -rf "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}/raw" "${OUT_ROOT}/runtime"

"${PYTHON_BIN}" -m vlmeval.probes.readout_carriers manifest \
  --repo-root "${ROOT}" \
  --output "${MANIFEST}" \
  --all-single-choice-manifest "${ALL_SINGLE_CHOICE_MANIFEST}" \
  --fixed-choice-manifest "${FIXED_CHOICE_MANIFEST}" \
  --mmstar-ai2d-manifest "${MMSTAR_AI2D_MANIFEST}" \
  --lmu-data "${LMU_DATA}" \
  --matrix-config "${MATRIX_CONFIG}" \
  --datasets "${ALL_DATASETS}" \
  --model-key "${MODEL_KEY}" \
  --model-path "${MODEL_PATH}" \
  --num-shards "${NUM_SHARDS}"

"${PYTHON_BIN}" -m vlmeval.probes.readout_carriers verify-run-contract \
  --repo-root "${ROOT}" \
  --manifest "${MANIFEST}" \
  --model-path "${MODEL_PATH}" \
  --lmu-data "${LMU_DATA}" \
  --matrix-config "${MATRIX_CONFIG}" \
  --output "${RUN_CONTRACT}"

"${PYTHON_BIN}" -m vlmeval.probes.readout_carriers run \
  --repo-root "${ROOT}" \
  --manifest "${MANIFEST}" \
  --output-jsonl "${OUT_ROOT}/smoke.jsonl" \
  --runtime-root "${OUT_ROOT}/runtime" \
  --datasets "${ALL_DATASETS}" \
  --model-key "${MODEL_KEY}" \
  --model-path "${MODEL_PATH}" \
  --lmu-data "${LMU_DATA}" \
  --matrix-config "${MATRIX_CONFIG}" \
  --run-contract-attestation "${RUN_CONTRACT}" \
  --gpu-id "${GPU_ID}" \
  --one-per-dataset \
  --dump-raw-root "${OUT_ROOT}/raw" \
  --diagnostics \
  --diagnostics-limit 1

"${PYTHON_BIN}" -m vlmeval.probes.readout_carriers validate-smoke \
  --manifest "${MANIFEST}" \
  --raw-root "${OUT_ROOT}/raw" \
  --output "${OUT_ROOT}/validation.json" \
  --expected-artifacts 5 \
  --require-diagnostics 1

echo "All-qualified smoke passed: ${OUT_ROOT}"
