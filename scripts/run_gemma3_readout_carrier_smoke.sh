#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/user/zyc1781/vlmevalkit-release-readout-gemma3}"
PYTHON_BIN="${PYTHON_BIN:-/user/zhangyicheng/miniconda3/envs/vlmeval_gemma4_vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/user/zhangyicheng/models/gemma-3-12b-it}"
LMU_DATA="${LMUData:-/user/zyc1781/LMUData}"
MATRIX_CONFIG="${MATRIX_CONFIG:-${ROOT}/configs/matrix.yaml}"
OUT_ROOT="${OUT_ROOT:-/user/zyc1781/outputs/readout_random_carriers/readout_carrier_gemma3_smoke_20260805}"
MANIFEST="${OUT_ROOT}/manifests/gemma3_12b.json"
GPU_ID="${GPU_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-8}"
MANIFEST_DATASETS="${MANIFEST_DATASETS:-DynaMath,WeMath,MMBench_DEV_EN_V11,MMStar,AI2D_TEST}"
SMOKE_DATASET="${SMOKE_DATASET:-DynaMath}"

ALL_SINGLE_CHOICE_MANIFEST="${ALL_SINGLE_CHOICE_MANIFEST:-/user/zyc1781/outputs/readout_v2/qwen25vl3b_readout_v2_all_single_choice_20260729/manifest_all_single_choice.json}"
FIXED_CHOICE_MANIFEST="${FIXED_CHOICE_MANIFEST:-/user/zyc1781/outputs/readout_v2/qwen25vl3b_readout_v2_fixed_choice_20260728_v2/manifest.json}"
MMSTAR_AI2D_MANIFEST="${MMSTAR_AI2D_MANIFEST:-/user/zyc1781/outputs/readout_v2/qwen25vl3b_mmstar_ai2d_new_baseline_full_v1_20260729/manifest.json}"

export PYTHONPATH="${ROOT}"
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
rm -rf "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}/manifests" "${OUT_ROOT}/gemma3_12b/raw" "${OUT_ROOT}/runtime"

"${PYTHON_BIN}" -m vlmeval.probes.readout_carriers manifest \
  --repo-root "${ROOT}" \
  --output "${MANIFEST}" \
  --all-single-choice-manifest "${ALL_SINGLE_CHOICE_MANIFEST}" \
  --fixed-choice-manifest "${FIXED_CHOICE_MANIFEST}" \
  --mmstar-ai2d-manifest "${MMSTAR_AI2D_MANIFEST}" \
  --lmu-data "${LMU_DATA}" \
  --matrix-config "${MATRIX_CONFIG}" \
  --datasets "${MANIFEST_DATASETS}" \
  --model-key gemma3_12b \
  --model-path "${MODEL_PATH}" \
  --num-shards "${NUM_SHARDS}" \
  --samples-per-dataset 128 \
  --selection-seed 20260804

"${PYTHON_BIN}" -m vlmeval.probes.readout_carriers verify-run-contract \
  --repo-root "${ROOT}" \
  --manifest "${MANIFEST}" \
  --model-path "${MODEL_PATH}" \
  --lmu-data "${LMU_DATA}" \
  --matrix-config "${MATRIX_CONFIG}" \
  --output "${OUT_ROOT}/manifests/gemma3_12b.run_contract.json"

"${PYTHON_BIN}" -m vlmeval.probes.readout_carriers run \
  --repo-root "${ROOT}" \
  --manifest "${MANIFEST}" \
  --output-jsonl "${OUT_ROOT}/gemma3_12b/smoke.jsonl" \
  --runtime-root "${OUT_ROOT}/runtime/gemma3_12b" \
  --datasets "${SMOKE_DATASET}" \
  --model-key gemma3_12b \
  --model-path "${MODEL_PATH}" \
  --lmu-data "${LMU_DATA}" \
  --matrix-config "${MATRIX_CONFIG}" \
  --run-contract-attestation "${OUT_ROOT}/manifests/gemma3_12b.run_contract.json" \
  --gpu-id "${GPU_ID}" \
  --one-per-dataset \
  --dump-raw-root "${OUT_ROOT}/gemma3_12b/raw" \
  --diagnostics \
  --diagnostics-limit 1

"${PYTHON_BIN}" -m vlmeval.probes.readout_carriers validate-smoke \
  --manifest "${MANIFEST}" \
  --raw-root "${OUT_ROOT}/gemma3_12b/raw" \
  --output "${OUT_ROOT}/gemma3_12b/validation.json" \
  --expected-artifacts 1 \
  --require-diagnostics 1

echo "Gemma3 readout carrier smoke passed: ${OUT_ROOT}"
