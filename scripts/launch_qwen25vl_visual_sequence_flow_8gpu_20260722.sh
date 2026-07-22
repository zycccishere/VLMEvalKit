#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:?manifest json path}
OUTPUT_DIR=${2:?output dir}
MODEL_PATH=${MODEL_PATH:-/user/zyc1781/models/Qwen2.5-VL-32B-Instruct}
PYTHON_BIN=${PYTHON_BIN:-python}
PYDEPS=${PYDEPS:-}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
EXPECTED_GPUS=${EXPECTED_GPUS:-0,1,2,3,4,5,6,7}
ATTN_LAYERS=${ATTN_LAYERS:-last4}
MODE=${MODE:-image_text_image}
POLICY=${POLICY:-identity}
INTERVENTIONS=${INTERVENTIONS:-shift_right_half_vit_token shift_right_one_vit_token shift_right_one_llm_token visual_sequence_roll_right_1}
DUMP_MODE=${DUMP_MODE:-full}
SCALAR_RAW_DUMP_LIMIT=${SCALAR_RAW_DUMP_LIMIT:-0}
SCALAR_QUERY_CHUNK_SIZE=${SCALAR_QUERY_CHUNK_SIZE:-256}
VISUAL_SEQUENCE_RAW_DUMP_LIMIT=${VISUAL_SEQUENCE_RAW_DUMP_LIMIT:-0}
STRICT_LOGICVISTA100=${STRICT_LOGICVISTA100:-1}
EXPECTED_MANIFEST_SHA256=aafd53d34e4e01689f0a16473105a67252d1d7deaf386cdd96d289536d98ca0d
EXPECTED_INTERVENTIONS="shift_right_half_vit_token shift_right_one_vit_token shift_right_one_llm_token visual_sequence_roll_right_1"

if [[ "${STRICT_LOGICVISTA100}" == "1" ]]; then
  actual_sha=$(sha256sum "${MANIFEST}" | awk '{print $1}')
  [[ "${actual_sha}" == "${EXPECTED_MANIFEST_SHA256}" ]] || {
    echo "Unexpected LogicVista100 manifest sha256: ${actual_sha}" >&2
    exit 2
  }
  [[ "${MODEL_PATH}" == */Qwen2.5-VL-32B-Instruct ]] || { echo "Strict run requires Qwen2.5-VL-32B-Instruct" >&2; exit 2; }
  [[ "${GPUS}" == "${EXPECTED_GPUS}" ]] || {
    echo "Strict run requires the explicitly pinned GPU set ${EXPECTED_GPUS}" >&2
    exit 2
  }
  [[ "${ATTN_LAYERS}" == "last4" && "${MODE}" == "image_text_image" && "${POLICY}" == "identity" ]] || {
    echo "Strict run requires last4, IQI, identity" >&2
    exit 2
  }
  [[ "${DUMP_MODE}" == "full" ]] || { echo "Strict run requires full attention dumps" >&2; exit 2; }
  [[ "${INTERVENTIONS}" == "${EXPECTED_INTERVENTIONS}" ]] || {
    echo "Strict run requires the exact canonical intervention list: ${EXPECTED_INTERVENTIONS}" >&2
    exit 2
  }
fi

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/shards"

"${PYTHON_BIN}" - "${MANIFEST}" "${OUTPUT_DIR}/shards" "${GPUS}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
gpus = [item for item in sys.argv[3].split(",") if item]
data = json.loads(manifest_path.read_text(encoding="utf-8"))
if not isinstance(data, list):
    raise SystemExit(f"manifest must be a JSON list: {manifest_path}")
manifest_base = manifest_path.resolve().parent
shards = [[] for _ in gpus]
for idx, item in enumerate(data):
    copied = dict(item)
    if copied.get("image"):
        image_path = Path(str(copied["image"]))
        if not image_path.is_absolute():
            copied["image"] = str((manifest_base / image_path).resolve())
    shards[idx % len(gpus)].append(copied)
for rank, items in enumerate(shards):
    path = shard_dir / f"shard_{rank}.json"
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{rank}\t{gpus[rank]}\t{path}\t{len(items)}")
PY

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
PIDS=()
RANK_DIRS=()
for rank in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$rank]}"
  shard="${OUTPUT_DIR}/shards/shard_${rank}.json"
  rank_dir="${OUTPUT_DIR}/rank_${rank}"
  mkdir -p "${rank_dir}"
  RANK_DIRS+=("${rank_dir}")
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONPATH="${PWD}${PYDEPS:+:${PYDEPS}}:${PYTHONPATH:-}"
    export VLMEVAL_LAZY_INIT=1
    export VLMEVAL_VLM_MINIMAL_IMPORT=1
    export VLMEVAL_API_MINIMAL_IMPORT=1
    export VLMEVAL_USE_QWEN_MINIMAL_CONFIG=1
    "${PYTHON_BIN}" scripts/qwen25vl_shift_flow_probe_20260629.py \
      --model-path "${MODEL_PATH}" \
      --manifest "${shard}" \
      --output-dir "${rank_dir}" \
      --device cuda \
      --mode "${MODE}" \
      --policy "${POLICY}" \
      --text-scope historical_all_non_image_non_special \
      --seed 1234 \
      --attn-layers "${ATTN_LAYERS}" \
      --dump-mode "${DUMP_MODE}" \
      --scalar-raw-dump-limit "${SCALAR_RAW_DUMP_LIMIT}" \
      --scalar-query-chunk-size "${SCALAR_QUERY_CHUNK_SIZE}" \
      --visual-sequence-raw-dump-limit "${VISUAL_SEQUENCE_RAW_DUMP_LIMIT}" \
      --transforms ${INTERVENTIONS}
  ) > "${OUTPUT_DIR}/logs/rank_${rank}.log" 2>&1 &
  pid=$!
  PIDS+=("${pid}")
  echo "${pid}" > "${OUTPUT_DIR}/logs/rank_${rank}.pid"
  echo "launched rank=${rank} gpu=${gpu} pid=${pid} shard=${shard}"
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" != "0" ]]; then
  echo "One or more ranks failed. See ${OUTPUT_DIR}/logs" >&2
  exit "${status}"
fi

for rank in "${!RANK_DIRS[@]}"; do
  rank_dir="${RANK_DIRS[$rank]}"
  expected_cases=$("${PYTHON_BIN}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${OUTPUT_DIR}/shards/shard_${rank}.json")
  strict_full_args=()
  if [[ "${STRICT_LOGICVISTA100}" == "1" ]]; then
    strict_full_args+=(--strict-logicvista100)
  fi
  "${PYTHON_BIN}" scripts/validate_qwen25vl_shift_flow_smoke_20260629.py \
    --output-dir "${rank_dir}" \
    --expected-cases "${expected_cases}" \
    --expected-interventions ${INTERVENTIONS} \
    --strict-contract \
    "${strict_full_args[@]}" \
    > "${OUTPUT_DIR}/logs/rank_${rank}_validation.json"
done

"${PYTHON_BIN}" scripts/analyze_qwen25vl_shift_flow_20260629.py \
  --output-dir "${OUTPUT_DIR}/analysis" \
  --input-dirs "${RANK_DIRS[@]}"

stats_args=()
if [[ "${STRICT_LOGICVISTA100}" == "1" ]]; then
  stats_args+=(--expected-cases 100 --strict-canonical)
fi
"${PYTHON_BIN}" scripts/analyze_qwen25vl_visual_sequence_flow_stats_20260722.py \
  --input-csv "${OUTPUT_DIR}/analysis/case_layer_metrics_with_delta.csv" \
  --output-dir "${OUTPUT_DIR}/analysis/paired_stats" \
  "${stats_args[@]}"

echo "done: ${OUTPUT_DIR}"
