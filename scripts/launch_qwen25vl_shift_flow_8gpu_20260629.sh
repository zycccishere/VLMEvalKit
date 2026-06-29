#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:?manifest json path}
OUTPUT_DIR=${2:?output dir}
MODEL_PATH=${MODEL_PATH:-/user/zyc1781/models/Qwen2.5-VL-32B-Instruct}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
ATTN_LAYERS=${ATTN_LAYERS:-last4}
MODE=${MODE:-image_text_image}
POLICY=${POLICY:-identity}
TRANSFORMS=${TRANSFORMS:-shift_right_half_vit_token shift_right_one_vit_token shift_right_one_llm_token}

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/shards"

python - "${MANIFEST}" "${OUTPUT_DIR}/shards" "${GPUS}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
gpus = [item for item in sys.argv[3].split(",") if item]
data = json.loads(manifest_path.read_text(encoding="utf-8"))
if not isinstance(data, list):
    raise SystemExit(f"manifest must be a JSON list: {manifest_path}")
shards = [[] for _ in gpus]
for idx, item in enumerate(data):
    shards[idx % len(gpus)].append(item)
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
    export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
    python scripts/qwen25vl_shift_flow_probe_20260629.py \
      --model-path "${MODEL_PATH}" \
      --manifest "${shard}" \
      --output-dir "${rank_dir}" \
      --device cuda \
      --mode "${MODE}" \
      --policy "${POLICY}" \
      --attn-layers "${ATTN_LAYERS}" \
      --transforms ${TRANSFORMS}
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

python scripts/analyze_qwen25vl_shift_flow_20260629.py \
  --output-dir "${OUTPUT_DIR}/analysis" \
  --input-dirs "${RANK_DIRS[@]}"

echo "done: ${OUTPUT_DIR}"
