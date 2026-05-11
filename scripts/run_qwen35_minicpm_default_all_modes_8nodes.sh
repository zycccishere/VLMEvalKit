#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NODES="${NODES:-8}"
NODE_RANK="${NODE_RANK:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MODEL_KEYS="${MODEL_KEYS:-qwen35_35b_a3b,qwen35_27b,qwen35_9b,qwen35_4b,minicpm_v_45,minicpm_o_45}"

exec "${SCRIPT_DIR}/run_benchmark.sh" \
  --nodes "${NODES}" \
  --node-rank "${NODE_RANK}" \
  --gpu-ids "${GPU_IDS}" \
  --models "${MODEL_KEYS}" \
  --policies default \
  "$@"
