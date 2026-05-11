#!/usr/bin/env bash
set -euo pipefail

# Run all re-eval scripts for runs/standard/20260304 sequentially.
# Ensure OPENAI_API_KEY is set before running (needed for 32b/72b replay8).
#
# Usage: cd vlmevalkit && bash scripts/run_reeval_all_20260304.sh
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_API_BASE_JUDGE="${OPENAI_API_BASE_JUDGE:-${OPENAI_API_BASE:-https://api.openai.com/v1}}"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit 1

echo "=== [REEVAL] Starting all re-evals for 20260304 ==="

echo "=== [1/6] qwen25_32b_replay8 ==="
bash "${SCRIPT_DIR}/run_reeval_qwen25_32b_replay8_1node_8gpu.sh"

echo "=== [2/6] qwen25_32b_newsets ==="
bash "${SCRIPT_DIR}/run_reeval_qwen25_32b_newsets_last1_1node_8gpu.sh"

echo "=== [3/6] qwen25_72b_replay8 ==="
bash "${SCRIPT_DIR}/run_reeval_qwen25_72b_replay8_1node_8gpu.sh"

echo "=== [4/6] qwen25_72b_newsets ==="
bash "${SCRIPT_DIR}/run_reeval_qwen25_72b_newsets_last1_1node_8gpu.sh"

echo "=== [5/6] qwen2_qwen25_newsets rank0 ==="
bash "${SCRIPT_DIR}/run_reeval_qwen2_qwen25_newsets_last1_1node_rank0.sh"

echo "=== [6/6] qwen2_qwen25_newsets rank1 (if 2-node: run rank0 and rank1 in parallel on each node) ==="
bash "${SCRIPT_DIR}/run_reeval_qwen2_qwen25_newsets_last1_1node_rank1.sh"

echo "=== [REEVAL] All re-evals finished ==="
