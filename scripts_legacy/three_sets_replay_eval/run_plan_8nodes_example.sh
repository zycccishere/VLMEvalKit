#!/usr/bin/env bash
set -euo pipefail

# Example 8-node allocation plan.
# Adjust NODE_RANK / host launch method according to your scheduler.
#
# Node 0-1:
#   run_qwen_small_3sets_direct_2node16workers.sh
# Node 2:
#   run_minicpm45_3sets_direct_1node8workers.sh
# Node 3:
#   run_qwen25_32b_3sets_direct_tp2_1node4workers.sh
# Node 4-5:
#   run_qwen25_72b_3sets_direct_tp4_2node4workers.sh
# Node 6-7:
#   run_qwen25_72b_3sets_default_tp4_2node4workers.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cat <<EOF
Recommended launch plan:

  nodes 0-1 -> ${SCRIPT_DIR}/run_qwen_small_3sets_direct_2node16workers.sh
  node 2    -> ${SCRIPT_DIR}/run_minicpm45_3sets_direct_1node8workers.sh
  node 3    -> ${SCRIPT_DIR}/run_qwen25_32b_3sets_direct_tp2_1node4workers.sh
  nodes 4-5 -> ${SCRIPT_DIR}/run_qwen25_72b_3sets_direct_tp4_2node4workers.sh
  nodes 6-7 -> ${SCRIPT_DIR}/run_qwen25_72b_3sets_default_tp4_2node4workers.sh

Environment variables you likely want to set before launch:
  EXP_DATE_TAG
  SAVE_ROOT
  LMUData
  HF_ENDPOINT
  MODEL_PATH_QWEN2_2B / MODEL_PATH_QWEN2_7B / MODEL_PATH_QWEN25_3B / MODEL_PATH_QWEN25_7B
  MODEL_PATH_QWEN25_32B / MODEL_PATH_QWEN25_72B / MODEL_PATH_MINICPM45

These wrappers only target:
  VisuLogic LogicVista VisualPuzzles
EOF
