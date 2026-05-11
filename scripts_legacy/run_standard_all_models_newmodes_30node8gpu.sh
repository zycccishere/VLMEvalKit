#!/usr/bin/env bash
set -euo pipefail

detect_node_rank() {
    if [[ -n "${NODE_RANK:-}" ]]; then
        echo "${NODE_RANK}"
        return 0
    fi
    if [[ -n "${SLURM_NODEID:-}" ]]; then
        echo "${SLURM_NODEID}"
        return 0
    fi
    if [[ -n "${RANK:-}" ]]; then
        echo "${RANK}"
        return 0
    fi
    if [[ -n "${OMPI_COMM_WORLD_RANK:-}" ]]; then
        echo "${OMPI_COMM_WORLD_RANK}"
        return 0
    fi
    if [[ -n "${PMI_RANK:-}" ]]; then
        echo "${PMI_RANK}"
        return 0
    fi
    echo "0"
}

GLOBAL_NODE_RANK="$(detect_node_rank)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NODES_SMALL_DIRECT="${NODES_SMALL_DIRECT:-2}"
NODES_SMALL_DEFAULT="${NODES_SMALL_DEFAULT:-2}"
NODES_Q25_32_DIRECT="${NODES_Q25_32_DIRECT:-3}"
NODES_Q25_32_DEFAULT="${NODES_Q25_32_DEFAULT:-3}"
NODES_Q25_72_DIRECT="${NODES_Q25_72_DIRECT:-6}"
NODES_Q25_72_DEFAULT="${NODES_Q25_72_DEFAULT:-6}"
NODES_MINICPM_V_DIRECT="${NODES_MINICPM_V_DIRECT:-1}"
NODES_MINICPM_V_DEFAULT="${NODES_MINICPM_V_DEFAULT:-1}"
NODES_MINICPM_O_DIRECT="${NODES_MINICPM_O_DIRECT:-1}"
NODES_MINICPM_O_DEFAULT="${NODES_MINICPM_O_DEFAULT:-1}"
NODES_Q35_4B="${NODES_Q35_4B:-1}"
NODES_Q35_9B="${NODES_Q35_9B:-1}"
NODES_Q35_27B="${NODES_Q35_27B:-2}"

TOTAL_NODES=$(( \
    NODES_SMALL_DIRECT + \
    NODES_SMALL_DEFAULT + \
    NODES_Q25_32_DIRECT + \
    NODES_Q25_32_DEFAULT + \
    NODES_Q25_72_DIRECT + \
    NODES_Q25_72_DEFAULT + \
    NODES_MINICPM_V_DIRECT + \
    NODES_MINICPM_V_DEFAULT + \
    NODES_MINICPM_O_DIRECT + \
    NODES_MINICPM_O_DEFAULT + \
    NODES_Q35_4B + \
    NODES_Q35_9B + \
    NODES_Q35_27B \
))

if (( TOTAL_NODES != 30 )); then
    echo "[FATAL] This launcher expects 30 nodes in total. Got ${TOTAL_NODES}." >&2
    exit 1
fi

dispatch_group() {
    local start="$1"
    local size="$2"
    local target_script="$3"
    local label="$4"
    if (( GLOBAL_NODE_RANK >= start && GLOBAL_NODE_RANK < start + size )); then
        export NODE_RANK="$(( GLOBAL_NODE_RANK - start ))"
        export NUM_NODES="${size}"
        echo "[DISPATCH] global_node_rank=${GLOBAL_NODE_RANK} -> ${label} node_rank=${NODE_RANK}/${NUM_NODES}"
        exec bash "${SCRIPT_DIR}/${target_script}"
    fi
}

cursor=0
dispatch_group "$cursor" "$NODES_SMALL_DIRECT" "run_standard_qwen2_qwen25_small_newsets_last1_direct_2node16workers.sh" "qwen2/qwen25-small-direct"
cursor=$(( cursor + NODES_SMALL_DIRECT ))
dispatch_group "$cursor" "$NODES_SMALL_DEFAULT" "run_standard_qwen2_qwen25_small_newsets_last1_default_2node16workers.sh" "qwen2/qwen25-small-default"
cursor=$(( cursor + NODES_SMALL_DEFAULT ))
dispatch_group "$cursor" "$NODES_Q25_32_DIRECT" "run_standard_qwen25_32b_newsets_last1_tp2_3node_dataset_direct.sh" "qwen25-32b-direct"
cursor=$(( cursor + NODES_Q25_32_DIRECT ))
dispatch_group "$cursor" "$NODES_Q25_32_DEFAULT" "run_standard_qwen25_32b_newsets_last1_tp2_3node_dataset_default.sh" "qwen25-32b-default"
cursor=$(( cursor + NODES_Q25_32_DEFAULT ))
dispatch_group "$cursor" "$NODES_Q25_72_DIRECT" "run_standard_qwen25_72b_newsets_last1_tp4_6node_dataset_direct.sh" "qwen25-72b-direct"
cursor=$(( cursor + NODES_Q25_72_DIRECT ))
dispatch_group "$cursor" "$NODES_Q25_72_DEFAULT" "run_standard_qwen25_72b_newsets_last1_tp4_6node_dataset_default.sh" "qwen25-72b-default"
cursor=$(( cursor + NODES_Q25_72_DEFAULT ))
dispatch_group "$cursor" "$NODES_MINICPM_V_DIRECT" "run_standard_minicpm45_newsets_last1_direct_1node8gpu.sh" "minicpm-v-direct"
cursor=$(( cursor + NODES_MINICPM_V_DIRECT ))
dispatch_group "$cursor" "$NODES_MINICPM_V_DEFAULT" "run_standard_minicpm45_newsets_last1_default_1node8gpu.sh" "minicpm-v-default"
cursor=$(( cursor + NODES_MINICPM_V_DEFAULT ))
dispatch_group "$cursor" "$NODES_MINICPM_O_DIRECT" "run_standard_minicpmo45_newsets_last1_direct_1node8gpu.sh" "minicpm-o-direct"
cursor=$(( cursor + NODES_MINICPM_O_DIRECT ))
dispatch_group "$cursor" "$NODES_MINICPM_O_DEFAULT" "run_standard_minicpmo45_newsets_last1_default_1node8gpu.sh" "minicpm-o-default"
cursor=$(( cursor + NODES_MINICPM_O_DEFAULT ))
dispatch_group "$cursor" "$NODES_Q35_4B" "run_standard_qwen35_4b_newsets_last1.sh" "qwen35-4b-both-policies"
cursor=$(( cursor + NODES_Q35_4B ))
dispatch_group "$cursor" "$NODES_Q35_9B" "run_standard_qwen35_9b_newsets_last1.sh" "qwen35-9b-both-policies"
cursor=$(( cursor + NODES_Q35_9B ))
dispatch_group "$cursor" "$NODES_Q35_27B" "run_standard_qwen35_27b_newsets_last1.sh" "qwen35-27b-both-policies"

echo "[FATAL] global node rank ${GLOBAL_NODE_RANK} is outside total node budget ${TOTAL_NODES}." >&2
exit 1
