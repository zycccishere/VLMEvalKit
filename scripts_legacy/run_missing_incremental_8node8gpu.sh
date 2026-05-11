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

dispatch_group 0 4 "run_missing_qwen35_all6_4node32gpu.sh" "qwen35-all6"
dispatch_group 4 1 "run_missing_small_text_image_1node8gpu.sh" "small-qwen-text-image"
dispatch_group 5 1 "run_missing_qwen25_32b_text_image_1node8gpu.sh" "qwen25-32b-text-image"
dispatch_group 6 1 "run_missing_qwen25_72b_text_image_1node8gpu.sh" "qwen25-72b-text-image"
dispatch_group 7 1 "run_missing_minicpm_text_image_1node8gpu.sh" "minicpm-text-image"

echo "[FATAL] global node rank ${GLOBAL_NODE_RANK} is outside total node budget 8." >&2
exit 1
