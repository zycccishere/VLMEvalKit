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

export REPLAY_MODE_LIST="${REPLAY_MODE_LIST:-text_image}"

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

dispatch_group 0 1 "run_standard_qwen35_4b_newsets_last1.sh" "qwen35-4b-text-image"
dispatch_group 1 1 "run_standard_qwen35_9b_newsets_last1.sh" "qwen35-9b-text-image"
dispatch_group 2 2 "run_standard_qwen35_27b_newsets_last1.sh" "qwen35-27b-text-image"

echo "[FATAL] global node rank ${GLOBAL_NODE_RANK} is outside total node budget 4." >&2
exit 1
