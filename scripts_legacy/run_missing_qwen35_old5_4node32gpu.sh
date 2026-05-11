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

export REPLAY_MODE_LIST="${REPLAY_MODE_LIST:-image_text,image_text_text,image_text_image,image_text_image_text,image_image_text}"

dispatch_group() {
    local start="$1"
    local size="$2"
    local target_script="$3"
    local label="$4"
    local group_tag="$5"
    if (( GLOBAL_NODE_RANK >= start && GLOBAL_NODE_RANK < start + size )); then
        export NODE_RANK="$(( GLOBAL_NODE_RANK - start ))"
        export NUM_NODES="${size}"
        export EXP_GROUP_TAG="${EXP_GROUP_TAG:-$group_tag}"
        echo "[DISPATCH] global_node_rank=${GLOBAL_NODE_RANK} -> ${label} node_rank=${NODE_RANK}/${NUM_NODES}"
        exec bash "${SCRIPT_DIR}/${target_script}"
    fi
}

dispatch_group 0 1 "run_standard_qwen35_4b_newsets_last1.sh" "qwen35-4b-old5" "missing_qwen35_4b_old5"
dispatch_group 1 1 "run_standard_qwen35_9b_newsets_last1.sh" "qwen35-9b-old5" "missing_qwen35_9b_old5"
dispatch_group 2 2 "run_standard_qwen35_27b_newsets_last1.sh" "qwen35-27b-old5" "missing_qwen35_27b_old5"

echo "[FATAL] global node rank ${GLOBAL_NODE_RANK} is outside total node budget 4." >&2
exit 1
