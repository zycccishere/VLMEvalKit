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
NODES_9B="${NODES_9B:-1}"
NODES_27B="${NODES_27B:-2}"
NODES_35B_A3B="${NODES_35B_A3B:-1}"
TOTAL_NODES=$(( NODES_9B + NODES_27B + NODES_35B_A3B ))

if (( TOTAL_NODES != 4 )); then
    echo "[FATAL] This launcher expects 4 nodes in total. Got ${TOTAL_NODES} from NODES_9B=${NODES_9B}, NODES_27B=${NODES_27B}, NODES_35B_A3B=${NODES_35B_A3B}." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if (( GLOBAL_NODE_RANK < NODES_9B )); then
    export NODE_RANK="${GLOBAL_NODE_RANK}"
    export NUM_NODES="${NODES_9B}"
    echo "[DISPATCH] global_node_rank=${GLOBAL_NODE_RANK} -> Qwen3.5-9B group node_rank=${NODE_RANK}/${NUM_NODES}"
    exec bash "${SCRIPT_DIR}/run_standard_qwen35_9b_newsets_last1.sh"
fi

if (( GLOBAL_NODE_RANK < NODES_9B + NODES_27B )); then
    export NODE_RANK="$(( GLOBAL_NODE_RANK - NODES_9B ))"
    export NUM_NODES="${NODES_27B}"
    echo "[DISPATCH] global_node_rank=${GLOBAL_NODE_RANK} -> Qwen3.5-27B group node_rank=${NODE_RANK}/${NUM_NODES}"
    exec bash "${SCRIPT_DIR}/run_standard_qwen35_27b_newsets_last1.sh"
fi

if (( GLOBAL_NODE_RANK < TOTAL_NODES )); then
    export NODE_RANK="$(( GLOBAL_NODE_RANK - NODES_9B - NODES_27B ))"
    export NUM_NODES="${NODES_35B_A3B}"
    echo "[DISPATCH] global_node_rank=${GLOBAL_NODE_RANK} -> Qwen3.5-35B-A3B group node_rank=${NODE_RANK}/${NUM_NODES}"
    exec bash "${SCRIPT_DIR}/run_standard_qwen35_35b_a3b_newsets_last1.sh"
fi

echo "[FATAL] global node rank ${GLOBAL_NODE_RANK} is outside total node budget ${TOTAL_NODES}." >&2
exit 1
