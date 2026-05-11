#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 && $# -ne 2 ]]; then
    cat >&2 <<'EOF'
Usage:
  bash scripts/ssh_launch_qwen25vl_minicpm45_all4_reasoning_perception4_2nodes_20260422.sh [host0 host1]

Environment overrides:
  SSH_HOSTS="host0 host1"
  REMOTE_REPO=/remote/path/to/vlmevalkit
  CONTROL_PYTHON=python
  GPU_IDS=0,1,2,3,4,5,6,7
  TMUX_SESSION_PREFIX=qwen25_minicpm_all4_2node_20260422
  EXTRA_ARGS="--plan-only"
EOF
    exit 2
fi

: "${REMOTE_REPO:?Set REMOTE_REPO=/remote/path/to/vlmevalkit before launching.}"
CONTROL_PYTHON="${CONTROL_PYTHON:-python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TMUX_SESSION_PREFIX="${TMUX_SESSION_PREFIX:-qwen25_minicpm_all4_2node_20260422}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ $# -eq 2 ]]; then
    HOSTS=("$1" "$2")
elif [[ -n "${SSH_HOSTS:-}" ]]; then
    read -r -a HOSTS <<< "${SSH_HOSTS}"
    if [[ "${#HOSTS[@]}" -ne 2 ]]; then
        echo "[FATAL] SSH_HOSTS must contain exactly two hosts" >&2
        exit 2
    fi
else
    echo "[FATAL] provide host0 host1 arguments or SSH_HOSTS=\"host0 host1\"" >&2
    exit 2
fi

for NODE_RANK in 0 1; do
    HOST="${HOSTS[$NODE_RANK]}"
    SESSION="${TMUX_SESSION_PREFIX}_n${NODE_RANK}"
    REMOTE_CMD=$(
        cat <<EOF
set -euo pipefail
cd "${REMOTE_REPO}"
mkdir -p tmp/qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422
"${CONTROL_PYTHON}" scripts/prepare_qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422.py --nodes 2 --gpus-per-node 8
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "CONTROL_PYTHON='${CONTROL_PYTHON}' bash scripts/run_qwen25vl_minicpm45_all4_reasoning_perception4_2nodes_20260422.sh ${NODE_RANK} ${GPU_IDS} ${EXTRA_ARGS}"
tmux display-message -t "${SESSION}" -p '#S #{pane_pid}'
EOF
    )
    echo "[SSH][LAUNCH] host=${HOST} node_rank=${NODE_RANK} session=${SESSION}"
    ssh "${HOST}" "${REMOTE_CMD}"
done
