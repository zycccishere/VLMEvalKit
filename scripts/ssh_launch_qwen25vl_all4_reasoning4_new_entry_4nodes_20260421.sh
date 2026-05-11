#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    cat >&2 <<'EOF'
Usage:
  bash scripts/ssh_launch_qwen25vl_all4_reasoning4_new_entry_4nodes_20260421.sh <host0> <host1> <host2> <host3>

Environment overrides:
  REMOTE_REPO=/remote/path/to/vlmevalkit
  GPU_IDS=0,1,2,3,4,5,6,7
  TMUX_SESSION_PREFIX=qwen25vl_reasoning4_20260421
  EXTRA_ARGS="--plan-only"
EOF
    exit 2
fi

: "${REMOTE_REPO:?Set REMOTE_REPO=/remote/path/to/vlmevalkit before launching.}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TMUX_SESSION_PREFIX="${TMUX_SESSION_PREFIX:-qwen25vl_reasoning4_20260421}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
HOSTS=("$@")

for NODE_RANK in 0 1 2 3; do
    HOST="${HOSTS[$NODE_RANK]}"
    SESSION="${TMUX_SESSION_PREFIX}_n${NODE_RANK}"
    REMOTE_CMD=$(
        cat <<EOF
set -euo pipefail
cd "${REMOTE_REPO}"
mkdir -p tmp/qwen25vl_all4_reasoning4_new_entry_20260421
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "bash scripts/run_qwen25vl_all4_reasoning4_new_entry_4nodes_20260421.sh ${NODE_RANK} ${GPU_IDS} ${EXTRA_ARGS}"
tmux display-message -t "${SESSION}" -p '#S #{pane_pid}'
EOF
    )
    echo "[SSH][LAUNCH] host=${HOST} node_rank=${NODE_RANK} session=${SESSION}"
    ssh "${HOST}" "${REMOTE_CMD}"
done
