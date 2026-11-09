#!/usr/bin/env bash
# One MPI JobPack submission = one editable queue owned by role 0.
set -euo pipefail
source /workspace/yicheng/env.sh
source /workspace/yicheng/tools/slime/ray_cluster.sh
: "${CTL_EXP_RUN_TS:?ctl-exp submission identity required}"
: "${RUN_ID:?}" "${QUEUE_BASE:?shared per-instance queue storage}"
: "${QUEUE_CONTROL_ROOT:?pinned control checkout}" "${QUEUE_CONTROL_SHA:?}"
: "${QUEUE_REGISTRY:?prepared bound runs}" "${QUEUE_LAUNCHER:?existing bound launcher}"
: "${QUEUE_PYTHON:?control interpreter}" "${ACTOR_NUM_NODES:?}" "${ACTOR_NUM_GPUS_PER_NODE:?}"
QUEUE_INSTANCE_ROOT="${QUEUE_RESUME_ROOT:-${QUEUE_BASE}/${RUN_ID}/${CTL_EXP_RUN_TS}}"
export QUEUE_INSTANCE_ROOT
mkdir -p "${QUEUE_INSTANCE_ROOT}"
if [[ "${MLP_ROLE_INDEX:?MPI role required}" != 0 ]]; then
  # Keep the replica available to main's SSH workers until this instance drains.
  "${QUEUE_PYTHON}" - "${QUEUE_INSTANCE_ROOT}" "${RUN_ID}/${CTL_EXP_RUN_TS}" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);started=time.time()
while True:
    terminal=p/'terminal.json'
    status=p/'state.json'
    state=json.loads(status.read_text()) if status.exists() else {}
    current=state.get('config',{}).get('instance_id')==sys.argv[2]
    if current and terminal.exists():raise SystemExit(int(json.loads(terminal.read_text()).get('failed',True)))
    if not current:
        if time.time()-started>600:raise SystemExit('main did not initialize this submission')
        time.sleep(5);continue
    heartbeat=state.get('heartbeat',0)
    if (heartbeat and time.time()-heartbeat>120) or (not heartbeat and time.time()-started>600):
        raise SystemExit('main controller unavailable; no worker-side rescheduling')
    time.sleep(5)
PY
  exit $?
fi
on_exit() {
  local rc=$?
  if [[ ! -f "${QUEUE_INSTANCE_ROOT}/terminal.json" ]]; then
    "${QUEUE_PYTHON}" - "${QUEUE_INSTANCE_ROOT}/terminal.json" "${rc}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);t=p.with_suffix('.tmp');t.write_text(json.dumps({'closed':True,'failed':True,'controller_exit':int(sys.argv[2])})+'\n');t.replace(p)
PY
  fi
}
trap on_exit EXIT
[[ "$(git -C "${QUEUE_CONTROL_ROOT}" rev-parse HEAD)" == "${QUEUE_CONTROL_SHA}" ]]
[[ -z "$(git -C "${QUEUE_CONTROL_ROOT}" status --porcelain)" ]]
# Reuse platform host enumeration without invoking Ray or its cleanup helpers.
QUEUE_START_DIR="${QUEUE_INSTANCE_ROOT}/starts/${CTL_EXP_RUN_TS}"
mkdir -p "${QUEUE_START_DIR}"
slime_ray_worker_ips > "${QUEUE_START_DIR}/hosts.txt"
: "${MLP_WORKER_0_HOST:?main host must be supplied by the MPI environment}"
: "${QUEUE_NODE_SSH_PORT:?verify the internal node SSH port for this JobPack}"
export QUEUE_NODE_SSH_PORT
"${QUEUE_PYTHON}" - "${QUEUE_START_DIR}" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);hosts=list(dict.fromkeys(p.joinpath('hosts.txt').read_text().split()))
assert len(hosts)==int(os.environ['ACTOR_NUM_NODES']) and os.environ['MLP_WORKER_0_HOST'] in hosts
nodes={}
for i,host in enumerate(hosts):
    ssh=[] if host==os.environ['MLP_WORKER_0_HOST'] else ['ssh','-p',os.environ['QUEUE_NODE_SSH_PORT'],'-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=accept-new','root@'+host]
    nodes['node'+str(i)]={'ssh':ssh,'gpu_ids':[str(g) for g in range(int(os.environ['ACTOR_NUM_GPUS_PER_NODE']))],
                        'python':os.environ['QUEUE_PYTHON'],'script':os.environ['QUEUE_CONTROL_ROOT']+'/scripts/bound_queue.py'}
p.joinpath('nodes.json').write_text(json.dumps(nodes,indent=2)+'\n')
PY
if [[ -n "${QUEUE_RESUME_ROOT:-}" ]]; then
  : "${QUEUE_PREVIOUS_STATUS:?saved terminal job-pack get JSON required}" "${QUEUE_PREVIOUS_TASK:?}"
  "${QUEUE_PYTHON}" "${QUEUE_CONTROL_ROOT}/scripts/bound_queue.py" --root "${QUEUE_INSTANCE_ROOT}" resume \
    --nodes-file "${QUEUE_START_DIR}/nodes.json" --instance-id "${RUN_ID}/${CTL_EXP_RUN_TS}" \
    --previous-status "${QUEUE_PREVIOUS_STATUS}" --previous-task "${QUEUE_PREVIOUS_TASK}"
else
  "${QUEUE_PYTHON}" "${QUEUE_CONTROL_ROOT}/scripts/bound_queue.py" --root "${QUEUE_INSTANCE_ROOT}" init \
    --registry "${QUEUE_REGISTRY}" --launcher "${QUEUE_LAUNCHER}" \
    --nodes-file "${QUEUE_START_DIR}/nodes.json" --instance-id "${RUN_ID}/${CTL_EXP_RUN_TS}"
fi
# No implicit admission. Use the development-machine client to add exact cells.
if [[ -n "${QUEUE_ESTIMATES_FILE:-}" ]]; then
  "${QUEUE_PYTHON}" "${QUEUE_CONTROL_ROOT}/scripts/bound_queue.py" --root "${QUEUE_INSTANCE_ROOT}" estimates --file "${QUEUE_ESTIMATES_FILE}"
fi
"${QUEUE_PYTHON}" "${QUEUE_CONTROL_ROOT}/scripts/bound_queue.py" --root "${QUEUE_INSTANCE_ROOT}" serve
