# Editable host queue

`scripts/bound_queue.py` adds mutable admission to the existing bound runner.
Fixed native pools cannot lend GPUs across separate matrices. This queue stores
all admitted cells together and invokes the existing bound launcher with one
cell CSV and the currently free GPU IDs. It keeps each registered matrix,
model entry, execution SHA, prediction path, and table binding unchanged.

The single main controller observes each configured node (local or SSH), then
waits two seconds before the next scan. Its
exclusive-card criterion is the same local `memory.used <= 1024 MiB` criterion
as the existing launch scripts. Pending entries share the whole host; declared
`gpus_per_job` determines whether a task fits. Within each explicit priority,
the shortest estimated remaining inference time goes first, both on initial
dispatch and subsequent refill. The existing native selector breaks ties.
Unknown estimates follow known estimates. Background judging is excluded from
the GPU duration; it never delays refill after inference releases its cards.

`tools/queue.sh estimates --file /absolute/runtime-estimates.json` connects the
existing viewer export. Estimates are refreshed at each dispatch scan, and
must match the cell ID, model entry, GPU width and registered model config path.
Their source and chosen duration are recorded on each attempt. This is a
forecast, not a task timeout or a promise of completion order.
Each cell's TP allocation stays within one node. Tasks that fit can pass a wider task waiting for enough cards. This maximizes
available use, but is not a reservation/fairness or global optimality guarantee.

## Commands

Initialize once with a registry and the existing credential-loading launcher:

```bash
python scripts/bound_queue.py --root /absolute/queue init \
  --registry /absolute/runs/bound --launcher /absolute/scripts/run_bound.sh
python scripts/bound_queue.py --root /absolute/queue serve
```

The prompt-replay topic provides `tools/queue.sh` with these paths and the
existing control venv configured. The service stays up when its queue is empty.
Only explicitly admitted cells can run. Registering a matrix or mentioning a
model does not admit its cells.

```bash
tools/queue.sh list
tools/queue.sh add --run-id final6-old-spaces-20261109 \
  --cell '最终协议|Qwen2.5-VL-32B|spaces|MathVision'
tools/queue.sh priority --id 31 --value 10
tools/queue.sh remove --id 31
tools/queue.sh retry --id 31
```

Use exact cell IDs from the saved registration (the illustrative model label
above must match that registration). Multiple `--cell` arguments are allowed.
`--all` explicitly admits an entire registered run. `add --observe` imports
existing jobs without launching missing work; live legacy controllers are also
detected automatically, including their still-pending manifests.

Removal cancels a pending item and retains its audit record. Active inference
or scoring is not terminated by `remove`; stopping it is a separate explicit
operation. Failed jobs remain failed until `retry`. Retry calls the existing
resume path: complete predictions are reused, incomplete predictions follow
native resume semantics. An already complete score is never silently rerun;
a deliberate fresh experiment needs a separately registered result identity.

## Process and artifact ownership

The JSON ledger is protected by `flock` and atomic rename. One daemon owns the
host queue. Claims reserve GPU IDs before children load models. Admission and
launch recheck the frozen configuration hashes. Repeated admission is
idempotent by cell ID; altered configuration under the same ID is rejected.
Each execution child holds a per-cell lock inherited by the native controller.
The queue also reads native phase PIDs and existing evaluator locks. A crashed
daemon can observe surviving children by PID/start-time/boot identity. Ambiguous
launches become `needs_review`, never an automatic duplicate submission.

Each cell uses `run_bound.sh -> run_benchmark -> run.py`. The existing native
background evaluator remains responsible for gpt-4o-mini, temperature zero,
per-cell concurrency 512, official scoring, and evaluator locking. GPU capacity
is released after the native inference launcher returns, while the cell remains
`scoring` until its bound score manifest completes. Completion here is an
artifact lifecycle state, not scientific score acceptance.

All future local jobs should enter this one queue. The daemon observes external
memory occupancy but cannot reserve against an unrelated process that starts
after its GPU query. The existing opt-in native readiness guard runs again at
model startup. External controllers already holding tasks are observed, not
stopped, edited in memory, or duplicated. Existing native APIs do not support
stealing their queued tasks safely.

`state.json` includes jobs, attempts, immutable admission specs, audit events,
daemon identity, node identities, heartbeat, free GPUs, and query failures. Per-attempt native
logs and the exact one-cell CSV live in `attempts/<id>-<attempt>/`. Restarting
the queue service does not kill its already running inference or scorers.

## Validation

`scripts/test_bound_queue.py` drives the real native CLI and detached evaluator
with tiny CPU fixtures and simulated GPU availability. It checks cross-run card
reuse during scoring, dynamic admission/removal/priority, duplicate admission,
restart observation, mixed TP fitting, failed-score retry preserving prediction
bytes, and configuration drift rejection. `test_bound_runner.py` covers the
underlying native process and 512-gate contract. These are controller behavior
tests, not a new real-model, Qwen, DockerProxy, E2B, or GPU performance run.

## One controller per platform submission

`scripts/run_queue_platform.sh` is the MPI entrypoint. It reuses
`/workspace/yicheng/tools/slime/ray_cluster.sh:slime_ray_worker_ips` for the
platform hostfile / `MLP_WORKER_N_HOST` list, without starting Ray or invoking
any cleanup. Only role 0 runs `serve`; other roles keep their replicas available
until the instance terminates. The queue root is
`QUEUE_BASE/RUN_ID/CTL_EXP_RUN_TS`. A new `ctl-exp submit` therefore creates a
separate instance, rather than sharing the development-machine queue.

Configure these existing JobPack fields/variables before submission:

- `spec.framework=2`, explicit replicas and GPU count, pinned image, verified
  mounts and SSH keys; set the entrypoint to this script in the pinned checkout.
- `QUEUE_CONTROL_ROOT`, `QUEUE_CONTROL_SHA`, `QUEUE_PYTHON`: pinned control code
  and interpreter available on all nodes.
- `QUEUE_REGISTRY`, `QUEUE_LAUNCHER`, `QUEUE_BASE`: prepared registrations,
  existing bound launcher, and writable shared queue storage.
- `ACTOR_NUM_NODES`, `ACTOR_NUM_GPUS_PER_NODE`, `QUEUE_NODE_SSH_PORT`: explicit
  topology and the verified internal SSH port. External container SSH normally
  uses port 2222; do not assume the internal hostfile uses that same endpoint.
- The normal `LMUData` and native model-profile environment. `ctl-exp` supplies
  the topic, run and submission timestamp; MPI supplies role and main address.

Every configured node must see the same queue/registration/result paths.
Startup checks the main-written random token, cross-node `flock`, control
source hash, instance ID, SSH reachability and GPU inventory before dispatch.
This is an explicit shared-volume deployment contract, consistent with the
existing JobPack volume layout; it is not an assumption about arbitrary hosts.
Different mounts must be reconciled before launching, not silently copied.
Node hostname/boot identity is pinned for the instance. An unreachable or
changed node keeps its claims; another node never silently reruns its cells.

Main probes nodes through SSH and launches a detached queue execution child
there. Its per-attempt marker fences ambiguous SSH startup: loss of the client
connection does not kill a launched inference, and missing acknowledgement
requires review rather than blind replay. Workers only execute their assigned
cell and report through the locked shared ledger; they do not run schedulers.

From the development machine, resolve the **current submission's** main using
`ctl-exp connect prompt-replay rebuttal-runs <run> --task <task>`, then route
ordinary queue commands to its exact instance:

```bash
export QUEUE_MAIN=root@<verified-main-IP>
export QUEUE_SSH_PORT=2222
export QUEUE_ROOT=<QUEUE_BASE>/<RUN_ID>/<CTL_EXP_RUN_TS>
tools/queue.sh list
tools/queue.sh add --run-id <registered-run> --cell '<exact cell ID>'
tools/queue.sh remove --id 31
tools/queue.sh close
```

When the main checkout or interpreter path differs from the development
machine, also set `QUEUE_REMOTE_SCRIPT` and `QUEUE_REMOTE_PYTHON` to the verified
main paths. The client does not assume that a development-machine venv exists
inside a different image.

The client quotes argv and uses SSH; it creates no public API or second
scheduler. `close` seals admission, drains queued work and background judging,
then writes `terminal.json` and exits. New admission to that instance is
rejected after closing. Stopping a platform task still uses `ctl-exp stop`.
No JobPack is submitted merely by installing this entrypoint.

CPU tests additionally exercise SSH command serialization and detached remote
worker control through a simulated transport, node-loss claim preservation,
and admission sealing. Real SSH between platform nodes, shared-volume locks,
and GPU execution remain deployment acceptance checks; no platform job was
launched as part of this implementation.

## Cancellation and resume

New queue attempts enable `--atomic-checkpoints` for both native inference and
the detached evaluator. The unchanged execution checkout's `run.py` runs through
a small import hook that wraps only `vlmeval.smp.file.dump` at its normal import
time. It preserves the original serializer, cache paths and native resume
selection. Checkpoints are written beside the destination, fsynced, atomically
replaced and directory-fsynced. A serialization error preserves the previous
checkpoint and fails visibly, rather than publishing a changed fallback format.
Native phase manifests record `checkpoint_io=atomic-fsync-replace-v1`; model
source, prompts, decoding and judge configuration are unchanged. Queue state
and native phase records use the same durable commit principle.

Only committed progress can be retained. An unfinished inference batch or an
API response not yet committed can run again after cancellation. This is not
live migration of vLLM/RNG state or an exactly-once API-billing guarantee.
Already completed prediction rows and judge-cache entries retain their bytes.

`pause` stops new claims while active inference and judging continue; `unpause`
reenables claims. To cancel a whole platform allocation, use the existing
`ctl-exp stop`. Keep the shared queue directory and registered result roots.
For a new submission to continue it, set:

```bash
QUEUE_RESUME_ROOT=<previous persistent queue root>
QUEUE_PREVIOUS_TASK=<previous platform task>
QUEUE_PREVIOUS_STATUS=<saved terminal ctl-exp status JSON>
```

The new entrypoint requires a terminal reason and matching previous task,
RUN_ID and CTL_EXP_RUN_TS in that control-plane receipt. It refuses a live
controller, active inference/eval locks, or unreadable published pickle caches.
It records the old node/attempt identities, rebinds to the new nodes, preserves
all artifacts, and requeues interrupted work. Complete and deliberately removed
items stay complete/removed; ordinary failed tasks still require explicit retry.
Local stopped instances can use `queue.sh resume` then `queue.sh serve` after
their prior processes are confirmed gone. Merely losing SSH connectivity does
not authorize resume onto a different node.

Kill-during-write tests cover both prediction and judge caches. Resume tests
cover preserved checkpoint bytes, corrupt-checkpoint rejection, previous-task
identity, and running-vs-terminal platform receipts. Real platform cancellation
remains a deployment test; the existing completed 30-cell batch was not killed
or rewritten to test this feature.
