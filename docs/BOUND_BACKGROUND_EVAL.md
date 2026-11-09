# Bound runs with background evaluation

This is an opt-in extension of `scripts/run_benchmark.sh` and
`vlmeval/cli/run_benchmark.py`. The existing `gpu_pool` placement, TP sizes,
worker queues, prediction completeness checks, answer-format step and
`run.py --mode eval` are reused. No replacement GPU scheduler or model wrapper
is introduced. Historical `fg` and `skip` configurations keep their behavior.

## Prepare and launch

Use an ordinary, fully rendered native matrix and model YAML. Its absolute
`repo_root` selects the existing clean, pinned model/dataset checkout; the new
control runner need not contain that checkout's VLM adaptations. Put the
ordinary five selector columns in the task CSV and add `cell_id`:

```csv
model,policy,mode,transform,dataset,cell_id
minicpmo45,default,image_text_image_text,blank,OCRBench,最终协议|MiniCPM-o 4.5|1 blank|OCRBench
```

Run `python scripts/prepare_bound_run.py --help`. Supply the current DAG,
matrix, model YAML, tasks, a unique run ID, stack identity, and the topic's
`runs/bound` registry. `--check-only` validates without registering. Preparation
checks the table identity, model entry, mode, transform and explicit filler
type against the final protocol and native selectors. It rejects duplicate
cell registration and nonempty result roots. Dots/spaces with the same model
key use separate native batches so their environment and result path cannot
collide. No inference or API request occurs during preparation.

The prepared matrix sets:

```yaml
scheduler: gpu_pool
evaluation:
  launch_mode: bg
  judge: gpt-4o-mini
  nproc: 512
experiment:
  run_id: unique-run-id
  execution_sha: pinned-commit
  bindings:
    model__policy__mode__transform__dataset:
      cell_id: stable-final-cell-id
      protocol_id: final-v1
      tables: [table-id]
```

The topic's `scripts/run_bound.sh RUN_ID` loads the existing judge credential
helper and invokes the standard shell launcher in the foreground. Native
options such as `--gpu-ids` still apply. The configured execution SHA and clean
checkout are checked in both the launcher and its background evaluator.

## Lifecycle

After each valid prediction file, a detached CPU/API evaluation worker starts
and the native GPU pool releases the slot. The worker reconstructs the same
native task and calls the original answer-format and eval paths; it never
re-enters inference. Its lock is inherited by the eval subprocess, so a repeat
launch or a crashed supervisor cannot start another scorer while the original
scorer still owns the cell. Background-worker logs live at `eval/worker.log`;
the native detailed infer/eval logs remain under the cell's `_logs`.

`predictions/status.json` and `eval/status.json` record independent phases,
host, process identity, logs, run ID and table binding. Native final manifests
still determine prediction/score completion. A zero exit without score files
is failed, not complete. The launcher's `INFERENCE_DONE` message does not mean
all detached scorers have completed.

For exclusive-card recovery on the current development host, pass
`--gpu-ready-timeout 600` to the same native launcher. Before each inference
child it waits for all assigned cards to use at most 1024 MiB (the existing
topic launch scripts' free-card criterion). Busy or unreadable cards time out
with failed status and no model launch. The guard is off by default, changes
no model memory/decoding settings, and does not reserve cards against other
launchers. It mitigates the observed startup-memory failures; it does not
establish which process held the card or prove a vLLM cleanup defect.

Each bound eval receives its own existing dynamic-concurrency gate and a
512-worker ceiling. Multiple cells can therefore exceed 512 fleet-wide.
Historical overrides of `LOCAL_LLM` or omitted temperature are cleared in
bound evals; gpt-4o-mini keeps the existing dated model mapping and temperature
0. Official rule-based datasets retain their rule scorer. This is a Linux
process integration using `flock`, inherited FDs and detached sessions.

## Viewer

The DAG owns table scope; `registration.json` owns each run-to-cell binding.
The viewer reads registered native manifests without scheduling anything.
Its five-second loop publishes a live overlay and final CSVs; the browser
updates cells, counts and complete-suite deltas without reloading the page.
Only changed score artifacts invoke the existing official per-sample reader.
A stale watcher is visible; unknown remote process liveness is not called
verified. Old in-flight tasks are observed under their original settings.

## Verification boundary

`python scripts/test_bound_runner.py` runs the real native scheduler and
detached subprocesses with tiny CPU-only fixture commands. It checks overlap,
GPU-slot release, duplicate prevention, separate 512 gates, correct bindings,
failed scoring, and exit-zero-without-score rejection. It makes no VLM or judge
API calls; it is not a formal benchmark or GPU/API acceptance run.
