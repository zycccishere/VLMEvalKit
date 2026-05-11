#!/usr/bin/env python3
from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from run_benchmark import BenchmarkRunner, parse_args, split_names


class TaskBalancedBenchmarkRunner(BenchmarkRunner):
    """Assign tasks directly to nodes instead of pinning whole models."""

    def __init__(self, script_dir: Path, args) -> None:
        super().__init__(script_dir, args)
        exclude_tags = set(split_names(os.environ.get("TASK_TAG_EXCLUDELIST", "")))
        if exclude_tags:
            self.tasks = [task for task in self.tasks if task.tag not in exclude_tags]

    def plan(self) -> list[list]:
        buckets = [[] for _ in range(self.args.nodes)]
        loads = [0.0 for _ in range(self.args.nodes)]
        tasks = sorted(
            self.tasks,
            key=lambda task: (-task.weight, self.model_order.index(task.model_key), task.index),
        )
        for task in tasks:
            node_index = min(
                range(self.args.nodes),
                key=lambda idx: (loads[idx], len(buckets[idx]), idx),
            )
            buckets[node_index].append(task)
            loads[node_index] += task.weight
        for bucket in buckets:
            bucket.sort(key=lambda task: task.index)
        self.planned_loads = loads
        return buckets

    def _pick_next_task(self, pending: list, free_gpus: list[str]):
        for idx, task in enumerate(pending):
            model = self.models[task.model_key]
            if model.gpus_per_job <= len(free_gpus):
                return idx, task
        return None, None

    def run(self) -> int:
        buckets = self.plan()
        self.print_plan(buckets)
        if self.args.plan_only:
            return 0
        if self.args.node_rank < 0 or self.args.node_rank >= self.args.nodes:
            raise SystemExit(f"node-rank out of range: {self.args.node_rank} / {self.args.nodes}")

        assigned = buckets[self.args.node_rank]
        if not assigned:
            print(f"[NODE][IDLE] node_rank={self.args.node_rank} has no assigned tasks.", flush=True)
            return 0

        pending = sorted(
            assigned,
            key=lambda task: (
                -self.models[task.model_key].gpus_per_job,
                -task.weight,
                self.model_order.index(task.model_key),
                task.index,
            ),
        )
        free_gpus = list(self.node_gpu_ids)
        running: dict = {}

        print(
            f"[NODE][MIXED] node_rank={self.args.node_rank} tasks={len(pending)} "
            f"gpus={','.join(free_gpus)} scheduling=slot-dynamic",
            flush=True,
        )

        with ThreadPoolExecutor(max_workers=len(assigned)) as pool:
            while pending or running:
                launched = False
                while pending:
                    pick_idx, task = self._pick_next_task(pending, free_gpus)
                    if task is None:
                        break
                    model = self.models[task.model_key]
                    self.ensure_profile_ready(model)
                    gpu_ids = free_gpus[: model.gpus_per_job]
                    del free_gpus[: model.gpus_per_job]
                    pending.pop(pick_idx)
                    print(
                        f"[NODE][LAUNCH] node_rank={self.args.node_rank} task={task.tag} "
                        f"gpus={','.join(gpu_ids)} free_after={','.join(free_gpus) if free_gpus else '-'}",
                        flush=True,
                    )
                    future = pool.submit(self.run_single_task, model, task, gpu_ids)
                    running[future] = (task, gpu_ids)
                    launched = True

                if running and (pending and not launched or not pending):
                    done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        task, gpu_ids = running.pop(future)
                        try:
                            future.result()
                            phase = "DONE"
                        except Exception as exc:
                            phase = "FAIL"
                            print(f"[NODE][TASK-{phase}] {task.tag}: {exc}", flush=True)
                        free_gpus.extend(gpu_ids)
                        free_gpus.sort(key=lambda x: int(x))
                        print(
                            f"[NODE][TASK-{phase}] node_rank={self.args.node_rank} task={task.tag} "
                            f"released={','.join(gpu_ids)} free_now={','.join(free_gpus)}",
                            flush=True,
                        )
                elif pending and not running and not launched:
                    raise RuntimeError(
                        f"No schedulable task found with free_gpus={free_gpus} and pending={len(pending)}"
                    )

        print(f"[NODE][DONE] node_rank={self.args.node_rank}", flush=True)
        return 0


def main() -> int:
    args = parse_args()
    runner = TaskBalancedBenchmarkRunner(Path(__file__).resolve().parent, args)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
