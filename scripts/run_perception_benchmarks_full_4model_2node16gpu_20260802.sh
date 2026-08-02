#!/usr/bin/env bash
set -euo pipefail

detect_job_node_rank() {
  for key in NODE_RANK SLURM_NODEID RANK OMPI_COMM_WORLD_RANK PMI_RANK; do
    if [[ -n "${!key:-}" ]]; then
      printf '%s\n' "${!key}"
      return 0
    fi
  done
  echo "[FATAL] unable to detect node rank for a two-node job" >&2
  return 1
}

wait_for_marker() {
  local path=$1
  local timeout_seconds=$2
  local started=$SECONDS
  while [[ ! -e "$path" ]]; do
    if (( SECONDS - started >= timeout_seconds )); then
      echo "[FATAL] timed out waiting for $path" >&2
      return 1
    fi
    sleep 10
  done
  local marker_value
  marker_value=$(<"$path")
  if [[ "$marker_value" != "$RUN_UUID" ]]; then
    echo "[FATAL] marker UUID mismatch for $path: $marker_value" >&2
    return 1
  fi
}

write_marker() {
  local path=$1
  local tmp="${path}.tmp.${JOB_NODE_RANK}.$$"
  printf '%s\n' "$RUN_UUID" >"$tmp"
  mv "$tmp" "$path"
}

JOB_NODE_RANK=$(detect_job_node_rank)
JOB_MASTER_ADDR=${MASTER_ADDR:-}
EXPECTED_NODES=2
if [[ ! "$JOB_NODE_RANK" =~ ^[01]$ ]]; then
  echo "[FATAL] node rank must be 0 or 1, got: $JOB_NODE_RANK" >&2
  exit 1
fi

REPO=${REPO:-/user/zyc1781/vlmevalkit-release-perception-benchmarks}
export LMUData=${LMUData:-/user/zyc1781/LMUData}
export MODEL_ROOT=${MODEL_ROOT:-/user/zyc1781/models}
export CONDA_ROOT=${CONDA_ROOT:-/user/zyc1781/runtime/perception-benchmarks/conda}
export CONTROL_PYTHON=${CONTROL_PYTHON:-$CONDA_ROOT/envs/vlmevalkit/bin/python}
if [[ -n "${PERCEPTION_RUN_UUID:-}" ]]; then
  RUN_UUID=$PERCEPTION_RUN_UUID
elif [[ -n "$JOB_MASTER_ADDR" ]]; then
  RUN_UUID="perception-$(printf '%s' "$JOB_MASTER_ADDR" | sha256sum | cut -c1-20)"
else
  echo "[FATAL] MASTER_ADDR is required to derive a submission-scoped run UUID" >&2
  exit 1
fi
if [[ ! "$RUN_UUID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[FATAL] unsafe PERCEPTION_RUN_UUID: $RUN_UUID" >&2
  exit 1
fi

cd "$REPO"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Each cell starts its own isolated vLLM process. Never leak the outer
# two-node allocation topology into run.py.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK GROUP_RANK

export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export VLMEVAL_LAZY_INIT=1
export VLMEVAL_VLM_MINIMAL_IMPORT=1
export VLMEVAL_API_MINIMAL_IMPORT=1
export REFCOCO_COORDINATE_MODE=normalized_0_1_xyxy

for path in \
  "$LMUData/CountQA.manifest.json" \
  "$LMUData/SpatialMQA/manifest.json" \
  "$LMUData/RefCOCO.tsv" \
  "$MODEL_ROOT/Qwen2.5-VL-3B-Instruct/config.json" \
  "$MODEL_ROOT/gemma-3-4b-it/config.json" \
  "$MODEL_ROOT/gemma-3-12b-it/config.json" \
  "$MODEL_ROOT/MiniCPM-o-4_5/config.json" \
  "$CONDA_ROOT/envs/vlmevalkit/bin/python" \
  "$CONDA_ROOT/envs/vlmeval_gemma3_vllm/bin/python" \
  "$CONDA_ROOT/envs/vlmeval_minicpm45_vllm/bin/python"; do
  [[ -e "$path" ]] || { echo "[FATAL] missing preflight path: $path" >&2; exit 1; }
done

OUT_ROOT="$REPO/runs/perception_benchmarks_full_4model_2node16gpu_20260802"
RUN_ID_FILE="$OUT_ROOT/_control/run_uuid"
PREFLIGHT_READY="$OUT_ROOT/_inputs/PREFLIGHT_READY"
WAIT_TIMEOUT_SECONDS=${WAIT_TIMEOUT_SECONDS:-21600}
PREFLIGHT_TIMEOUT_SECONDS=${PREFLIGHT_TIMEOUT_SECONDS:-1800}

if [[ "$JOB_NODE_RANK" == "0" ]]; then
  if [[ -e "$OUT_ROOT" ]] && find "$OUT_ROOT" -mindepth 1 -print -quit | grep -q .; then
    echo "[FATAL] output root must be fresh for this submission: $OUT_ROOT" >&2
    exit 1
  fi
  mkdir -p "$OUT_ROOT/_control" "$OUT_ROOT/_inputs"
  write_marker "$RUN_ID_FILE"
  "$CONTROL_PYTHON" scripts/build_dataset_split_allowlist.py \
    --dataset RefCOCO \
    --split-column split \
    --split-value RefCOCOg_test \
    --expected-count 9602 \
    --output "$OUT_ROOT/_inputs/refcocog_test_indices.txt" \
    --manifest-output "$OUT_ROOT/_inputs/refcocog_test_indices.manifest.json"
  "$CONTROL_PYTHON" scripts/materialize_dataset_split_images.py \
    --dataset RefCOCO \
    --split-column split \
    --split-value RefCOCOg_test \
    --expected-count 9602 \
    --manifest-output "$OUT_ROOT/_inputs/refcocog_test_images.manifest.json"
  "$CONTROL_PYTHON" scripts/validate_perception_dataset_protocols.py \
    --output "$OUT_ROOT/dataset_protocol_full_preflight.json"
  write_marker "$PREFLIGHT_READY"
else
  wait_for_marker "$PREFLIGHT_READY" "$PREFLIGHT_TIMEOUT_SECONDS"
fi

NODE_MONITOR_ROOT="$OUT_ROOT/_gpu_monitor/node${JOB_NODE_RANK}"
mkdir -p "$NODE_MONITOR_ROOT"
TASK_MANIFEST="configs/task_manifests/perception_benchmarks_full_4model_2node16gpu_20260802/node${JOB_NODE_RANK}_tasks.csv"
COMMON_ARGS=(
  --matrix-config configs/matrix_perception_benchmarks_full_4model_2node16gpu_20260802.yaml
  --model-config configs/models.yaml
  --nodes "$EXPECTED_NODES"
  --node-rank "$JOB_NODE_RANK"
  --gpu-ids 0,1,2,3,4,5,6,7
  --task-manifest "$TASK_MANIFEST"
  --manifest-is-node-shard
  --scheduler gpu_pool
)

bash scripts/run_benchmark.sh "${COMMON_ARGS[@]}" --plan-only \
  | tee "$NODE_MONITOR_ROOT/plan.log"

monitor_gpu() {
  while true; do
    {
      date -Is
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits
      nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
        --format=csv,noheader,nounits || true
    } >>"$NODE_MONITOR_ROOT/gpu_usage.log" 2>&1
    sleep 10
  done
}

monitor_gpu &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

bash scripts/run_benchmark.sh "${COMMON_ARGS[@]}" \
  2>&1 | tee "$NODE_MONITOR_ROOT/runner.log"

write_marker "$OUT_ROOT/NODE_${JOB_NODE_RANK}_RUNNER_DONE"
if [[ "$JOB_NODE_RANK" == "0" ]]; then
  wait_for_marker "$OUT_ROOT/NODE_1_RUNNER_DONE" "$WAIT_TIMEOUT_SECONDS"
  "$CONTROL_PYTHON" scripts/validate_perception_full_results.py \
    --root "$OUT_ROOT" \
    --models qwen25vl_3b gemma3_4b gemma3_12b minicpm_o_45_no_reasoning \
    --matrix perception_benchmarks_full_4model_2node16gpu_20260802 \
    --run-uuid "$RUN_UUID" \
    --allowlist-manifest "$OUT_ROOT/_inputs/refcocog_test_indices.manifest.json" \
    --json-output "$OUT_ROOT/full_result_validation.json" \
    --csv-output "$OUT_ROOT/full_results.csv"
  write_marker "$OUT_ROOT/FULL_RUN_ACCEPTED_BY_RUNNER"
fi
