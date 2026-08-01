#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/user/zyc1781/vlmevalkit-release-perception-benchmarks}
export LMUData=${LMUData:-/user/zyc1781/LMUData}
export MODEL_ROOT=${MODEL_ROOT:-/user/zyc1781/models}
export CONDA_ROOT=${CONDA_ROOT:-/user/zyc1781/runtime/perception-benchmarks/conda}
export CONTROL_PYTHON=${CONTROL_PYTHON:-$CONDA_ROOT/envs/vlmevalkit/bin/python}

unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT NODE_RANK GROUP_RANK

cd "$REPO"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Experiment semantics must not depend on stale values from a machine-local .env.
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

OUT_ROOT="$REPO/runs/perception_benchmarks_smoke_8gpu_20260731"
rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT/_gpu_monitor"

"$CONTROL_PYTHON" scripts/validate_perception_dataset_protocols.py \
  --output "$OUT_ROOT/dataset_protocol_smoke.json"

COMMON_ARGS=(
  --matrix-config configs/matrix_perception_benchmarks_smoke_8gpu_20260731.yaml
  --model-config configs/models.yaml
  --nodes 1
  --node-rank 0
  --gpu-ids 0,1,2,3,4,5,6,7
  --task-manifest configs/task_manifests/perception_benchmarks_smoke_8gpu_20260731.csv
  --manifest-is-node-shard
  --scheduler gpu_pool
)

bash scripts/run_benchmark.sh "${COMMON_ARGS[@]}" --plan-only \
  | tee "$OUT_ROOT/_gpu_monitor/plan.log"

monitor_gpu() {
  while true; do
    {
      date -Is
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits
      nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
        --format=csv,noheader,nounits || true
    } >>"$OUT_ROOT/_gpu_monitor/gpu_usage.log" 2>&1
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
  2>&1 | tee "$OUT_ROOT/_gpu_monitor/runner.log"

REQUIRED_TASK_ARGS=()
for dataset in CountQA SpatialMQA RefCOCO; do
  for model in qwen25vl_3b gemma3_4b gemma3_12b minicpm_o_45_no_reasoning; do
    for condition in iq iqiq; do
      REQUIRED_TASK_ARGS+=(--required-task "$dataset:$model:$condition")
    done
  done
done

"$CONTROL_PYTHON" scripts/validate_perception_runtime_smoke.py \
  --root "$OUT_ROOT" \
  --expect-records-per-task 2 \
  "${REQUIRED_TASK_ARGS[@]}" \
  --output "$OUT_ROOT/runtime_smoke_validation.json"

touch "$OUT_ROOT/SMOKE_ACCEPTED_BY_RUNNER"
