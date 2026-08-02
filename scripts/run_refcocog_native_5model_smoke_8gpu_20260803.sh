#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/user/zyc1781/vlmevalkit-release-perception-benchmarks}
export LMUData=${LMUData:-/user/zyc1781/LMUData}
export MODEL_ROOT=${MODEL_ROOT:-/user/zyc1781/models}
export CONDA_ROOT=${CONDA_ROOT:-/user/zyc1781/runtime/perception-benchmarks/conda}
export CONTROL_PYTHON=${CONTROL_PYTHON:-$CONDA_ROOT/envs/vlmevalkit/bin/python}

cd "$REPO"
git config --global --add safe.directory "$REPO"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK GROUP_RANK
unset SLURM_NODEID SLURM_PROCID SLURM_LOCALID SLURM_NTASKS SLURM_NPROCS
unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_LOCAL_RANK
unset PMI_RANK PMI_SIZE PMI_LOCAL_RANK PMIX_RANK
unset REFCOCO_COORDINATE_MODE

export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export VLMEVAL_LAZY_INIT=1
export VLMEVAL_VLM_MINIMAL_IMPORT=1
export VLMEVAL_API_MINIMAL_IMPORT=1

for path in \
  "$LMUData/RefCOCO.tsv" \
  "$MODEL_ROOT/gemma-3-4b-it/config.json" \
  "$MODEL_ROOT/gemma-3-12b-it/config.json" \
  "$MODEL_ROOT/gemma-3-27b-it/config.json" \
  "$MODEL_ROOT/MiniCPM-o-4_5/config.json" \
  "$MODEL_ROOT/MiniCPM-V-4_5/config.json" \
  "$CONDA_ROOT/envs/vlmevalkit/bin/python" \
  "$CONDA_ROOT/envs/vlmeval_gemma3_vllm/bin/python" \
  "$CONDA_ROOT/envs/vlmeval_minicpm45_vllm/bin/python"; do
  [[ -e "$path" ]] || { echo "[FATAL] missing preflight path: $path" >&2; exit 1; }
done

OUT_ROOT="$REPO/runs/refcocog_native_5model_smoke_20260803"
if [[ -e "$OUT_ROOT" ]] && find "$OUT_ROOT" -mindepth 1 -print -quit | grep -q .; then
  echo "[FATAL] smoke output root must be fresh: $OUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUT_ROOT/_gpu_monitor"

COMMON_ARGS=(
  --matrix-config configs/matrix_refcocog_native_5model_smoke_20260803.yaml
  --model-config configs/models_refcocog_native.yaml
  --nodes 1
  --node-rank 0
  --gpu-ids 0,1,2,3,4,5,6,7
  --task-manifest configs/task_manifests/refcocog_native_5model_smoke_20260803.csv
  --manifest-is-node-shard
  --scheduler gpu_pool
)

bash scripts/run_benchmark.sh "${COMMON_ARGS[@]}" --plan-only \
  | tee "$OUT_ROOT/_gpu_monitor/plan.log"

monitor_gpu() {
  while true; do
    date -Is
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits
    sleep 5
  done >>"$OUT_ROOT/_gpu_monitor/gpu_usage.log" 2>&1
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

"$CONTROL_PYTHON" scripts/validate_refcocog_native_smoke.py \
  --root "$OUT_ROOT" \
  --expected-records 2 \
  --output "$OUT_ROOT/smoke_acceptance.json"
touch "$OUT_ROOT/SMOKE_ACCEPTED"
