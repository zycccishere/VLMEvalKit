#!/usr/bin/env bash
set -euo pipefail

GROUP=${REFCOCO_NATIVE_GROUP:-}
case "$GROUP" in
  gemma|minicpm) ;;
  *) echo "[FATAL] REFCOCO_NATIVE_GROUP must be gemma or minicpm" >&2; exit 1 ;;
esac

REPO=${REPO:-/user/zyc1781/vlmevalkit-release-perception-benchmarks}
export LMUData=${LMUData:-/user/zyc1781/LMUData}
export MODEL_ROOT=${MODEL_ROOT:-/user/zyc1781/models}
export CONDA_ROOT=${CONDA_ROOT:-/user/zyc1781/runtime/perception-benchmarks/conda}
export CONTROL_PYTHON=${CONTROL_PYTHON:-$CONDA_ROOT/envs/vlmevalkit/bin/python}
RUN_UUID=${REFCOCO_NATIVE_RUN_UUID:-}
if [[ -z "$RUN_UUID" || ! "$RUN_UUID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[FATAL] REFCOCO_NATIVE_RUN_UUID must be a nonempty safe identifier" >&2
  exit 1
fi

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

COMMON_PATHS=(
  "$LMUData/RefCOCO.tsv"
  "$CONDA_ROOT/envs/vlmevalkit/bin/python"
)
if [[ "$GROUP" == gemma ]]; then
  COMMON_PATHS+=(
    "$MODEL_ROOT/gemma-3-4b-it/config.json"
    "$MODEL_ROOT/gemma-3-12b-it/config.json"
    "$MODEL_ROOT/gemma-3-27b-it/config.json"
    "$CONDA_ROOT/envs/vlmeval_gemma3_vllm/bin/python"
  )
else
  COMMON_PATHS+=(
    "$MODEL_ROOT/MiniCPM-o-4_5/config.json"
    "$MODEL_ROOT/MiniCPM-V-4_5/config.json"
    "$CONDA_ROOT/envs/vlmeval_minicpm45_vllm/bin/python"
  )
fi
for path in "${COMMON_PATHS[@]}"; do
  [[ -e "$path" ]] || { echo "[FATAL] missing preflight path: $path" >&2; exit 1; }
done

MATRIX_CONFIG=${REFCOCO_NATIVE_MATRIX_CONFIG:-configs/matrix_refcocog_native_5model_full_20260803.yaml}
MATRIX_NAME=${REFCOCO_NATIVE_MATRIX_NAME:-refcocog_native_5model_full_20260803}
OUT_ROOT=${REFCOCO_NATIVE_OUT_ROOT:-$REPO/runs/refcocog_native_5model_full_20260803}
mkdir -p "$OUT_ROOT/_control" "$OUT_ROOT/_inputs" "$OUT_ROOT/_gpu_monitor/$GROUP"
GROUP_MARKER="$OUT_ROOT/_control/${GROUP}_runner_done"
if [[ -e "$GROUP_MARKER" ]]; then
  echo "[FATAL] group already completed: $GROUP_MARKER" >&2
  exit 1
fi

exec 9>"$OUT_ROOT/_control/preflight.lock"
flock 9
CURRENT_CONTRACT=$(
  find run.py vlmeval configs scripts -type f \
    ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | awk '{print $1}'
)
CONTRACT_FILE="$OUT_ROOT/_control/code_config_contract.sha256"
if [[ -e "$CONTRACT_FILE" ]]; then
  [[ "$(<"$CONTRACT_FILE")" == "$CURRENT_CONTRACT" ]] || {
    echo "[FATAL] code/config contract differs between groups" >&2
    exit 1
  }
else
  printf '%s\n' "$CURRENT_CONTRACT" >"$CONTRACT_FILE"
  git rev-parse HEAD >"$OUT_ROOT/_control/git_commit.txt"
  git status --short >"$OUT_ROOT/_control/git_status.txt"
fi
RUN_ID_FILE="$OUT_ROOT/_control/run_uuid"
if [[ -e "$RUN_ID_FILE" ]]; then
  [[ "$(<"$RUN_ID_FILE")" == "$RUN_UUID" ]] || {
    echo "[FATAL] run UUID mismatch in shared output root" >&2
    exit 1
  }
else
  printf '%s\n' "$RUN_UUID" >"$RUN_ID_FILE"
fi
if [[ ! -e "$OUT_ROOT/_inputs/PREFLIGHT_READY" ]]; then
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
  printf '%s\n' "$RUN_UUID" >"$OUT_ROOT/_inputs/PREFLIGHT_READY"
fi
[[ "$(<"$OUT_ROOT/_inputs/PREFLIGHT_READY")" == "$RUN_UUID" ]] || {
  echo "[FATAL] preflight marker UUID mismatch" >&2
  exit 1
}
flock -u 9

TASK_MANIFEST="configs/task_manifests/refcocog_native_5model_full_20260803/${GROUP}_tasks.csv"
COMMON_ARGS=(
  --matrix-config "$MATRIX_CONFIG"
  --model-config configs/models_refcocog_native.yaml
  --nodes 1
  --node-rank 0
  --gpu-ids 0,1,2,3,4,5,6,7
  --task-manifest "$TASK_MANIFEST"
  --manifest-is-node-shard
  --scheduler gpu_pool
)

bash scripts/run_benchmark.sh "${COMMON_ARGS[@]}" --plan-only \
  | tee "$OUT_ROOT/_gpu_monitor/$GROUP/plan.log"

monitor_gpu() {
  while true; do
    date -Is
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
      --format=csv,noheader,nounits || true
    sleep 10
  done >>"$OUT_ROOT/_gpu_monitor/$GROUP/gpu_usage.log" 2>&1
}

monitor_gpu &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

bash scripts/run_benchmark.sh "${COMMON_ARGS[@]}" \
  2>&1 | tee "$OUT_ROOT/_gpu_monitor/$GROUP/runner.log"

if [[ "$GROUP" == gemma ]]; then
  GROUP_MODELS=(gemma3_4b gemma3_12b gemma3_27b)
else
  GROUP_MODELS=(minicpm_o_45_no_reasoning minicpm_v_45_no_reasoning)
fi
"$CONTROL_PYTHON" scripts/validate_refcocog_native_full.py \
  --root "$OUT_ROOT" \
  --matrix "$MATRIX_NAME" \
  --allowlist "$OUT_ROOT/_inputs/refcocog_test_indices.txt" \
  --models "${GROUP_MODELS[@]}" \
  --expected-rows 9602 \
  --json-output "$OUT_ROOT/${GROUP}_result_validation.json" \
  --csv-output "$OUT_ROOT/${GROUP}_results.csv"

printf '%s\n' "$RUN_UUID" >"$GROUP_MARKER"

exec 8>"$OUT_ROOT/_control/finalize.lock"
flock 8
if [[ -e "$OUT_ROOT/_control/gemma_runner_done" && -e "$OUT_ROOT/_control/minicpm_runner_done" ]]; then
  [[ "$(<"$OUT_ROOT/_control/gemma_runner_done")" == "$RUN_UUID" ]]
  [[ "$(<"$OUT_ROOT/_control/minicpm_runner_done")" == "$RUN_UUID" ]]
  "$CONTROL_PYTHON" scripts/validate_refcocog_native_full.py \
    --root "$OUT_ROOT" \
    --matrix "$MATRIX_NAME" \
    --allowlist "$OUT_ROOT/_inputs/refcocog_test_indices.txt" \
    --models gemma3_4b gemma3_12b gemma3_27b minicpm_o_45_no_reasoning minicpm_v_45_no_reasoning \
    --expected-rows 9602 \
    --json-output "$OUT_ROOT/full_result_validation.json" \
    --csv-output "$OUT_ROOT/full_results.csv"
  printf '%s\n' "$RUN_UUID" >"$OUT_ROOT/FULL_RUN_ACCEPTED"
fi
