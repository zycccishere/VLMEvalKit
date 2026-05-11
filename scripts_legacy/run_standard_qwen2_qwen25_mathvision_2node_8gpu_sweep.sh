#!/usr/bin/env bash
set -euo pipefail

if [[ "${SWEEP_DEBUG:-0}" == "1" ]]; then
    set -x
fi

# 2-node x 8-GPU-per-node sweep for MathVision replay settings.
# Target matrix:
#   4 models x 5 replay modes x 2 last-text flags = 40 tasks
# Default scheduling:
#   NUM_NODES=2, JOBS_PER_NODE=8, GPUS_PER_JOB=1  -> 16 concurrent tasks
#
# Run this script on EACH node. If launcher rank env exists (SLURM/torchrun/mpi),
# NODE_RANK is auto-detected; otherwise set NODE_RANK manually per node.

DEFAULT_NUM_NODES=2
DEFAULT_JOBS_PER_NODE=8
DEFAULT_GPUS_PER_JOB=1
DEFAULT_NODE_GPU_IDS="0,1,2,3,4,5,6,7"

detect_node_rank() {
    if [[ -n "${NODE_RANK:-}" ]]; then
        echo "${NODE_RANK}"
        return 0
    fi
    if [[ -n "${SLURM_NODEID:-}" ]]; then
        echo "${SLURM_NODEID}"
        return 0
    fi
    if [[ -n "${RANK:-}" ]]; then
        echo "${RANK}"
        return 0
    fi
    if [[ -n "${OMPI_COMM_WORLD_RANK:-}" ]]; then
        echo "${OMPI_COMM_WORLD_RANK}"
        return 0
    fi
    if [[ -n "${PMI_RANK:-}" ]]; then
        echo "${PMI_RANK}"
        return 0
    fi
    echo "0"
}

init_cluster_env() {
    local has_explicit_rank=0
    if [[ -n "${NODE_RANK:-}" ]]; then
        has_explicit_rank=1
    fi

    export NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-$DEFAULT_NUM_NODES}}"
    export JOBS_PER_NODE="${JOBS_PER_NODE:-$DEFAULT_JOBS_PER_NODE}"
    export GPUS_PER_JOB="${GPUS_PER_JOB:-$DEFAULT_GPUS_PER_JOB}"
    export NODE_GPU_IDS="${NODE_GPU_IDS:-$DEFAULT_NODE_GPU_IDS}"
    export NODE_RANK="$(detect_node_rank)"
    if [[ -z "${NODE_RANK:-}" ]]; then
        export NODE_RANK=0
    fi

    if [[ "$has_explicit_rank" == "0" && -z "${SLURM_NODEID:-}" && -z "${RANK:-}" && -z "${OMPI_COMM_WORLD_RANK:-}" && -z "${PMI_RANK:-}" ]]; then
        if [[ "${NUM_NODES}" != "1" ]]; then
            echo "[WARN] Rank env not detected; using NODE_RANK=${NODE_RANK}. Set NODE_RANK manually per node if needed." >&2
        fi
    fi
}

activate_env() {
    export PATH=/usr/local/cuda/bin:$PATH
    if [[ -f /opt/miniconda3/bin/activate ]]; then
        # shellcheck source=/dev/null
        source /opt/miniconda3/bin/activate
    fi

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV_NAME:-vlmevalkit}"
    else
        echo "[FATAL] conda not found after sourcing activate script." >&2
        exit 1
    fi
}

init_common_env() {
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export FORCE_LOCAL=True
    export OLD_VERSION='False'

    export REPLAY_TIMES="${REPLAY_TIMES:-1}"
    export REPLAY_DEBUG="${REPLAY_DEBUG:-0}"
    export REPLAY_LIMIT_MM_PER_PROMPT="${REPLAY_LIMIT_MM_PER_PROMPT:-2}"
    export REPLAY_IMAGE_COPY_MODE="${REPLAY_IMAGE_COPY_MODE:-reuse_path}"
    export VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    export VLLM_USE_V1="${VLLM_USE_V1:-0}"
    export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
    export REPLAY_SAFE_FALLBACK="${REPLAY_SAFE_FALLBACK:-1}"
    export REPLAY_SAFE_TRUNCATE_CHARS="${REPLAY_SAFE_TRUNCATE_CHARS:-6000}"

    export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME:-directly_answer}"
    export REPLAY_STAGE_DEBUG="${REPLAY_STAGE_DEBUG:-1}"
    export REPLAY_STAGE_DEBUG_SAMPLES="${REPLAY_STAGE_DEBUG_SAMPLES:-8}"
    export REPLAY_PROMPT_AUDIT="${REPLAY_PROMPT_AUDIT:-1}"
    export REPLAY_PROMPT_AUDIT_PRINT="${REPLAY_PROMPT_AUDIT_PRINT:-1}"

    export ANSWER_FORMAT_ENABLE="${ANSWER_FORMAT_ENABLE:-1}"
    export ANSWER_FORMAT_REQUIRE_BOXED="${ANSWER_FORMAT_REQUIRE_BOXED:-0}"
    export ANSWER_FORMAT_RESPONSE_COL="${ANSWER_FORMAT_RESPONSE_COL:-prediction}"
    export ANSWER_FORMAT_FALLBACK_COL="${ANSWER_FORMAT_FALLBACK_COL:-detailed_prediction}"
    export ANSWER_FORMAT_MAX_FAILS="${ANSWER_FORMAT_MAX_FAILS:-50}"

    export SAVE_ROOT="${SAVE_ROOT:-/path/to/vlmevalkit}"
    export EXP_DATE_TAG="${EXP_DATE_TAG:-$(date +%Y%m%d)}"
    export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen2_qwen25_mathvision_replay_2node16gpu}"

    export DATALIST="${DATALIST:-MathVision}"
    export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-32}"
}

build_tasks() {
    TASK_MODEL_TAGS=()
    TASK_MODEL_PATHS=()
    TASK_MODES=()
    TASK_LAST_FLAGS=()
    TASK_TAGS=()

    local -a model_tags=(
        Qwen2-VL-2B-Instruct
        Qwen2-VL-7B-Instruct
        Qwen2.5-VL-3B-Instruct
        Qwen2.5-VL-7B-Instruct
    )
    local -a model_paths=(
        /models/Qwen2-VL-2B-Instruct
        /models/Qwen2-VL-7B-Instruct
        /models/Qwen2.5-VL-3B-Instruct
        /models/Qwen2.5-VL-7B-Instruct
    )
    local -a modes=(
        none
        image_text_text
        image_text_image
        image_text_image_text
        image_image_text
    )

    local mi
    local mode
    local last
    for mi in "${!model_tags[@]}"; do
        for mode in "${modes[@]}"; do
            for last in 0 1; do
                TASK_MODEL_TAGS+=("${model_tags[$mi]}")
                TASK_MODEL_PATHS+=("${model_paths[$mi]}")
                TASK_MODES+=("$mode")
                TASK_LAST_FLAGS+=("$last")
                TASK_TAGS+=("${model_tags[$mi]}__${mode}__last${last}")
            done
        done
    done
}

filter_tasks_by_allowlist() {
    local raw="${TASK_TAG_ALLOWLIST:-}"
    if [[ -z "$raw" ]]; then
        return 0
    fi

    local normalized
    normalized="$(echo "$raw" | tr ',' ' ')"

    declare -A allow_map=()
    local tag
    for tag in $normalized; do
        allow_map["$tag"]=1
    done

    local -a keep_model_tags=()
    local -a keep_model_paths=()
    local -a keep_modes=()
    local -a keep_last_flags=()
    local -a keep_tags=()

    local i
    for i in "${!TASK_TAGS[@]}"; do
        if [[ -n "${allow_map[${TASK_TAGS[$i]}]:-}" ]]; then
            keep_model_tags+=("${TASK_MODEL_TAGS[$i]}")
            keep_model_paths+=("${TASK_MODEL_PATHS[$i]}")
            keep_modes+=("${TASK_MODES[$i]}")
            keep_last_flags+=("${TASK_LAST_FLAGS[$i]}")
            keep_tags+=("${TASK_TAGS[$i]}")
        fi
    done

    TASK_MODEL_TAGS=("${keep_model_tags[@]}")
    TASK_MODEL_PATHS=("${keep_model_paths[@]}")
    TASK_MODES=("${keep_modes[@]}")
    TASK_LAST_FLAGS=("${keep_last_flags[@]}")
    TASK_TAGS=("${keep_tags[@]}")
}

gpu_slice_for_slot() {
    local slot="$1"
    local gpus_per_job="$2"
    local gpu_csv="$3"

    IFS=',' read -r -a all_gpus <<< "$gpu_csv"
    local need=$(( (slot + 1) * gpus_per_job ))
    if (( ${#all_gpus[@]} < need )); then
        echo "[FATAL] NODE_GPU_IDS has ${#all_gpus[@]} GPUs, but slot=${slot} requires at least ${need}." >&2
        exit 1
    fi

    local start=$(( slot * gpus_per_job ))
    local end=$(( start + gpus_per_job ))
    local picked=()
    local i
    for ((i = start; i < end; i++)); do
        picked+=("${all_gpus[$i]}")
    done

    local out
    out=$(IFS=','; echo "${picked[*]}")
    echo "$out"
}

run_one_task() {
    local model_tag="$1"
    local model_path="$2"
    local mode="$3"
    local last_flag="$4"
    local task_tag="$5"
    local gpu_ids="$6"

    export CUDA_VISIBLE_DEVICES="$gpu_ids"
    export MODEL_PATH="$model_path"
    export REPLAY_MODE="$mode"
    export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="$last_flag"

    local save_prefix="runs/standard/${EXP_DATE_TAG}/${EXP_GROUP_TAG}/${task_tag}"
    work_dir="${SAVE_ROOT}/${save_prefix}/output"
    export REPLAY_DUMP_DIR="${REPLAY_DUMP_DIR:-${work_dir}/_logs/replay_dump}"
    export REPLAY_DUMP_MAX_CHARS="${REPLAY_DUMP_MAX_CHARS:-20000}"

    echo "[TASK][START] ${task_tag}"
    echo "[TASK][INFO] MODEL_PATH=${MODEL_PATH}"
    echo "[TASK][INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[TASK][INFO] REPLAY_MODE=${REPLAY_MODE} REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT=${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT}"
    echo "[TASK][INFO] work_dir=${work_dir}"
    echo "[TASK][INFO] DATALIST=${DATALIST}"

    if [[ "${SWEEP_DRY_RUN:-0}" == "1" ]]; then
        echo "[TASK][DRY-RUN] skip real infer/eval for ${task_tag}"
        return 0
    fi

    # Use generic replay model entry + dynamic MODEL_PATH to cover Qwen2/Qwen2.5 2B/3B/7B uniformly.
    local model_name="Qwen2VLChatReplay"

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # Backward-compatibility for sourced helpers expecting SCRIPT_DIR.
    SCRIPT_DIR="$script_dir"
    # shellcheck source=/dev/null
    source "${script_dir}/run_standard_guard.sh"
    standard_guard_init

    local dataset
    for dataset in $DATALIST; do
        local expected_count
        expected_count=$(get_expected_count "$dataset")
        echo "Expected samples for ${dataset}: ${expected_count}"
        if [ "$expected_count" -lt 0 ]; then
            echo "[SKIP][DATASET] ${model_tag}/${model_name} x ${dataset}: dataset unavailable or build failed."
            continue
        fi

        if infer_complete "$model_name" "$dataset" "$expected_count"; then
            echo "[SKIP][INFER] ${model_tag}/${model_name} x ${dataset}: infer result is complete."
        else
            if infer_artifacts_exist "$model_name" "$dataset"; then
                echo "[CLEAN][INFER+EVAL] ${model_tag}/${model_name} x ${dataset}: infer incomplete, remove stale artifacts."
                cleanup_all_artifacts "$model_name" "$dataset"
            fi
            if ! launch_infer_fg "$model_name" "$dataset" "$INFER_BATCH_SIZE"; then
                echo "[SKIP][EVAL] ${model_tag}/${model_name} x ${dataset}: infer failed."
                continue
            fi
        fi

        if infer_complete "$model_name" "$dataset" "$expected_count"; then
            run_answer_format_postprocess "$model_name" "$dataset"
            if eval_complete "$model_name" "$dataset" "$expected_count"; then
                echo "[SKIP][EVAL] ${model_tag}/${model_name} x ${dataset}: eval result is complete."
            else
                if eval_artifacts_exist "$model_name" "$dataset"; then
                    echo "[CLEAN][EVAL] ${model_tag}/${model_name} x ${dataset}: eval incomplete, remove stale eval artifacts."
                    cleanup_eval_artifacts "$model_name" "$dataset"
                fi
                launch_eval_bg "$model_name" "$dataset"
            fi
        else
            echo "[SKIP][EVAL] ${model_tag}/${model_name} x ${dataset}: infer is still incomplete."
        fi
    done

    echo "[TASK][DONE] ${task_tag}"
}

worker_main() {
    local local_slot="$1"

    local num_nodes="${NUM_NODES}"
    local node_rank="${NODE_RANK}"
    local jobs_per_node="${JOBS_PER_NODE}"
    local gpus_per_job="${GPUS_PER_JOB}"
    local node_gpu_ids="${NODE_GPU_IDS}"

    if (( node_rank < 0 || node_rank >= num_nodes )); then
        echo "[FATAL] NODE_RANK=${node_rank} is out of range [0, $((num_nodes - 1))]." >&2
        exit 1
    fi

    local total_workers=$(( num_nodes * jobs_per_node ))
    local global_worker_id=$(( node_rank * jobs_per_node + local_slot ))
    if (( global_worker_id >= total_workers )); then
        echo "[FATAL] global_worker_id=${global_worker_id} >= total_workers=${total_workers}" >&2
        exit 1
    fi

    local gpu_ids
    gpu_ids=$(gpu_slice_for_slot "$local_slot" "$gpus_per_job" "$node_gpu_ids")

    build_tasks
    filter_tasks_by_allowlist

    echo "[WORKER][START] node_rank=${node_rank} local_slot=${local_slot} global_worker_id=${global_worker_id}/${total_workers} gpu_ids=${gpu_ids}"
    if (( local_slot == 0 )); then
        echo "[WORKER][INFO] allowed tasks: ${TASK_TAGS[*]}"
    fi

    local assigned=0
    local idx
    for idx in "${!TASK_TAGS[@]}"; do
        if (( idx % total_workers != global_worker_id )); then
            continue
        fi
        assigned=1
        run_one_task \
            "${TASK_MODEL_TAGS[$idx]}" \
            "${TASK_MODEL_PATHS[$idx]}" \
            "${TASK_MODES[$idx]}" \
            "${TASK_LAST_FLAGS[$idx]}" \
            "${TASK_TAGS[$idx]}" \
            "$gpu_ids"
    done

    if (( assigned == 0 )); then
        echo "[WORKER][IDLE] No task assigned to worker ${global_worker_id}."
    fi

    echo "[WORKER][DONE] node_rank=${node_rank} local_slot=${local_slot}"
}

parse_worker_slot() {
    local slot=""
    while (($#)); do
        case "$1" in
            --local-slot)
                slot="$2"
                shift 2
                ;;
            *)
                echo "[FATAL] Unknown arg: $1" >&2
                exit 1
                ;;
        esac
    done

    if [[ -z "$slot" ]]; then
        echo "[FATAL] --local-slot is required in worker mode." >&2
        exit 1
    fi
    echo "$slot"
}

main() {
    activate_env
    init_cluster_env
    init_common_env

    if [[ "${1:-}" == "--worker" ]]; then
        shift
        local local_slot
        local_slot=$(parse_worker_slot "$@")
        worker_main "$local_slot"
        return 0
    fi

    local jobs_per_node="${JOBS_PER_NODE}"
    local i
    local pids=()

    echo "[NODE][START] NODE_RANK=${NODE_RANK} NUM_NODES=${NUM_NODES} JOBS_PER_NODE=${jobs_per_node} GPUS_PER_JOB=${GPUS_PER_JOB}"

    for ((i = 0; i < jobs_per_node; i++)); do
        bash "$0" --worker --local-slot "$i" &
        pids+=("$!")
    done

    local rc=0
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            rc=1
        fi
    done

    echo "[NODE][DONE] NODE_RANK=${NODE_RANK} exit_code=${rc}"
    return "$rc"
}

main "$@"
