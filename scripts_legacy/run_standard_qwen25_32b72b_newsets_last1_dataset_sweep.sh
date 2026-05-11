#!/usr/bin/env bash
set -uo pipefail

if [[ "${SWEEP_DEBUG:-0}" == "1" ]]; then
    set -x
fi

# Dataset-level sweep for Qwen2.5-VL-{32B,72B} replay tests.
# Compared with the original task-level scheduler, this script distributes
# (model_tag, replay_mode, dataset) units across workers so that adding more
# nodes remains useful even when the number of replay modes is small.

DEFAULT_NUM_NODES=2
DEFAULT_JOBS_PER_NODE=1
DEFAULT_GPUS_PER_JOB=8
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
    export NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-$DEFAULT_NUM_NODES}}"
    export JOBS_PER_NODE="${JOBS_PER_NODE:-1}"
    export GPUS_PER_JOB="${GPUS_PER_JOB:-$DEFAULT_GPUS_PER_JOB}"
    export NODE_GPU_IDS="${NODE_GPU_IDS:-$DEFAULT_NODE_GPU_IDS}"
    export NODE_RANK="$(detect_node_rank)"
    if [[ -z "${NODE_RANK:-}" ]]; then
        export NODE_RANK=0
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
        echo "[FATAL] conda not found." >&2
        exit 1
    fi
}

init_common_env() {
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export FORCE_LOCAL=True
    export OLD_VERSION='False'
    export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

    export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
    export OPENAI_API_KEY_JUDGE="${OPENAI_API_KEY_JUDGE:-${OPENAI_API_KEY}}"
    export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
    export OPENAI_API_BASE_JUDGE="${OPENAI_API_BASE_JUDGE:-${OPENAI_API_BASE}}"
    export JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
    export JUDGE_NPROC="${JUDGE_NPROC:-8}"

    export REPLAY_TIMES="${REPLAY_TIMES:-1}"
    export REPLAY_DEBUG="${REPLAY_DEBUG:-0}"
    export REPLAY_IMAGE_COPY_MODE="${REPLAY_IMAGE_COPY_MODE:-reuse_path}"
    export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME:-directly_answer}"
    export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT:-1}"
    export REPLAY_STAGE_DEBUG="${REPLAY_STAGE_DEBUG:-1}"
    export REPLAY_STAGE_DEBUG_SAMPLES="${REPLAY_STAGE_DEBUG_SAMPLES:-8}"
    export REPLAY_PROMPT_AUDIT="${REPLAY_PROMPT_AUDIT:-1}"
    export REPLAY_PROMPT_AUDIT_PRINT="${REPLAY_PROMPT_AUDIT_PRINT:-1}"

    export VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    export VLLM_USE_V1="${VLLM_USE_V1:-0}"
    export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"
    export VLLM_MAX_MODEL_LEN_32B="${VLLM_MAX_MODEL_LEN_32B:-49152}"
    export VLLM_MAX_MODEL_LEN_72B="${VLLM_MAX_MODEL_LEN_72B:-49152}"
    export VLLM_MAX_NUM_SEQS_32B="${VLLM_MAX_NUM_SEQS_32B:-${INFER_BATCH_SIZE}}"
    export VLLM_MAX_NUM_SEQS_72B="${VLLM_MAX_NUM_SEQS_72B:-${INFER_BATCH_SIZE}}"
    export REPLAY_LIMIT_MM_PER_PROMPT_32B="${REPLAY_LIMIT_MM_PER_PROMPT_32B:-2}"
    export REPLAY_LIMIT_MM_PER_PROMPT_72B="${REPLAY_LIMIT_MM_PER_PROMPT_72B:-2}"

    export ANSWER_FORMAT_ENABLE="${ANSWER_FORMAT_ENABLE:-1}"
    export ANSWER_FORMAT_REQUIRE_BOXED="${ANSWER_FORMAT_REQUIRE_BOXED:-0}"
    export ANSWER_FORMAT_RESPONSE_COL="${ANSWER_FORMAT_RESPONSE_COL:-prediction}"
    export ANSWER_FORMAT_FALLBACK_COL="${ANSWER_FORMAT_FALLBACK_COL:-detailed_prediction}"
    export ANSWER_FORMAT_MAX_FAILS="${ANSWER_FORMAT_MAX_FAILS:-50}"

    export SAVE_ROOT="${SAVE_ROOT:-/path/to/vlmevalkit}"
    export EXP_DATE_TAG="${EXP_DATE_TAG:-$(date +%Y%m%d)}"
    export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen25_32b72b_newsets_last1_dataset_sweep}"
    export DATALIST="${DATALIST:-AI2D_TEST DynaMath MathVista_MINI OCRBench SEEDBench2_Plus VisuLogic LogicVista VisualPuzzles MathVision}"
    export INFER_RESUME_ENABLED="${INFER_RESUME_ENABLED:-0}"
}

build_tasks() {
    TASK_MODEL_TAGS=()
    TASK_MODEL_PATHS=()
    TASK_MODES=()
    TASK_LAST_FLAGS=()
    TASK_TAGS=()

    local -a model_tags=(
        Qwen2.5-VL-32B-Instruct
        Qwen2.5-VL-72B-Instruct
        Qwen2.5-VL-72B-Instruct-AWQ
    )
    local -a model_paths=(
        "${MODEL_PATH_QWEN25_32B:-/models/Qwen2.5-VL-32B-Instruct}"
        "${MODEL_PATH_QWEN25_72B:-/models/Qwen2.5-VL-72B-Instruct}"
        "${MODEL_PATH_QWEN25_72B_AWQ:-/models/Qwen2.5-VL-72B-Instruct-AWQ}"
    )
    local -a modes=(
        image_text
        text_image
        image_text_text
        image_text_image
        image_text_image_text
        image_image_text
    )

    local mi
    local mode
    for mi in "${!model_tags[@]}"; do
        if [[ ! -d "${model_paths[$mi]}" ]]; then
            echo "[WARN] model path missing, skip ${model_tags[$mi]}: ${model_paths[$mi]}"
            continue
        fi
        for mode in "${modes[@]}"; do
            TASK_MODEL_TAGS+=("${model_tags[$mi]}")
            TASK_MODEL_PATHS+=("${model_paths[$mi]}")
            TASK_MODES+=("$mode")
            TASK_LAST_FLAGS+=("1")
            TASK_TAGS+=("${model_tags[$mi]}__${mode}__last1")
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
    normalized="${normalized//__none__/__image_text__}"
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

build_dataset_units() {
    UNIT_MODEL_TAGS=()
    UNIT_MODEL_PATHS=()
    UNIT_MODES=()
    UNIT_LAST_FLAGS=()
    UNIT_TASK_TAGS=()
    UNIT_DATASETS=()

    local i
    local dataset
    for i in "${!TASK_TAGS[@]}"; do
        for dataset in $DATALIST; do
            UNIT_MODEL_TAGS+=("${TASK_MODEL_TAGS[$i]}")
            UNIT_MODEL_PATHS+=("${TASK_MODEL_PATHS[$i]}")
            UNIT_MODES+=("${TASK_MODES[$i]}")
            UNIT_LAST_FLAGS+=("${TASK_LAST_FLAGS[$i]}")
            UNIT_TASK_TAGS+=("${TASK_TAGS[$i]}")
            UNIT_DATASETS+=("${dataset}")
        done
    done
}

gpu_slice_for_slot() {
    local slot="$1"
    local gpus_per_job="$2"
    local gpu_csv="$3"
    IFS=',' read -r -a all_gpus <<< "$gpu_csv"
    local need=$(( (slot + 1) * gpus_per_job ))
    if (( ${#all_gpus[@]} < need )); then
        echo "[FATAL] NODE_GPU_IDS has ${#all_gpus[@]} GPUs, slot=${slot} requires ${need}." >&2
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

set_model_runtime_profile() {
    local model_tag="$1"

    export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"
    export VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"

    if [[ "$model_tag" == *"72B"* ]]; then
        export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$VLLM_MAX_MODEL_LEN_72B}"
        export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$VLLM_MAX_NUM_SEQS_72B}"
        export REPLAY_LIMIT_MM_PER_PROMPT="${REPLAY_LIMIT_MM_PER_PROMPT:-$REPLAY_LIMIT_MM_PER_PROMPT_72B}"
    else
        export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$VLLM_MAX_MODEL_LEN_32B}"
        export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$VLLM_MAX_NUM_SEQS_32B}"
        export REPLAY_LIMIT_MM_PER_PROMPT="${REPLAY_LIMIT_MM_PER_PROMPT:-$REPLAY_LIMIT_MM_PER_PROMPT_32B}"
    fi
}

maybe_reuse_source_artifacts() {
    local task_tag="$1"
    local dataset="$2"
    local model_name="$3"

    local reuse_group="${REUSE_FROM_EXP_GROUP_TAG:-}"
    local reuse_date="${REUSE_FROM_EXP_DATE_TAG:-${EXP_DATE_TAG}}"
    local reuse_root="${REUSE_FROM_SAVE_ROOT:-${SAVE_ROOT}}"
    if [[ -z "${reuse_group}" ]]; then
        return 0
    fi

    local target_model_dir="${work_dir}/${model_name}"
    if infer_artifacts_exist "$model_name" "$dataset" || eval_artifacts_exist "$model_name" "$dataset"; then
        return 0
    fi

    local source_work_dir="${reuse_root}/runs/standard/${reuse_date}/${reuse_group}/${task_tag}/output"
    local source_model_dir="${source_work_dir}/${model_name}"
    if [[ ! -d "${source_model_dir}" ]]; then
        echo "[REUSE][MISS] ${task_tag} x ${dataset}: source model dir missing: ${source_model_dir}"
        return 0
    fi

    local src_files=("${source_model_dir}/${model_name}_${dataset}"*)
    if (( ${#src_files[@]} == 0 )); then
        echo "[REUSE][MISS] ${task_tag} x ${dataset}: no source artifacts."
        return 0
    fi

    mkdir -p "${target_model_dir}"
    local copied=0
    local src
    for src in "${src_files[@]}"; do
        if [[ ! -f "${src}" ]]; then
            continue
        fi
        cp -f "${src}" "${target_model_dir}/"
        copied=$((copied + 1))
    done

    if (( copied > 0 )); then
        echo "[REUSE][COPY] ${task_tag} x ${dataset}: copied ${copied} artifact(s) from ${source_model_dir}"
    else
        echo "[REUSE][MISS] ${task_tag} x ${dataset}: no regular files copied."
    fi
}

run_one_dataset_unit() {
    local model_tag="$1"
    local model_path="$2"
    local mode="$3"
    local last_flag="$4"
    local task_tag="$5"
    local dataset="$6"
    local gpu_ids="$7"

    export CUDA_VISIBLE_DEVICES="$gpu_ids"
    export MODEL_PATH="$model_path"
    export REPLAY_MODE="$mode"
    export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="$last_flag"
    set_model_runtime_profile "$model_tag"

    local save_prefix="runs/standard/${EXP_DATE_TAG}/${EXP_GROUP_TAG}/${task_tag}"
    work_dir="${SAVE_ROOT}/${save_prefix}/output"
    export REPLAY_DUMP_DIR="${REPLAY_DUMP_DIR:-${work_dir}/_logs/replay_dump}"
    export REPLAY_DUMP_MAX_CHARS="${REPLAY_DUMP_MAX_CHARS:-20000}"

    echo "[UNIT][START] ${task_tag} x ${dataset}"
    echo "[UNIT][INFO] MODEL_PATH=${MODEL_PATH}"
    echo "[UNIT][INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[UNIT][INFO] REPLAY_MODE=${REPLAY_MODE} LAST=${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT}"
    echo "[UNIT][INFO] VLLM_TP_SIZE=${VLLM_TP_SIZE} VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN} VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
    echo "[UNIT][INFO] REPLAY_LIMIT_MM_PER_PROMPT=${REPLAY_LIMIT_MM_PER_PROMPT} INFER_BATCH_SIZE=${INFER_BATCH_SIZE}"
    echo "[UNIT][INFO] work_dir=${work_dir}"

    local model_name="Qwen2VLChatReplay"
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SCRIPT_DIR="$script_dir"
    # shellcheck source=/dev/null
    source "${script_dir}/run_standard_guard.sh"
    standard_guard_init

    local expected_count
    expected_count=$(get_expected_count "$dataset")
    echo "Expected samples for ${dataset}: ${expected_count}"
    if [[ "$expected_count" -lt 0 ]]; then
        echo "[SKIP][DATASET] ${model_tag}/${model_name} x ${dataset}: unavailable/build failed."
        return 0
    fi

    maybe_reuse_source_artifacts "$task_tag" "$dataset" "$model_name"

    if infer_complete "$model_name" "$dataset" "$expected_count"; then
        echo "[SKIP][INFER] ${model_tag}/${model_name} x ${dataset}: complete."
    else
        if infer_artifacts_exist "$model_name" "$dataset"; then
            if [[ "${INFER_RESUME_ENABLED:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
                echo "[RESUME][INFER] ${model_tag}/${model_name} x ${dataset}: reuse partial artifacts and continue."
            else
                echo "[CLEAN][INFER+EVAL] ${model_tag}/${model_name} x ${dataset}: remove stale artifacts."
                cleanup_all_artifacts "$model_name" "$dataset"
            fi
        fi
        if ! launch_infer_fg "$model_name" "$dataset" "${INFER_BATCH_SIZE}"; then
            echo "[FAIL][INFER] ${model_tag}/${model_name} x ${dataset}: continue to next unit."
            return 1
        fi
    fi

    if infer_complete "$model_name" "$dataset" "$expected_count"; then
        run_answer_format_postprocess "$model_name" "$dataset" || true
        if eval_complete "$model_name" "$dataset" "$expected_count"; then
            echo "[SKIP][EVAL] ${model_tag}/${model_name} x ${dataset}: complete."
        else
            if eval_artifacts_exist "$model_name" "$dataset"; then
                echo "[CLEAN][EVAL] ${model_tag}/${model_name} x ${dataset}: remove stale eval artifacts."
                cleanup_eval_artifacts "$model_name" "$dataset"
            fi
            launch_eval_bg "$model_name" "$dataset" || true
        fi
    else
        echo "[SKIP][EVAL] ${model_tag}/${model_name} x ${dataset}: infer incomplete."
    fi

    echo "[UNIT][DONE] ${task_tag} x ${dataset}"
    return 0
}

worker_main() {
    local local_slot="$1"
    local num_nodes="${NUM_NODES}"
    local node_rank="${NODE_RANK}"
    local jobs_per_node="${JOBS_PER_NODE}"
    local gpus_per_job="${GPUS_PER_JOB}"
    local node_gpu_ids="${NODE_GPU_IDS}"

    local total_workers=$(( num_nodes * jobs_per_node ))
    local global_worker_id=$(( node_rank * jobs_per_node + local_slot ))
    local gpu_ids
    gpu_ids=$(gpu_slice_for_slot "$local_slot" "$gpus_per_job" "$node_gpu_ids")

    build_tasks
    filter_tasks_by_allowlist
    build_dataset_units

    echo "[WORKER][START] node_rank=${node_rank} local_slot=${local_slot} worker=${global_worker_id}/${total_workers} gpu_ids=${gpu_ids}"
    if (( local_slot == 0 )); then
        echo "[WORKER][INFO] task tags: ${TASK_TAGS[*]}"
        echo "[WORKER][INFO] datasets: ${DATALIST}"
        echo "[WORKER][INFO] total_units=${#UNIT_TASK_TAGS[@]}"
    fi

    local assigned=0
    local fail_count=0
    local idx
    for idx in "${!UNIT_TASK_TAGS[@]}"; do
        if (( idx % total_workers != global_worker_id )); then
            continue
        fi
        assigned=1
        if ! run_one_dataset_unit \
            "${UNIT_MODEL_TAGS[$idx]}" \
            "${UNIT_MODEL_PATHS[$idx]}" \
            "${UNIT_MODES[$idx]}" \
            "${UNIT_LAST_FLAGS[$idx]}" \
            "${UNIT_TASK_TAGS[$idx]}" \
            "${UNIT_DATASETS[$idx]}" \
            "$gpu_ids"; then
            fail_count=$((fail_count + 1))
            echo "[WORKER][WARN] unit failed but continue: ${UNIT_TASK_TAGS[$idx]} x ${UNIT_DATASETS[$idx]}"
        fi
    done

    if (( assigned == 0 )); then
        echo "[WORKER][IDLE] No unit assigned to worker ${global_worker_id}."
    fi
    echo "[WORKER][DONE] node_rank=${node_rank} local_slot=${local_slot} failed_units=${fail_count}"
    return 0
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

    echo "[NODE][START] NODE_RANK=${NODE_RANK} NUM_NODES=${NUM_NODES} JOBS_PER_NODE=${JOBS_PER_NODE} GPUS_PER_JOB=${GPUS_PER_JOB}"
    local i
    local pids=()
    for ((i = 0; i < JOBS_PER_NODE; i++)); do
        bash "$0" --worker --local-slot "$i" &
        pids+=("$!")
    done

    local pid
    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done
    echo "[NODE][DONE] NODE_RANK=${NODE_RANK}"
    return 0
}

main "$@"
