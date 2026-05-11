#!/usr/bin/env bash
set -uo pipefail

if [[ "${SWEEP_DEBUG:-0}" == "1" ]]; then
    set -x
fi

# 2-node x 8-GPU-per-node sweep for MiniCPM-V-4_5 replay tests.
# Settings:
# - replay modes: image_text / text_image / image_text_text / image_text_image / image_text_image_text / image_image_text
# - last1 template behavior enabled
# - one GPU per task, suitable for standard HF inference path

DEFAULT_NUM_NODES=2
DEFAULT_JOBS_PER_NODE=8
DEFAULT_GPUS_PER_JOB=1
DEFAULT_NODE_GPU_IDS="0,1,2,3,4,5,6,7"

truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

ensure_python_pkg() {
    local module_name="$1"
    local pip_name="${2:-$1}"
    local index_url="${3:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    if python - <<PY >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("${module_name}") else 1)
PY
    then
        return 0
    fi
    echo "[SETUP] missing python package ${module_name}; installing ${pip_name} from ${index_url}"
    python -m pip install -i "${index_url}" "${pip_name}"
}

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
    export JOBS_PER_NODE="${JOBS_PER_NODE:-$DEFAULT_JOBS_PER_NODE}"
    export GPUS_PER_JOB="${GPUS_PER_JOB:-$DEFAULT_GPUS_PER_JOB}"
    export NODE_GPU_IDS="${NODE_GPU_IDS:-$DEFAULT_NODE_GPU_IDS}"
    export NODE_RANK="$(detect_node_rank)"
    if [[ -z "${NODE_RANK:-}" ]]; then
        export NODE_RANK=0
    fi
}

activate_env() {
    export PATH=/usr/local/cuda/bin:$PATH
    local use_vllm="${MINICPM45_USE_VLLM:-1}"
    local activate_script=""
    local conda_target=""
    if truthy "${use_vllm}"; then
        activate_script="${MINICPM45_VLLM_ACTIVATE:-/opt/miniconda3/bin/activate}"
        conda_target="${MINICPM45_VLLM_CONDA_ENV:-/opt/miniconda3/envs/vlmeval_qwen35_vllm}"
    else
        activate_script="${MINICPM45_HF_ACTIVATE:-/opt/miniconda3/bin/activate}"
        conda_target="${CONDA_ENV_NAME:-vlmevalkit}"
    fi
    if [[ -f "${activate_script}" ]]; then
        # shellcheck source=/dev/null
        source "${activate_script}"
    fi
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "${conda_target}"
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

    export REPLAY_TIMES="${REPLAY_TIMES:-1}"
    export REPLAY_DEBUG="${REPLAY_DEBUG:-0}"
    export REPLAY_IMAGE_COPY_MODE="${REPLAY_IMAGE_COPY_MODE:-reuse_path}"
    export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME:-directly_answer}"
    export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT:-1}"
    export MINICPM45_USE_VLLM="${MINICPM45_USE_VLLM:-1}"

    export ANSWER_FORMAT_ENABLE="${ANSWER_FORMAT_ENABLE:-1}"
    export ANSWER_FORMAT_REQUIRE_BOXED="${ANSWER_FORMAT_REQUIRE_BOXED:-0}"
    export ANSWER_FORMAT_RESPONSE_COL="${ANSWER_FORMAT_RESPONSE_COL:-prediction}"
    export ANSWER_FORMAT_FALLBACK_COL="${ANSWER_FORMAT_FALLBACK_COL:-detailed_prediction}"
    export ANSWER_FORMAT_MAX_FAILS="${ANSWER_FORMAT_MAX_FAILS:-50}"

    # MiniCPM-V 4.5 replay I/O debug:
    # - MINICPM_DEBUG_IO=1 enables logging
    # - MINICPM_DEBUG_IO_EVERY=N prints every N samples
    export MINICPM_DEBUG_IO="${MINICPM_DEBUG_IO:-0}"
    export MINICPM_DEBUG_IO_EVERY="${MINICPM_DEBUG_IO_EVERY:-50}"
    export MINICPM_DEBUG_IO_MAX_TEXT_CHARS="${MINICPM_DEBUG_IO_MAX_TEXT_CHARS:-4000}"
    export MINICPM_DEBUG_IO_MAX_OUTPUT_CHARS="${MINICPM_DEBUG_IO_MAX_OUTPUT_CHARS:-4000}"
    export MINICPM_DEBUG_IO_MAX_ITEMS="${MINICPM_DEBUG_IO_MAX_ITEMS:-32}"

    export SAVE_ROOT="${SAVE_ROOT:-/path/to/vlmevalkit}"
    export EXP_DATE_TAG="${EXP_DATE_TAG:-$(date +%Y%m%d)}"
    export EXP_GROUP_TAG="${EXP_GROUP_TAG:-minicpm45_newsets_last1_2node16gpu}"
    export DATALIST="${DATALIST:-AI2D_TEST DynaMath MathVista_MINI OCRBench SEEDBench2_Plus VisuLogic LogicVista VisualPuzzles MathVision}"
    export PYTHONPATH="${SAVE_ROOT}:${PYTHONPATH:-}"
    if truthy "${MINICPM45_USE_VLLM}"; then
        export MINICPM45_VLLM_TP_SIZE="${MINICPM45_VLLM_TP_SIZE:-1}"
        export MINICPM45_VLLM_MAX_MODEL_LEN="${MINICPM45_VLLM_MAX_MODEL_LEN:-8192}"
        export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-4}"
        export MINICPM45_VLLM_MAX_NUM_SEQS="${MINICPM45_VLLM_MAX_NUM_SEQS:-${INFER_BATCH_SIZE}}"
        export MINICPM45_VLLM_MAX_IMAGES="${MINICPM45_VLLM_MAX_IMAGES:-8}"
        export MINICPM45_VLLM_GPU_MEMORY_UTILIZATION="${MINICPM45_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
        export VLMEVAL_USE_MINICPM45_MINIMAL_CONFIG="${VLMEVAL_USE_MINICPM45_MINIMAL_CONFIG:-1}"
        export VLMEVAL_API_MINIMAL_IMPORT="${VLMEVAL_API_MINIMAL_IMPORT:-1}"
        export VLMEVAL_VLM_MINIMAL_IMPORT="${VLMEVAL_VLM_MINIMAL_IMPORT:-1}"
        export VLMEVAL_LAZY_INIT="${VLMEVAL_LAZY_INIT:-1}"
        unset VLMEVAL_USE_QWEN_MINIMAL_CONFIG
    else
        export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"
    fi
}

prepare_runtime_deps() {
    ensure_python_pkg "librosa" "librosa"
}

build_tasks() {
    TASK_MODEL_TAGS=()
    TASK_MODEL_PATHS=()
    TASK_MODES=()
    TASK_LAST_FLAGS=()
    TASK_TAGS=()

    local model_tag="${MODEL_TAG_MINICPM45:-MiniCPM-V-4_5}"
    local model_path="${MODEL_PATH_MINICPM45:-/models/MiniCPM-V-4_5}"
    local -a modes=(
        image_text
        text_image
        image_text_text
        image_text_image
        image_text_image_text
        image_image_text
    )

    if [[ ! -d "${model_path}" && "${model_path}" == /* ]]; then
        echo "[FATAL] model path missing: ${model_path}" >&2
        exit 1
    fi

    local mode
    for mode in "${modes[@]}"; do
        TASK_MODEL_TAGS+=("${model_tag}")
        TASK_MODEL_PATHS+=("${model_path}")
        TASK_MODES+=("$mode")
        TASK_LAST_FLAGS+=("1")
        TASK_TAGS+=("${model_tag}__${mode}__last1")
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

build_task_instances() {
    INST_MODEL_TAGS=()
    INST_MODEL_PATHS=()
    INST_MODES=()
    INST_LAST_FLAGS=()
    INST_TASK_TAGS=()
    INST_DATASETS=()
    INST_JOB_TAGS=()

    local i
    local dataset
    for i in "${!TASK_TAGS[@]}"; do
        for dataset in $DATALIST; do
            INST_MODEL_TAGS+=("${TASK_MODEL_TAGS[$i]}")
            INST_MODEL_PATHS+=("${TASK_MODEL_PATHS[$i]}")
            INST_MODES+=("${TASK_MODES[$i]}")
            INST_LAST_FLAGS+=("${TASK_LAST_FLAGS[$i]}")
            INST_TASK_TAGS+=("${TASK_TAGS[$i]}")
            INST_DATASETS+=("${dataset}")
            INST_JOB_TAGS+=("${TASK_TAGS[$i]}__${dataset}")
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

run_one_task() {
    local model_tag="$1"
    local model_path="$2"
    local mode="$3"
    local last_flag="$4"
    local task_tag="$5"
    local dataset_name="$6"
    local gpu_ids="$7"

    export CUDA_VISIBLE_DEVICES="$gpu_ids"
    export MODEL_PATH="$model_path"
    export REPLAY_MODE="$mode"
    export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="$last_flag"

    local save_prefix="runs/standard/${EXP_DATE_TAG}/${EXP_GROUP_TAG}/${task_tag}"
    work_dir="${SAVE_ROOT}/${save_prefix}/output"
    export MINICPM_DEBUG_IO_DIR="${MINICPM_DEBUG_IO_DIR:-${work_dir}/_logs/minicpm_debug_io/${dataset_name}}"

    echo "[TASK][START] ${task_tag} x ${dataset_name}"
    echo "[TASK][INFO] MODEL_PATH=${MODEL_PATH} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[TASK][INFO] REPLAY_MODE=${REPLAY_MODE} LAST=${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT}"
    echo "[TASK][INFO] INFER_BATCH_SIZE=${INFER_BATCH_SIZE}"
    echo "[TASK][INFO] MINICPM45_USE_VLLM=${MINICPM45_USE_VLLM}"
    echo "[TASK][INFO] MINICPM_DEBUG_IO=${MINICPM_DEBUG_IO} EVERY=${MINICPM_DEBUG_IO_EVERY}"
    echo "[TASK][INFO] MINICPM_DEBUG_IO_DIR=${MINICPM_DEBUG_IO_DIR}"
    echo "[TASK][INFO] work_dir=${work_dir}"
    echo "[TASK][INFO] DATASET=${dataset_name}"

    local model_name="${MODEL_NAME_MINICPM45:-MiniCPM-V-4_5-Replay}"
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SCRIPT_DIR="$script_dir"
    # shellcheck source=/dev/null
    source "${script_dir}/run_standard_guard.sh"
    standard_guard_init

    local expected_count
    expected_count=$(get_expected_count "$dataset_name")
    echo "Expected samples for ${dataset_name}: ${expected_count}"
    if [[ "$expected_count" -lt 0 ]]; then
        echo "[SKIP][DATASET] ${model_tag}/${model_name} x ${dataset_name}: unavailable/build failed."
        return 0
    fi

    if infer_complete "$model_name" "$dataset_name" "$expected_count"; then
        echo "[SKIP][INFER] ${model_tag}/${model_name} x ${dataset_name}: complete."
    else
        if infer_artifacts_exist "$model_name" "$dataset_name"; then
            echo "[CLEAN][INFER+EVAL] ${model_tag}/${model_name} x ${dataset_name}: remove stale artifacts."
            cleanup_all_artifacts "$model_name" "$dataset_name"
        fi
        if ! launch_infer_fg "$model_name" "$dataset_name" "${INFER_BATCH_SIZE}"; then
            echo "[FAIL][INFER] ${model_tag}/${model_name} x ${dataset_name}: skip eval."
            return 0
        fi
    fi

    if infer_complete "$model_name" "$dataset_name" "$expected_count"; then
        run_answer_format_postprocess "$model_name" "$dataset_name" || true
        if eval_complete "$model_name" "$dataset_name" "$expected_count"; then
            echo "[SKIP][EVAL] ${model_tag}/${model_name} x ${dataset_name}: complete."
        else
            if eval_artifacts_exist "$model_name" "$dataset_name"; then
                echo "[CLEAN][EVAL] ${model_tag}/${model_name} x ${dataset_name}: remove stale eval artifacts."
                cleanup_eval_artifacts "$model_name" "$dataset_name"
            fi
            launch_eval_bg "$model_name" "$dataset_name" || true
        fi
    else
        echo "[SKIP][EVAL] ${model_tag}/${model_name} x ${dataset_name}: infer incomplete."
    fi

    echo "[TASK][DONE] ${task_tag} x ${dataset_name}"
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
    build_task_instances

    echo "[WORKER][START] node_rank=${node_rank} local_slot=${local_slot} worker=${global_worker_id}/${total_workers} gpu_ids=${gpu_ids}"
    if (( local_slot == 0 )); then
        echo "[WORKER][INFO] tasks: ${TASK_TAGS[*]}"
        echo "[WORKER][INFO] expanded job count: ${#INST_JOB_TAGS[@]}"
    fi

    local assigned=0
    local fail_count=0
    local idx
    for idx in "${!INST_JOB_TAGS[@]}"; do
        if (( idx % total_workers != global_worker_id )); then
            continue
        fi
        assigned=1
        if ! run_one_task \
            "${INST_MODEL_TAGS[$idx]}" \
            "${INST_MODEL_PATHS[$idx]}" \
            "${INST_MODES[$idx]}" \
            "${INST_LAST_FLAGS[$idx]}" \
            "${INST_TASK_TAGS[$idx]}" \
            "${INST_DATASETS[$idx]}" \
            "$gpu_ids"; then
            fail_count=$((fail_count + 1))
            echo "[WORKER][WARN] task failed but continue: ${INST_JOB_TAGS[$idx]}"
        fi
    done

    if (( assigned == 0 )); then
        echo "[WORKER][IDLE] No task assigned to worker ${global_worker_id}."
    fi
    echo "[WORKER][DONE] node_rank=${node_rank} local_slot=${local_slot} failed_tasks=${fail_count}"
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
    prepare_runtime_deps
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
