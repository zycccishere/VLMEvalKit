#!/usr/bin/env bash
set -uo pipefail

if [[ "${SWEEP_DEBUG:-0}" == "1" ]]; then
    set -x
fi

DEFAULT_NUM_NODES=1
DEFAULT_JOBS_PER_NODE=8
DEFAULT_GPUS_PER_JOB=1
DEFAULT_NODE_GPU_IDS="0,1,2,3,4,5,6,7"

split_list() {
    echo "$1" | tr ',' ' '
}

truthy() {
    case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
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
    local use_vllm="${QWEN35_USE_VLLM:-1}"
    local conda_bin env_target
    if truthy "${use_vllm}"; then
        conda_bin="${QWEN35_VLLM_CONDA_BIN:-/opt/miniconda3/bin/conda}"
        env_target="${QWEN35_VLLM_ENV_PREFIX:-/opt/miniconda3/envs/vlmeval_qwen35_vllm}"
    else
        conda_bin="${CONDA_BIN:-/opt/miniconda3/bin/conda}"
        env_target="${CONDA_ENV_NAME:-vlmevalkit}"
    fi

    if [[ ! -x "${conda_bin}" ]]; then
        echo "[FATAL] conda not found: ${conda_bin}" >&2
        exit 1
    fi

    eval "$("${conda_bin}" shell.bash hook)"
    conda activate "${env_target}"
}

prepend_py_path() {
    local add_path="$1"
    if [[ -z "${PYTHONPATH:-}" ]]; then
        export PYTHONPATH="$add_path"
        return 0
    fi
    case ":${PYTHONPATH}:" in
        *":${add_path}:"*) ;;
        *) export PYTHONPATH="${add_path}:${PYTHONPATH}" ;;
    esac
}

init_common_env() {
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export FORCE_LOCAL=True
    export OLD_VERSION='False'

    export QWEN35_USE_VLLM="${QWEN35_USE_VLLM:-1}"
    if truthy "${QWEN35_USE_VLLM}"; then
        prepend_py_path "/path/to/vlmevalkit"
    else
        prepend_py_path "/path/to/qwen35_transformers_pypi2"
    fi
    export VLMEVAL_USE_QWEN_MINIMAL_CONFIG="${VLMEVAL_USE_QWEN_MINIMAL_CONFIG:-1}"
    export VLMEVAL_API_MINIMAL_IMPORT="${VLMEVAL_API_MINIMAL_IMPORT:-1}"
    export VLMEVAL_LAZY_INIT="${VLMEVAL_LAZY_INIT:-1}"
    export VLMEVAL_VLM_MINIMAL_IMPORT="${VLMEVAL_VLM_MINIMAL_IMPORT:-1}"

    export REPLAY_TIMES="${REPLAY_TIMES:-1}"
    export REPLAY_DEBUG="${REPLAY_DEBUG:-0}"
    export REPLAY_IMAGE_COPY_MODE="${REPLAY_IMAGE_COPY_MODE:-reuse_path}"
    export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT:-1}"
    export REPLAY_LIMIT_MM_PER_PROMPT="${REPLAY_LIMIT_MM_PER_PROMPT:-2}"
    if [[ -z "${QWEN35_VLLM_TP_SIZE:-}" && -n "${VLLM_TP_SIZE:-}" ]]; then
        export QWEN35_VLLM_TP_SIZE="${VLLM_TP_SIZE}"
    fi
    if [[ -z "${QWEN35_VLLM_MAX_MODEL_LEN:-}" && -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
        export QWEN35_VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}"
    fi
    if [[ -z "${QWEN35_VLLM_MAX_NUM_SEQS:-}" && -n "${VLLM_MAX_NUM_SEQS:-}" ]]; then
        export QWEN35_VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}"
    fi
    export REPLAY_SAFE_FALLBACK="${REPLAY_SAFE_FALLBACK:-1}"
    export REPLAY_SAFE_TRUNCATE_CHARS="${REPLAY_SAFE_TRUNCATE_CHARS:-6000}"
    export REPLAY_STAGE_DEBUG="${REPLAY_STAGE_DEBUG:-1}"
    export REPLAY_STAGE_DEBUG_SAMPLES="${REPLAY_STAGE_DEBUG_SAMPLES:-8}"
    export REPLAY_PROMPT_AUDIT="${REPLAY_PROMPT_AUDIT:-1}"
    export REPLAY_PROMPT_AUDIT_PRINT="${REPLAY_PROMPT_AUDIT_PRINT:-1}"
    unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE

    export ANSWER_FORMAT_ENABLE="${ANSWER_FORMAT_ENABLE:-1}"
    export ANSWER_FORMAT_REQUIRE_BOXED="${ANSWER_FORMAT_REQUIRE_BOXED:-0}"
    export ANSWER_FORMAT_RESPONSE_COL="${ANSWER_FORMAT_RESPONSE_COL:-prediction}"
    export ANSWER_FORMAT_FALLBACK_COL="${ANSWER_FORMAT_FALLBACK_COL:-detailed_prediction}"
    export ANSWER_FORMAT_MAX_FAILS="${ANSWER_FORMAT_MAX_FAILS:-50}"

    export EVAL_LAUNCH_MODE="${EVAL_LAUNCH_MODE:-bg}"
    export INFER_RESUME_ENABLED="${INFER_RESUME_ENABLED:-0}"

    export SAVE_ROOT="${SAVE_ROOT:-/path/to/vlmevalkit}"
    export EXP_DATE_TAG="${EXP_DATE_TAG:-$(date +%Y%m%d)}"
    export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen35_newsets_last1_dataset_sweep}"
    export DATALIST="${DATALIST:-AI2D_TEST DynaMath MathVista_MINI OCRBench SEEDBench2_Plus VisuLogic LogicVista VisualPuzzles}"
    export POLICY_LIST="${POLICY_LIST:-directly_answer identity}"
    export REPLAY_MODE_LIST="${REPLAY_MODE_LIST:-image_text,text_image,image_text_text,image_text_image,image_text_image_text,image_image_text}"
    export STRICT_SINGLE_MODEL="${STRICT_SINGLE_MODEL:-1}"
}

model_path_for_tag() {
    case "$1" in
        Qwen3.5-9B-Replay) echo "${MODEL_PATH_QWEN35_9B:-/models/Qwen3.5-9B}" ;;
        Qwen3.5-27B-Replay) echo "${MODEL_PATH_QWEN35_27B:-/models/Qwen3.5-27B}" ;;
        Qwen3.5-35B-A3B-Replay) echo "${MODEL_PATH_QWEN35_35B_A3B:-/models/Qwen3.5-35B-A3B}" ;;
        Qwen3.5-4B-Replay) echo "${MODEL_PATH_QWEN35_4B:-/models/Qwen3.5-4B}" ;;
        *) echo "" ;;
    esac
}

build_tasks() {
    TASK_MODEL_TAGS=()
    TASK_MODEL_PATHS=()
    TASK_POLICIES=()
    TASK_MODES=()
    TASK_LAST_FLAGS=()
    TASK_TAGS=()

    declare -A allow_model_map=()
    local allow_models_raw="${MODEL_TAG_ALLOWLIST:-}"
    local item
    for item in $(split_list "$allow_models_raw"); do
        allow_model_map["$item"]=1
    done

    local -a model_tags=(
        Qwen3.5-9B-Replay
        Qwen3.5-27B-Replay
        Qwen3.5-35B-A3B-Replay
    )
    local model_tag
    local model_path
    local policy
    local mode
    for model_tag in "${model_tags[@]}"; do
        if [[ -n "$allow_models_raw" && -z "${allow_model_map[$model_tag]:-}" ]]; then
            continue
        fi
        model_path="$(model_path_for_tag "$model_tag")"
        if [[ -z "$model_path" || ! -d "$model_path" ]]; then
            echo "[WARN] model path missing, skip ${model_tag}: ${model_path}" >&2
            continue
        fi
        for policy in $(split_list "${POLICY_LIST}"); do
            case "$policy" in
                directly_answer|identity) ;;
                *)
                    echo "[WARN] unsupported policy, skip: ${policy}" >&2
                    continue
                    ;;
            esac
            for mode in $(split_list "${REPLAY_MODE_LIST}"); do
                TASK_MODEL_TAGS+=("${model_tag}")
                TASK_MODEL_PATHS+=("${model_path}")
                TASK_POLICIES+=("${policy}")
                TASK_MODES+=("${mode}")
                TASK_LAST_FLAGS+=("1")
                TASK_TAGS+=("${model_tag}__${policy}__${mode}__last1")
            done
        done
    done
}

ensure_single_model_profile() {
    if ! truthy "${STRICT_SINGLE_MODEL}"; then
        return 0
    fi
    declare -A uniq=()
    local tag
    for tag in "${TASK_MODEL_TAGS[@]}"; do
        uniq["$tag"]=1
    done
    local uniq_count="${#uniq[@]}"
    if (( uniq_count > 1 )); then
        echo "[FATAL] Mixed model sizes in one sweep are not allowed by default." >&2
        echo "[FATAL] Set MODEL_TAG_ALLOWLIST to one of: Qwen3.5-9B-Replay / Qwen3.5-27B-Replay / Qwen3.5-35B-A3B-Replay" >&2
        exit 1
    fi
}

filter_tasks_by_allowlist() {
    local raw="${TASK_TAG_ALLOWLIST:-}"
    if [[ -z "$raw" ]]; then
        return 0
    fi
    local normalized
    normalized="$(split_list "$raw")"
    normalized="${normalized//__none__/__image_text__}"
    declare -A allow_map=()
    local tag
    for tag in $normalized; do
        allow_map["$tag"]=1
    done

    local -a keep_model_tags=()
    local -a keep_model_paths=()
    local -a keep_policies=()
    local -a keep_modes=()
    local -a keep_last_flags=()
    local -a keep_tags=()
    local i
    for i in "${!TASK_TAGS[@]}"; do
        if [[ -n "${allow_map[${TASK_TAGS[$i]}]:-}" ]]; then
            keep_model_tags+=("${TASK_MODEL_TAGS[$i]}")
            keep_model_paths+=("${TASK_MODEL_PATHS[$i]}")
            keep_policies+=("${TASK_POLICIES[$i]}")
            keep_modes+=("${TASK_MODES[$i]}")
            keep_last_flags+=("${TASK_LAST_FLAGS[$i]}")
            keep_tags+=("${TASK_TAGS[$i]}")
        fi
    done
    TASK_MODEL_TAGS=("${keep_model_tags[@]}")
    TASK_MODEL_PATHS=("${keep_model_paths[@]}")
    TASK_POLICIES=("${keep_policies[@]}")
    TASK_MODES=("${keep_modes[@]}")
    TASK_LAST_FLAGS=("${keep_last_flags[@]}")
    TASK_TAGS=("${keep_tags[@]}")
}

build_dataset_units() {
    UNIT_MODEL_TAGS=()
    UNIT_MODEL_PATHS=()
    UNIT_POLICIES=()
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
            UNIT_POLICIES+=("${TASK_POLICIES[$i]}")
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
    unset MODEL_PATH_QWEN35_4B MODEL_PATH_QWEN35_9B MODEL_PATH_QWEN35_27B
    unset QWEN35_VLLM_MAX_NUM_SEQS
    case "$model_tag" in
        Qwen3.5-9B-Replay)
            export MODEL_PATH_QWEN35_9B="$2"
            export INFER_BATCH_SIZE="${INFER_BATCH_SIZE_9B:-2}"
            export QWEN35_VLLM_TP_SIZE="${QWEN35_VLLM_TP_SIZE_9B:-1}"
            ;;
        Qwen3.5-27B-Replay)
            export MODEL_PATH_QWEN35_27B="$2"
            export INFER_BATCH_SIZE="${INFER_BATCH_SIZE_27B:-1}"
            export QWEN35_VLLM_TP_SIZE="${QWEN35_VLLM_TP_SIZE_27B:-2}"
            ;;
        Qwen3.5-35B-A3B-Replay)
            export MODEL_PATH_QWEN35_35B_A3B="$2"
            export INFER_BATCH_SIZE="${INFER_BATCH_SIZE_35B_A3B:-1}"
            export QWEN35_VLLM_TP_SIZE="${QWEN35_VLLM_TP_SIZE_35B_A3B:-2}"
            ;;
        *)
            echo "[FATAL] unsupported model tag: ${model_tag}" >&2
            exit 1
            ;;
    esac
    export QWEN35_VLLM_MAX_NUM_SEQS="${QWEN35_VLLM_MAX_NUM_SEQS:-${INFER_BATCH_SIZE}}"
}

launch_eval_fg_local() {
    local model_name="$1"
    local dataset_name="$2"
    local eval_log="${eval_log_dir}/${model_name}_${dataset_name}_$(date +%Y%m%d%H%M%S).log"
    echo "[START][EVAL-FG] ${model_name} x ${dataset_name}: log=${eval_log}"
    python run.py --data "$dataset_name" --model "$model_name" --work-dir "${work_dir}" --mode eval --nproc 8 --verbose --judge gpt-4o 2>&1 | tee "${eval_log}"
    local rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
        echo "[FAIL][EVAL] ${model_name} x ${dataset_name}: exit_code=${rc} log=${eval_log}"
        return "$rc"
    fi
    echo "[DONE][EVAL] ${model_name} x ${dataset_name}"
    return 0
}

run_one_unit() {
    local model_tag="$1"
    local model_path="$2"
    local policy="$3"
    local mode="$4"
    local last_flag="$5"
    local task_tag="$6"
    local dataset="$7"
    local gpu_ids="$8"

    export CUDA_VISIBLE_DEVICES="$gpu_ids"
    export MODEL_PATH="$model_path"
    export REPLAY_MODE="$mode"
    export REPLAY_PROMPT_TEMPLATE_NAME="$policy"
    export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="$last_flag"
    set_model_runtime_profile "$model_tag" "$model_path"

    local save_prefix="runs/standard/${EXP_DATE_TAG}/${EXP_GROUP_TAG}/${task_tag}"
    work_dir="${SAVE_ROOT}/${save_prefix}/output"
    export REPLAY_DUMP_DIR="${REPLAY_DUMP_DIR:-${work_dir}/_logs/replay_dump}"
    export REPLAY_DUMP_MAX_CHARS="${REPLAY_DUMP_MAX_CHARS:-20000}"

    local model_name="${model_tag}"
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SCRIPT_DIR="/path/to/vlmevalkit/scripts"
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/run_standard_guard.sh"
    standard_guard_init

    echo "[TASK][START] ${task_tag} x ${dataset}"
    echo "[TASK][INFO] MODEL_PATH=${MODEL_PATH} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[TASK][INFO] POLICY=${REPLAY_PROMPT_TEMPLATE_NAME} MODE=${REPLAY_MODE} LAST=${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT}"
    echo "[TASK][INFO] work_dir=${work_dir} INFER_BATCH_SIZE=${INFER_BATCH_SIZE}"

    local expected_count
    expected_count=$(get_expected_count "$dataset")
    echo "Expected samples for ${dataset}: ${expected_count}"
    if [[ "$expected_count" -lt 0 ]]; then
        echo "[SKIP][DATASET] ${model_tag}/${model_name} x ${dataset}: unavailable/build failed."
        return 0
    fi

    if infer_complete "$model_name" "$dataset" "$expected_count"; then
        echo "[SKIP][INFER] ${model_tag}/${model_name} x ${dataset}: complete."
    else
        if infer_artifacts_exist "$model_name" "$dataset"; then
            if truthy "${INFER_RESUME_ENABLED:-1}"; then
                echo "[RESUME][INFER] ${model_tag}/${model_name} x ${dataset}: reuse partial artifacts and continue."
            else
                echo "[CLEAN][INFER+EVAL] ${model_tag}/${model_name} x ${dataset}: remove stale artifacts."
                cleanup_all_artifacts "$model_name" "$dataset"
            fi
        fi
        if ! launch_infer_fg "$model_name" "$dataset" "${INFER_BATCH_SIZE}"; then
            echo "[FAIL][INFER] ${model_tag}/${model_name} x ${dataset}: stop this unit."
            return 0
        fi
    fi

    if ! infer_complete "$model_name" "$dataset" "$expected_count"; then
        echo "[SKIP][EVAL] ${model_tag}/${model_name} x ${dataset}: infer incomplete."
        return 0
    fi

    run_answer_format_postprocess "$model_name" "$dataset" || true
    if eval_complete "$model_name" "$dataset" "$expected_count"; then
        echo "[SKIP][EVAL] ${model_tag}/${model_name} x ${dataset}: complete."
        return 0
    fi
    if eval_artifacts_exist "$model_name" "$dataset"; then
        echo "[CLEAN][EVAL] ${model_tag}/${model_name} x ${dataset}: remove stale eval artifacts."
        cleanup_eval_artifacts "$model_name" "$dataset"
    fi

    case "${EVAL_LAUNCH_MODE}" in
        skip)
            echo "[SKIP][EVAL] ${model_tag}/${model_name} x ${dataset}: EVAL_LAUNCH_MODE=skip"
            ;;
        fg)
            launch_eval_fg_local "$model_name" "$dataset" || true
            ;;
        bg)
            launch_eval_bg "$model_name" "$dataset" || true
            ;;
        *)
            echo "[FATAL] unsupported EVAL_LAUNCH_MODE=${EVAL_LAUNCH_MODE}" >&2
            exit 1
            ;;
    esac

    echo "[TASK][DONE] ${task_tag} x ${dataset}"
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
    ensure_single_model_profile
    build_dataset_units

    echo "[WORKER][START] node_rank=${node_rank} local_slot=${local_slot} worker=${global_worker_id}/${total_workers} gpu_ids=${gpu_ids}"
    if (( local_slot == 0 )); then
        echo "[WORKER][INFO] tasks: ${TASK_TAGS[*]}"
    fi

    local assigned=0
    local idx
    for idx in "${!UNIT_TASK_TAGS[@]}"; do
        if (( idx % total_workers != global_worker_id )); then
            continue
        fi
        assigned=1
        run_one_unit \
            "${UNIT_MODEL_TAGS[$idx]}" \
            "${UNIT_MODEL_PATHS[$idx]}" \
            "${UNIT_POLICIES[$idx]}" \
            "${UNIT_MODES[$idx]}" \
            "${UNIT_LAST_FLAGS[$idx]}" \
            "${UNIT_TASK_TAGS[$idx]}" \
            "${UNIT_DATASETS[$idx]}" \
            "${gpu_ids}" || true
    done

    if (( assigned == 0 )); then
        echo "[WORKER][IDLE] No task assigned to worker ${global_worker_id}."
    fi
    echo "[WORKER][DONE] node_rank=${node_rank} local_slot=${local_slot}"
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
