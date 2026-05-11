#!/usr/bin/env bash
set -euo pipefail

if [[ "${SWEEP_DEBUG:-0}" == "1" ]]; then
    set -x
fi

# Multi-node + multi-job-per-node sweep runner for Qwen2-VL replay settings.
#
# Default target: 10 settings = 5 replay modes (including none) x last_text_flag {0,1}
# With default NUM_NODES=5 and JOBS_PER_NODE=2, all 10 settings can run concurrently.
#
# Usage (run on EACH node):
#   bash scripts/run_standard_qwen2_5node_8gpu_sweep.sh
# If scheduler rank env exists (SLURM/torchrun/mpi), NODE_RANK is auto-detected.
# You can still override any default by exporting env vars.
#
# Common envs:
#   NUM_NODES=5
#   NODE_RANK=0
#   JOBS_PER_NODE=2
#   GPUS_PER_JOB=4
#   NODE_GPU_IDS=0,1,2,3,4,5,6,7
#   DATALIST="SEEDBench2_Plus MathVista_MINI MMStar AI2D_TEST MMVet OCRBench MMMU_DEV_VAL MathVision DynaMath ObjHal MMHal"
#   EXP_GROUP_TAG=qwen2_sweep10
#   REPLAY_PROMPT_TEMPLATE_NAME=directly_answer

DEFAULT_NUM_NODES=5
DEFAULT_JOBS_PER_NODE=2
DEFAULT_GPUS_PER_JOB=4
DEFAULT_NODE_GPU_IDS="0,1,2,3,4,5,6,7"
DEFAULT_DATASET_PARALLEL_PER_SETTING=4

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

    # If no launcher provides rank info, multiple nodes may all become rank 0.
    if [[ "$has_explicit_rank" == "0" && -z "${SLURM_NODEID:-}" && -z "${RANK:-}" && -z "${OMPI_COMM_WORLD_RANK:-}" && -z "${PMI_RANK:-}" ]]; then
        if [[ "${NUM_NODES}" != "1" ]]; then
            echo "[WARN] Rank env not detected; using NODE_RANK=${NODE_RANK}. If not using SLURM/torchrun/mpi, set NODE_RANK manually per node." >&2
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

    export MODEL_PATH_QWEN2="${MODEL_PATH_QWEN2:-/models/Qwen2-VL-7B-Instruct}"
    export REPLAY_TIMES="${REPLAY_TIMES:-1}"
    export REPLAY_DEBUG="${REPLAY_DEBUG:-0}"
    export REPLAY_LIMIT_MM_PER_PROMPT="${REPLAY_LIMIT_MM_PER_PROMPT:-1}"
    export REPLAY_IMAGE_COPY_MODE="${REPLAY_IMAGE_COPY_MODE:-reuse_path}"
    # Default to no tensor parallel; use dataset-parallel workers instead.
    export VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    export DATASET_PARALLEL_PER_SETTING="${DATASET_PARALLEL_PER_SETTING:-$DEFAULT_DATASET_PARALLEL_PER_SETTING}"

    # Direct-answer mode defaults.
    export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME:-directly_answer}"
    # Prompt/replay diagnostics.
    export REPLAY_STAGE_DEBUG="${REPLAY_STAGE_DEBUG:-1}"
    export REPLAY_STAGE_DEBUG_SAMPLES="${REPLAY_STAGE_DEBUG_SAMPLES:-8}"
    export REPLAY_PROMPT_AUDIT="${REPLAY_PROMPT_AUDIT:-1}"
    export REPLAY_PROMPT_AUDIT_PRINT="${REPLAY_PROMPT_AUDIT_PRINT:-1}"

    # Postprocess defaults.
    export ANSWER_FORMAT_ENABLE="${ANSWER_FORMAT_ENABLE:-1}"
    export ANSWER_FORMAT_REQUIRE_BOXED="${ANSWER_FORMAT_REQUIRE_BOXED:-0}"
    export ANSWER_FORMAT_RESPONSE_COL="${ANSWER_FORMAT_RESPONSE_COL:-prediction}"
    export ANSWER_FORMAT_FALLBACK_COL="${ANSWER_FORMAT_FALLBACK_COL:-detailed_prediction}"
    export ANSWER_FORMAT_MAX_FAILS="${ANSWER_FORMAT_MAX_FAILS:-50}"

    export SAVE_ROOT="${SAVE_ROOT:-/path/to/vlmevalkit}"
    export EXP_DATE_TAG="${EXP_DATE_TAG:-$(date +%Y%m%d)}"
    export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen2_sweep10}"

    DEFAULT_DATALIST="SEEDBench2_Plus MathVista_MINI MMStar AI2D_TEST MMVet OCRBench MathVision DynaMath"
    export DATALIST="${DATALIST:-$DEFAULT_DATALIST}"

    export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-32}"
}

build_settings() {
    SETTING_MODES=()
    SETTING_LAST_FLAGS=()
    SETTING_TAGS=()

    local modes=(
        none
        image_text_text
        image_text_image
        image_text_image_text
        image_image_text
    )

    local mode
    local last
    for mode in "${modes[@]}"; do
        for last in 0 1; do
            SETTING_MODES+=("$mode")
            SETTING_LAST_FLAGS+=("$last")
            SETTING_TAGS+=("${mode}__last${last}")
        done
    done
}

filter_settings_by_allowlist() {
    local raw="${SETTING_TAG_ALLOWLIST:-}"
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

    local -a keep_modes=()
    local -a keep_last_flags=()
    local -a keep_tags=()

    local i
    for i in "${!SETTING_TAGS[@]}"; do
        if [[ -n "${allow_map[${SETTING_TAGS[$i]}]:-}" ]]; then
            keep_modes+=("${SETTING_MODES[$i]}")
            keep_last_flags+=("${SETTING_LAST_FLAGS[$i]}")
            keep_tags+=("${SETTING_TAGS[$i]}")
        fi
    done

    SETTING_MODES=("${keep_modes[@]}")
    SETTING_LAST_FLAGS=("${keep_last_flags[@]}")
    SETTING_TAGS=("${keep_tags[@]}")
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

run_one_setting() {
    local mode="$1"
    local last_flag="$2"
    local setting_tag="$3"
    local gpu_ids="$4"

    export CUDA_VISIBLE_DEVICES="$gpu_ids"
    export REPLAY_MODE="$mode"
    export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="$last_flag"

    local save_prefix="runs/standard/${EXP_DATE_TAG}/${EXP_GROUP_TAG}/${setting_tag}"
    work_dir="${SAVE_ROOT}/${save_prefix}/output"
    export REPLAY_DUMP_DIR="${REPLAY_DUMP_DIR:-${work_dir}/_logs/replay_dump}"
    export REPLAY_DUMP_MAX_CHARS="${REPLAY_DUMP_MAX_CHARS:-20000}"

    echo "[SETTING][START] ${setting_tag}"
    echo "[SETTING][INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[SETTING][INFO] VLLM_TP_SIZE=${VLLM_TP_SIZE}"
    echo "[SETTING][INFO] DATASET_PARALLEL_PER_SETTING=${DATASET_PARALLEL_PER_SETTING}"
    echo "[SETTING][INFO] work_dir=${work_dir}"
    echo "[SETTING][INFO] DATALIST=${DATALIST}"
    echo "[SETTING][INFO] REPLAY_PROMPT_TEMPLATE_NAME=${REPLAY_PROMPT_TEMPLATE_NAME}"
    echo "[SETTING][INFO] REPLAY_PROMPT_TEMPLATE_FILE=${REPLAY_PROMPT_TEMPLATE_FILE:-<none>} REPLAY_PROMPT_TEMPLATE_SET=$([[ -n "${REPLAY_PROMPT_TEMPLATE:-}" ]] && echo 1 || echo 0)"
    echo "[SETTING][INFO] REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT=${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT}"
    echo "[SETTING][INFO] REPLAY_STAGE_DEBUG=${REPLAY_STAGE_DEBUG} REPLAY_STAGE_DEBUG_SAMPLES=${REPLAY_STAGE_DEBUG_SAMPLES}"
    echo "[SETTING][INFO] REPLAY_PROMPT_AUDIT=${REPLAY_PROMPT_AUDIT} REPLAY_PROMPT_AUDIT_PRINT=${REPLAY_PROMPT_AUDIT_PRINT}"
    echo "[SETTING][INFO] REPLAY_DUMP_DIR=${REPLAY_DUMP_DIR} REPLAY_DUMP_MAX_CHARS=${REPLAY_DUMP_MAX_CHARS}"

    if [[ "${SWEEP_DRY_RUN:-0}" == "1" ]]; then
        echo "[SETTING][DRY-RUN] skip real infer/eval for ${setting_tag}"
        return 0
    fi

    local model_name="Qwen2-VL-7B-Instruct-Replay"
    local -a gpu_arr=()
    IFS=',' read -r -a gpu_arr <<< "$gpu_ids"

    local -a datasets=()
    local ds
    for ds in $DATALIST; do
        datasets+=("$ds")
    done

    local ngpu=${#gpu_arr[@]}
    local nds=${#datasets[@]}
    local nshard="${DATASET_PARALLEL_PER_SETTING:-1}"
    if (( nshard < 1 )); then
        nshard=1
    fi
    if (( nshard > ngpu )); then
        nshard=$ngpu
    fi
    if (( nshard > nds )); then
        nshard=$nds
    fi

    run_dataset_slice() {
        local slice_gpu="$1"
        shift
        local -a slice_datasets=("$@")
        export CUDA_VISIBLE_DEVICES="$slice_gpu"

        local script_dir
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        # Backward-compatibility for sourced helpers expecting SCRIPT_DIR.
        SCRIPT_DIR="$script_dir"
        # shellcheck source=/dev/null
        source "${script_dir}/run_standard_guard.sh"
        standard_guard_init

        local dataset
        for dataset in "${slice_datasets[@]}"; do
            local expected_count
            expected_count=$(get_expected_count "$dataset")
            echo "Expected samples for ${dataset}: ${expected_count}"
            if [ "$expected_count" -lt 0 ]; then
                echo "[SKIP][DATASET] ${model_name} x ${dataset}: dataset unavailable or build failed."
                continue
            fi

            if infer_complete "$model_name" "$dataset" "$expected_count"; then
                echo "[SKIP][INFER] ${model_name} x ${dataset}: infer result is complete."
            else
                if infer_artifacts_exist "$model_name" "$dataset"; then
                    echo "[CLEAN][INFER+EVAL] ${model_name} x ${dataset}: infer incomplete, remove stale artifacts."
                    cleanup_all_artifacts "$model_name" "$dataset"
                fi
                if ! launch_infer_fg "$model_name" "$dataset" "$INFER_BATCH_SIZE"; then
                    echo "[SKIP][EVAL] ${model_name} x ${dataset}: infer failed."
                    continue
                fi
            fi

            if infer_complete "$model_name" "$dataset" "$expected_count"; then
                if [[ -n "${HAL_EVAL_DATASET:-}" && "${dataset}" == "${HAL_EVAL_DATASET}" ]]; then
                    launch_hal_eval_after_infer "$model_name" "$dataset"
                else
                    run_answer_format_postprocess "$model_name" "$dataset"
                    if eval_complete "$model_name" "$dataset" "$expected_count"; then
                        echo "[SKIP][EVAL] ${model_name} x ${dataset}: eval result is complete."
                    else
                        if eval_artifacts_exist "$model_name" "$dataset"; then
                            echo "[CLEAN][EVAL] ${model_name} x ${dataset}: eval incomplete, remove stale eval artifacts."
                            cleanup_eval_artifacts "$model_name" "$dataset"
                        fi
                        launch_eval_bg "$model_name" "$dataset"
                    fi
                fi
            else
                echo "[SKIP][EVAL] ${model_name} x ${dataset}: infer is still incomplete."
            fi
        done
    }

    if (( nshard <= 1 )); then
        run_dataset_slice "${gpu_arr[0]}" "${datasets[@]}"
    else
        local -a pids=()
        local s
        for ((s = 0; s < nshard; s++)); do
            local -a shard_ds=()
            local idx
            for idx in "${!datasets[@]}"; do
                if (( idx % nshard == s )); then
                    shard_ds+=("${datasets[$idx]}")
                fi
            done
            if (( ${#shard_ds[@]} == 0 )); then
                continue
            fi
            echo "[SETTING][SHARD] ${setting_tag} shard=${s}/${nshard} gpu=${gpu_arr[$s]} datasets=${shard_ds[*]}"
            run_dataset_slice "${gpu_arr[$s]}" "${shard_ds[@]}" &
            pids+=("$!")
        done

        local rc=0
        local pid
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                rc=1
            fi
        done
        if (( rc != 0 )); then
            echo "[SETTING][FAIL] ${setting_tag}: one or more dataset shards failed."
            return 1
        fi
    fi

    echo "[SETTING][DONE] ${setting_tag}"
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

    build_settings
    filter_settings_by_allowlist

    echo "[WORKER][START] node_rank=${node_rank} local_slot=${local_slot} global_worker_id=${global_worker_id}/${total_workers} gpu_ids=${gpu_ids}"
    if (( local_slot == 0 )); then
        echo "[WORKER][INFO] allowed settings: ${SETTING_TAGS[*]}"
    fi

    local assigned=0
    local idx
    for idx in "${!SETTING_MODES[@]}"; do
        if (( idx % total_workers != global_worker_id )); then
            continue
        fi
        assigned=1
        run_one_setting \
            "${SETTING_MODES[$idx]}" \
            "${SETTING_LAST_FLAGS[$idx]}" \
            "${SETTING_TAGS[$idx]}" \
            "$gpu_ids"
    done

    if (( assigned == 0 )); then
        echo "[WORKER][IDLE] No setting assigned to worker ${global_worker_id}."
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
