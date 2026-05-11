#!/usr/bin/env bash
set -euo pipefail

# Repair selected missing default-prompt results on a single node.
#
# Targets:
# - 72B image_text_image_text / OCRBench
# - 3B image_text_image / MathVista_MINI
# - 3B image_text_image_text / MathVision
# - 3B image_text_image_text / MathVista_MINI
#
# Single-node topology:
# - Stage 1 parallel:
#   - GPUs 0,1,2,3 -> 72B TP=4 repair
#   - GPU 4 -> 3B image_text_image / MathVista_MINI
#   - GPU 5 -> 3B image_text_image_text / MathVista_MINI
# - Stage 2:
#   - GPU 4 -> 3B image_text_image_text / MathVision
#
# Important:
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPAIR_LMU_ROOT="${REPAIR_LMU_ROOT:-${SCRIPT_DIR}/../../.repair_lmu_data}"

prepare_local_tsvs() {
    mkdir -p "${REPAIR_LMU_ROOT}"

    cp -f "/datasets/vlmeval/OCRBench.tsv" \
        "${REPAIR_LMU_ROOT}/OCRBench.tsv"
    cp -f "/datasets/vlmeval/MathVista_MINI.tsv" \
        "${REPAIR_LMU_ROOT}/MathVista_MINI.tsv"
    cp -f "/datasets/vlmeval/MathVision.tsv" \
        "${REPAIR_LMU_ROOT}/MathVision.tsv"
}

run_72b_ocrbench_repair() {
    (
        export NUM_NODES=1
        export JOBS_PER_NODE=1
        export GPUS_PER_JOB=4
        export NODE_GPU_IDS="${REPAIR_GPU_IDS_72B:-0,1,2,3}"

        export EXP_DATE_TAG="${EXP_DATE_TAG_72B:-20260306}"
        export EXP_GROUP_TAG="${EXP_GROUP_TAG_72B:-qwen25_72b_newsets_last1_6node12workers_tp4_default_prompt}"
        export DATALIST="OCRBench"
        export LMUData="${REPAIR_LMU_ROOT}"

        export MODEL_PATH_QWEN25_72B="${MODEL_PATH_QWEN25_72B:-/models/Qwen2.5-VL-72B-Instruct}"
        export TASK_TAG_ALLOWLIST="Qwen2.5-VL-72B-Instruct__image_text_image_text__last1"

        unset VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS REPLAY_LIMIT_MM_PER_PROMPT INFER_BATCH_SIZE
        export VLLM_TP_SIZE=4
        export VLLM_MAX_MODEL_LEN_72B="${VLLM_MAX_MODEL_LEN_72B:-32768}"
        export VLLM_MAX_NUM_SEQS_72B="${VLLM_MAX_NUM_SEQS_72B:-1}"
        export REPLAY_LIMIT_MM_PER_PROMPT_72B="${REPLAY_LIMIT_MM_PER_PROMPT_72B:-2}"
        export INFER_BATCH_SIZE="${INFER_BATCH_SIZE_72B:-1}"

        export REPLAY_PROMPT_TEMPLATE_NAME="identity"
        unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE
        export INFER_RESUME_ENABLED=0

        exec bash "${SCRIPT_DIR}/../run_standard_qwen25_32b72b_newsets_last1_dataset_sweep.sh"
    )
}

run_3b_repair() {
    local gpu_id="$1"
    local dataset="$2"
    local task_tag="$3"

    (
        export NUM_NODES=1
        export JOBS_PER_NODE=1
        export GPUS_PER_JOB=1
        export NODE_GPU_IDS="${gpu_id}"

        export EXP_DATE_TAG="${EXP_DATE_TAG_3B:-20260306}"
        export EXP_GROUP_TAG="${EXP_GROUP_TAG_3B:-qwen2_qwen25_small_newsets_last1_2node16tasks_default_prompt}"
        export DATALIST="${dataset}"
        export LMUData="${REPAIR_LMU_ROOT}"

        export MODEL_PATH_QWEN25_3B="${MODEL_PATH_QWEN25_3B:-/models/Qwen2.5-VL-3B-Instruct}"
        export TASK_TAG_ALLOWLIST="${task_tag}"

        export REPLAY_PROMPT_TEMPLATE_NAME="identity"
        unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE
        export VLLM_TP_SIZE=1
        export INFER_RESUME_ENABLED=0

        exec bash "${SCRIPT_DIR}/../run_standard_qwen2_qwen25_newsets_last1_2node_8gpu_sweep.sh"
    )
}

pids=()

prepare_local_tsvs

run_72b_ocrbench_repair &
pids+=("$!")

run_3b_repair 4 MathVista_MINI "Qwen2.5-VL-3B-Instruct__image_text_image__last1" &
pids+=("$!")

run_3b_repair 5 MathVista_MINI "Qwen2.5-VL-3B-Instruct__image_text_image_text__last1" &
pids+=("$!")

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done

run_3b_repair 4 MathVision "Qwen2.5-VL-3B-Instruct__image_text_image_text__last1" || rc=1

exit "$rc"
