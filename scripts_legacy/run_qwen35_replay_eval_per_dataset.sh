#!/usr/bin/env bash
set -euo pipefail

truthy() {
    case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

activate_qwen35_env() {
    export PATH=/usr/local/cuda/bin:$PATH
    export QWEN35_USE_VLLM="${QWEN35_USE_VLLM:-1}"

    local conda_bin env_target
    if truthy "${QWEN35_USE_VLLM}"; then
        conda_bin="${QWEN35_VLLM_CONDA_BIN:-/opt/miniconda3/bin/conda}"
        env_target="${QWEN35_VLLM_ENV_PREFIX:-/opt/miniconda3/envs/vlmeval_qwen35_vllm}"
    else
        conda_bin="${CONDA_BIN:-/opt/miniconda3/bin/conda}"
        env_target="${CONDA_ENV_NAME:-vlmevalkit}"
    fi

    if [[ ! -x "${conda_bin}" ]]; then
        echo "[FATAL] conda binary not found: ${conda_bin}" >&2
        exit 1
    fi

    eval "$("${conda_bin}" shell.bash hook)"
    conda activate "${env_target}"
}

activate_qwen35_env

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FORCE_LOCAL=True
export OLD_VERSION='False'
export VLMEVAL_USE_QWEN_MINIMAL_CONFIG="${VLMEVAL_USE_QWEN_MINIMAL_CONFIG:-1}"
export VLMEVAL_API_MINIMAL_IMPORT="${VLMEVAL_API_MINIMAL_IMPORT:-1}"
export VLMEVAL_LAZY_INIT="${VLMEVAL_LAZY_INIT:-1}"
export VLMEVAL_VLM_MINIMAL_IMPORT="${VLMEVAL_VLM_MINIMAL_IMPORT:-1}"
if truthy "${QWEN35_USE_VLLM}"; then
    export PYTHONPATH="/path/to/vlmevalkit:${PYTHONPATH:-}"
else
    export PYTHONPATH="/path/to/qwen35_transformers_pypi2:${PYTHONPATH:-}"
fi

MODELNAME="${1:-Qwen35VLChatReplay}"
DATALIST="${2:-MathVistaSample}"
SAVE_PREFIX="${3:-qwen35_smoke}"
export MODEL_PATH="${4:?MODEL_PATH required}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"

export REPLAY_MODE="${REPLAY_MODE:-none}"
export REPLAY_TIMES="${REPLAY_TIMES:-1}"
export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME:-directly_answer}"
export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT:-1}"
if [[ -z "${QWEN35_VLLM_TP_SIZE:-}" && -n "${VLLM_TP_SIZE:-}" ]]; then
    export QWEN35_VLLM_TP_SIZE="${VLLM_TP_SIZE}"
fi
if [[ -z "${QWEN35_VLLM_MAX_MODEL_LEN:-}" && -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
    export QWEN35_VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}"
fi
if [[ -z "${QWEN35_VLLM_MAX_NUM_SEQS:-}" && -n "${VLLM_MAX_NUM_SEQS:-}" ]]; then
    export QWEN35_VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}"
fi
if [[ -z "${QWEN35_VLLM_TP_SIZE:-}" ]]; then
    case "${MODELNAME}" in
        *27B*) export QWEN35_VLLM_TP_SIZE=2 ;;
        *) export QWEN35_VLLM_TP_SIZE=1 ;;
    esac
fi
export QWEN35_VLLM_MAX_MODEL_LEN="${QWEN35_VLLM_MAX_MODEL_LEN:-32768}"
export QWEN35_VLLM_MAX_NUM_SEQS="${QWEN35_VLLM_MAX_NUM_SEQS:-${INFER_BATCH_SIZE}}"
export REPLAY_LIMIT_MM_PER_PROMPT="${REPLAY_LIMIT_MM_PER_PROMPT:-2}"

save_root="${SAVE_ROOT:-/path/to/vlmevalkit}"
work_dir="$save_root/$SAVE_PREFIX/output"

echo "work directory of $MODELNAME: $work_dir"
echo "MODEL_PATH=$MODEL_PATH"
echo "REPLAY_MODE=$REPLAY_MODE TEMPLATE=$REPLAY_PROMPT_TEMPLATE_NAME LAST=$REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT"
echo "TP=$QWEN35_VLLM_TP_SIZE MAX_MODEL_LEN=$QWEN35_VLLM_MAX_MODEL_LEN MAX_NUM_SEQS=$QWEN35_VLLM_MAX_NUM_SEQS"
echo "MINIMAL_CONFIG=$VLMEVAL_USE_QWEN_MINIMAL_CONFIG API_MINIMAL=$VLMEVAL_API_MINIMAL_IMPORT VLM_MINIMAL=$VLMEVAL_VLM_MINIMAL_IMPORT QWEN35_USE_VLLM=$QWEN35_USE_VLLM"

for DATASET in $DATALIST; do
    echo "Starting inference with model $MODELNAME on dataset $DATASET"
    python run.py --data "$DATASET" --model "$MODELNAME" --work-dir "$work_dir" --mode infer --verbose --batch-size "${INFER_BATCH_SIZE}"

    echo "Starting evaluation with model $MODELNAME on dataset $DATASET"
    python run.py --data "$DATASET" --model "$MODELNAME" --work-dir "$work_dir" --nproc 16 --verbose --judge gpt-4o
done
