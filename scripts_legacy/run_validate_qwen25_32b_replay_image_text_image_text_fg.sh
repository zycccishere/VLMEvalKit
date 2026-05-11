#!/usr/bin/env bash
set -euo pipefail

# Foreground smoke test for Qwen2.5-VL-32B under replay mode:
# image_text_image_text + last1.
# Default: infer only (no API judge needed).

export PATH=/usr/local/cuda/bin:$PATH
if [[ -f /opt/miniconda3/bin/activate ]]; then
    # shellcheck source=/dev/null
    source /opt/miniconda3/bin/activate
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME:-vlmevalkit}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FORCE_LOCAL=True
export OLD_VERSION='False'

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-VL-32B-Instruct}"

export REPLAY_MODE="${REPLAY_MODE:-image_text_image_text}"
export REPLAY_TIMES="${REPLAY_TIMES:-1}"
export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT:-1}"
export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME:-directly_answer}"
export REPLAY_LIMIT_MM_PER_PROMPT="${REPLAY_LIMIT_MM_PER_PROMPT:-2}"
export REPLAY_IMAGE_COPY_MODE="${REPLAY_IMAGE_COPY_MODE:-reuse_path}"

export VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"

MODEL_NAME="${MODEL_NAME:-Qwen2VLChatReplay}"
DATASET_NAME="${DATASET_NAME:-MathVision}"
RUN_MODE="${RUN_MODE:-infer}"

SAVE_ROOT="${SAVE_ROOT:-/path/to/vlmevalkit}"
DATE_TAG="${EXP_DATE_TAG:-$(date +%Y%m%d)}"
GROUP_TAG="${EXP_GROUP_TAG:-qwen25_32b_fg_smoke}"
SETTING_TAG="Qwen2.5-VL-32B-Instruct__image_text_image_text__last1"
WORK_DIR="${SAVE_ROOT}/runs/standard/${DATE_TAG}/${GROUP_TAG}/${SETTING_TAG}/output"

echo "[FG-TEST][START] model=Qwen2.5-VL-32B-Instruct mode=${RUN_MODE} dataset=${DATASET_NAME}"
echo "[FG-TEST][INFO] MODEL_PATH=${MODEL_PATH}"
echo "[FG-TEST][INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} TP=${VLLM_TP_SIZE}"
echo "[FG-TEST][INFO] REPLAY_MODE=${REPLAY_MODE} LAST=${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT}"
echo "[FG-TEST][INFO] VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN} VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
echo "[FG-TEST][INFO] work_dir=${WORK_DIR}"

mkdir -p "${WORK_DIR}"
python run.py \
    --data "${DATASET_NAME}" \
    --model "${MODEL_NAME}" \
    --work-dir "${WORK_DIR}" \
    --mode "${RUN_MODE}" \
    --verbose \
    --batch-size 1

echo "[FG-TEST][DONE] Qwen2.5-VL-32B-Instruct smoke test finished."
