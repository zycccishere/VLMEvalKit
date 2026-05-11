#!/usr/bin/env bash
set -euo pipefail

# Replay8 benchmark for stronger API models with replay settings.
# Models (exact proxy model names):
#   - gemini-3-flash-preview-nothinking
#   - claude-3-sonnet-20240229
#   - claude-sonnet-4-20250514
#   - claude-sonnet-4-5-20250929
#   - gpt-4o
#   - gpt-5
#   - gpt-5.2
#   - grok-4-fast

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export LMUData="${LMUData:-/path/to/vlmevalkit/exp_debug/replay_8subsets_v1}"
export EXP_DATE_TAG="${EXP_DATE_TAG:-$(date +%Y%m%d)}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-api_strong_replay8}"
export SAVE_ROOT="${SAVE_ROOT:-/path/to/vlmevalkit}"

export DATALIST="${DATALIST:-ReplayIconA_L2R ReplayIconA_R2L ReplayIconB_L2R ReplayIconB_R2L ReplayShapeA_L2R ReplayShapeA_R2L ReplayShapeB_L2R ReplayShapeB_R2L}"
export REPLAY_MODES="${REPLAY_MODES:-none image_text_text image_text_image image_text_image_text image_image_text}"
export REPLAY_TIMES="${REPLAY_TIMES:-1}"
export REPLAY_IMAGE_COPY_MODE="${REPLAY_IMAGE_COPY_MODE:-reuse_path}"
export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME:-directly_answer}"
export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT:-1}"
export NPROC="${NPROC:-8}"

# Unified proxy settings for all models (OpenAI-compatible endpoint).
# Priority: explicit OPENAI_API_* > OPENAI_API_*_JUDGE (to align eval/infer) > legacy defaults.
if [[ -z "${OPENAI_API_KEY:-}" && -n "${OPENAI_API_KEY_JUDGE:-}" ]]; then
  export OPENAI_API_KEY="${OPENAI_API_KEY_JUDGE}"
fi
if [[ -z "${OPENAI_API_BASE:-}" && -n "${OPENAI_API_BASE_JUDGE:-}" ]]; then
  export OPENAI_API_BASE="${OPENAI_API_BASE_JUDGE}"
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1/chat/completions}"

MODELS=(
  "${MODELS:-gpt-4o gpt-5 gpt-5.2 grok-4-fast gemini-3-flash-preview-nothinking claude-3-sonnet-20240229 claude-sonnet-4-20250514}"
)

echo "[INFO] LMUData=${LMUData}"
echo "[INFO] EXP_GROUP_TAG=${EXP_GROUP_TAG}"
echo "[INFO] DATALIST=${DATALIST}"
echo "[INFO] REPLAY_MODES=${REPLAY_MODES}"
echo "[INFO] NPROC=${NPROC}"
echo "[INFO] OPENAI_API_BASE=${OPENAI_API_BASE}"
echo "[INFO] OPENAI_API_KEY_LEN=${#OPENAI_API_KEY}"

for model in ${MODELS[*]}; do
  for mode in ${REPLAY_MODES}; do
    export REPLAY_MODE="${mode}"
    export TOKEN_USAGE_LOG_FILE="${SAVE_ROOT}/runs/standard/${EXP_DATE_TAG}/${EXP_GROUP_TAG}/${model}__${mode}/token_usage.jsonl"
    work_dir="${SAVE_ROOT}/runs/standard/${EXP_DATE_TAG}/${EXP_GROUP_TAG}/${model}__${mode}/output"
    mkdir -p "${work_dir}"

    echo "[TASK][START] model=${model} replay_mode=${mode}"
    for dataset in ${DATALIST}; do
      echo "[DATASET][START] ${model} ${mode} x ${dataset}"
      python run.py --data "${dataset}" --model "${model}" --work-dir "${work_dir}" --verbose --nproc "${NPROC}"
      echo "[DATASET][DONE] ${model} ${mode} x ${dataset}"
    done
    echo "[TASK][DONE] model=${model} replay_mode=${mode}"
  done
done

echo "[ALL DONE] replay8 API strong-model sweep finished."
