#!/usr/bin/env bash
# 仅对 ALL__dataset_x_setting_no_reasoning.csv 中缺 eval 的 three_sets 数据点跑 eval（infer 已有）。
# 参照 run_qwen25_72b_3sets_direct_tp4_2node4workers.sh 等，调用上层 run_reeval_* 脚本并限定 TASK_TAG_ALLOWLIST + DATALIST。
#
# 缺失项（按 task 缺的 three_sets 数据集）：
#   - Qwen2.5-VL-72B-Instruct|image_image_text: VisuLogic
#   - Qwen2-VL-7B-Instruct|image_text_image_text: VisualPuzzles
#   - Qwen2.5-VL-7B-Instruct|image_image_text: VisualPuzzles（LogicVista 已有）
#   - Qwen2.5-VL-7B-Instruct|none: VisuLogic、LogicVista（.xlsx 只有 infer，eval 未跑出 _gpt-4o_score.csv）
#   - MiniCPM-V-4_5|image_text_text: VisuLogic
#
# 用法（在 vlmevalkit 根目录，或先 export SAVE_ROOT=/path/to/vlmevalkit）：
#   单机：NUM_NODES=1 JOBS_PER_NODE=1 bash scripts/three_sets_replay_eval/run_reeval_three_sets_missing_only.sh
#   多机：按原 three_sets 脚本设 NODE_RANK/NUM_NODES 后执行同上。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# vlmevalkit 根目录（脚本在 scripts/three_sets_replay_eval/ 下，上两级）
export SAVE_ROOT="${SAVE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SCRIPTS_DIR="${SCRIPT_DIR}/.."
export EXP_DATE_TAG="${EXP_DATE_TAG:-20260307}"

cd "${SAVE_ROOT}" || exit 1

echo "[REEVAL][THREE_SETS] SAVE_ROOT=${SAVE_ROOT} EXP_DATE_TAG=${EXP_DATE_TAG}"

# ---------------------------------------------------------------------------
# 1) Qwen2.5-VL-72B three_sets_direct: 仅 image_image_text x VisuLogic
# ---------------------------------------------------------------------------
echo "[REEVAL][THREE_SETS] --- Qwen2.5-VL-72B direct: image_image_text x VisuLogic ---"
export EXP_GROUP_TAG="three_sets_qwen25_72b_direct_last1_tp4_2node4workers"
export DATALIST="VisuLogic"
export TASK_TAG_ALLOWLIST="Qwen2.5-VL-72B-Instruct__image_image_text__last1"
export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME:-directly_answer}"
export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT:-1}"

bash "${SCRIPTS_DIR}/run_reeval_qwen25_32b72b_sweep.sh" || true

# ---------------------------------------------------------------------------
# 2) Qwen small three_sets_direct: 各 task 缺的不一（none 缺 VisuLogic，其余有缺 VisualPuzzles 等）
#    用三数据集都跑一遍，reeval 内部会跳过已 complete 的，只补缺的
# ---------------------------------------------------------------------------
echo "[REEVAL][THREE_SETS] --- Qwen small direct: VisuLogic / LogicVista / VisualPuzzles（只补未完成的）---"
export EXP_GROUP_TAG="three_sets_qwen_small_direct_last1_2node16workers"
export DATALIST="VisuLogic LogicVista VisualPuzzles"
export TASK_TAG_ALLOWLIST="Qwen2-VL-7B-Instruct__image_text_image_text__last1,Qwen2.5-VL-7B-Instruct__image_image_text__last1,Qwen2.5-VL-7B-Instruct__none__last1"
export REPLAY_PROMPT_TEMPLATE_NAME="${REPLAY_PROMPT_TEMPLATE_NAME:-directly_answer}"
export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT="${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT:-1}"

bash "${SCRIPTS_DIR}/run_reeval_qwen2_qwen25_sweep.sh" || true

# ---------------------------------------------------------------------------
# 2b) 单独补跑 Qwen2.5-VL-7B-Instruct__none__last1 的 LogicVista eval（若上面 sweep 没跑到）
#     infer 有 LogicVista.xlsx，但 eval 需产出 _gpt-4o_score.csv，此处显式跑一次
# ---------------------------------------------------------------------------
WORK_DIR_NONE_7B="${SAVE_ROOT}/runs/standard/${EXP_DATE_TAG}/three_sets_qwen_small_direct_last1_2node16workers/Qwen2.5-VL-7B-Instruct__none__last1/output"
if [[ -d "${WORK_DIR_NONE_7B}" ]] && [[ ! -f "${WORK_DIR_NONE_7B}/Qwen2VLChatReplay/Qwen2VLChatReplay_LogicVista_gpt-4o_score.csv" ]]; then
    echo "[REEVAL][THREE_SETS] --- 补跑 Qwen2.5-VL-7B none x LogicVista ---"
    python run.py --data LogicVista --model Qwen2VLChatReplay --work-dir "${WORK_DIR_NONE_7B}" --mode eval --nproc 8 --verbose --judge gpt-4o || true
fi

# ---------------------------------------------------------------------------
# 3) MiniCPM-V-4_5 three_sets_direct: 仅 image_text_text x VisuLogic（无独立 reeval 脚本，直接调 run.py eval）
# ---------------------------------------------------------------------------
echo "[REEVAL][THREE_SETS] --- MiniCPM-V-4_5 direct: image_text_text x VisuLogic ---"
EXP_GROUP_MINICPM="three_sets_minicpm45_direct_last1_1node8workers"
WORK_DIR="${SAVE_ROOT}/runs/standard/${EXP_DATE_TAG}/${EXP_GROUP_MINICPM}/MiniCPM-V-4_5__image_text_text__last1/output"
MODEL_NAME="MiniCPM-V-4_5-Replay"
DATASET="VisuLogic"

if [[ -d "${WORK_DIR}" ]]; then
    echo "[REEVAL][EVAL] ${MODEL_NAME} x ${DATASET} work_dir=${WORK_DIR}"
    python run.py --data "${DATASET}" --model "${MODEL_NAME}" --work-dir "${WORK_DIR}" --mode eval --nproc 8 --verbose --judge gpt-4o || true
else
    echo "[REEVAL][SKIP] work_dir missing: ${WORK_DIR}"
fi

echo "[REEVAL][THREE_SETS] done."
