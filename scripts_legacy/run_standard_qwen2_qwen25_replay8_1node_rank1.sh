#!/usr/bin/env bash
set -euo pipefail

# Replay-8 subsets entry for node rank 1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LMUData="${LMUData:-/path/to/vlmevalkit/exp_debug/replay_8subsets_v1}"
export DATALIST="${DATALIST:-ReplayIconA_L2R ReplayIconA_R2L ReplayIconB_L2R ReplayIconB_R2L ReplayShapeA_L2R ReplayShapeA_R2L ReplayShapeB_L2R ReplayShapeB_R2L}"
export EXP_GROUP_TAG="${EXP_GROUP_TAG:-qwen2_qwen25_minicpm_replay8_2node16gpu}"
export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-Qwen2-VL-7B-Instruct__none__last1,Qwen2-VL-7B-Instruct__image_text_text__last1,Qwen2-VL-7B-Instruct__image_text_image__last1,Qwen2-VL-7B-Instruct__image_text_image_text__last1,Qwen2-VL-7B-Instruct__image_image_text__last1,Qwen2.5-VL-7B-Instruct__none__last1,Qwen2.5-VL-7B-Instruct__image_text_text__last1,Qwen2.5-VL-7B-Instruct__image_text_image__last1,Qwen2.5-VL-7B-Instruct__image_text_image_text__last1,Qwen2.5-VL-7B-Instruct__image_image_text__last1}"
export FORCE_GPT_JUDGE_ALL="${FORCE_GPT_JUDGE_ALL:-1}"

exec bash "${SCRIPT_DIR}/run_standard_qwen2_qwen25_minicpm_last1_dynamath_mathvision_1node_rank1.sh"
