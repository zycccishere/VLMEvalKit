#!/usr/bin/env bash
set -euo pipefail

# Run Qwen2.5-VL-7B-Instruct on MathVision with 4 GPUs in parallel (1 GPU/job)
# - 1 node, 4 jobs, 1 GPU per job
# - 5 replay modes => distributed across 4 workers
# - Legacy vllm v0, bsz=1, identity prompt template

export NUM_NODES="${NUM_NODES:-1}"
export JOBS_PER_NODE="${JOBS_PER_NODE:-4}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
export NODE_GPU_IDS="${NODE_GPU_IDS:-0,1,2,3}"

export EXP_GROUP_TAG="${EXP_GROUP_TAG:-repair_qwen25_7b_mathvision_1node4gpu_default_prompt}"
export DATALIST="${DATALIST:-MathVision}"
export TASK_TAG_ALLOWLIST="${TASK_TAG_ALLOWLIST:-Qwen2.5-VL-7B-Instruct__none__last1,Qwen2.5-VL-7B-Instruct__image_text_text__last1,Qwen2.5-VL-7B-Instruct__image_text_image__last1,Qwen2.5-VL-7B-Instruct__image_text_image_text__last1,Qwen2.5-VL-7B-Instruct__image_image_text__last1}"

# Legacy vllm v0 + bsz=1
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"

export REPLAY_PROMPT_TEMPLATE_NAME="identity"
unset REPLAY_PROMPT_TEMPLATE_FILE REPLAY_PROMPT_TEMPLATE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../run_standard_qwen2_qwen25_newsets_last1_2node_8gpu_sweep.sh"
