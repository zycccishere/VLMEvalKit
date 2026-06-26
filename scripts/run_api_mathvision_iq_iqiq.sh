#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${MODE:-smoke}}"
case "${MODE}" in
  smoke|full) ;;
  *)
    echo "[FATAL] MODE must be smoke or full, got: ${MODE}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="${VLMEVAL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-api_mathvision_iq_iqiq_${MODE}_${RUN_STAMP}}"
RESULTS_ROOT="${RESULTS_ROOT:-runs/${RUN_NAME}}"
RUN_DIR="${RUN_DIR:-${VLMEVAL_ROOT}/tmp/${RUN_NAME}}"

BASE_MODEL_CONFIG="${BASE_MODEL_CONFIG:-${VLMEVAL_ROOT}/scripts/configs/models.yaml}"
BASE_MATRIX="${BASE_MATRIX:-${VLMEVAL_ROOT}/scripts/configs/matrix_qwen25vl_all4_reasoning4_new_entry_20260421.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-${RUN_DIR}/models_api_mathvision.yaml}"
MATRIX_CONFIG="${MATRIX_CONFIG:-${RUN_DIR}/matrix.yaml}"

MODELS_CSV="${MODELS_CSV:-gpt-4o-mini,gpt-5-mini,gpt-5-2025-08-07,gpt-5-chat,claude-haiku-4-5-20251001,gemini-2.5-flash-lite,gemini-2.5-flash-nothinking,gemini-2.5-flash-thinking,gemini-3-flash-preview-nothinking,gemini-3.1-flash-lite}"
DATASETS_CSV="${DATASETS_CSV:-MathVision}"
MODES_CSV="${MODES_CSV:-image_text,image_text_image_text}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SCHEDULER="${SCHEDULER:-gpu_pool}"
EVAL_NPROC="${EVAL_NPROC:-8}"
API_INFER_NPROC="${API_INFER_NPROC:-1}"
API_MAX_TOKENS="${API_MAX_TOKENS:-32768}"
API_TIMEOUT="${API_TIMEOUT:-1200}"
API_IMG_SIZE="${API_IMG_SIZE:--1}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_API_BASE="${JUDGE_API_BASE:-${OPENAI_API_BASE_JUDGE:-${OPENAI_API_BASE:-https://api.openai.com/v1}}}"

mkdir -p "${RUN_DIR}"

"${PYTHON}" - "${BASE_MODEL_CONFIG}" "${MODEL_CONFIG}" "${MODELS_CSV}" "${API_INFER_NPROC}" "${API_MAX_TOKENS}" "${API_TIMEOUT}" "${API_IMG_SIZE}" <<'PY'
from pathlib import Path
import sys
import yaml

base_path, out_path, models_csv, api_infer_nproc, api_max_tokens, api_timeout, api_img_size = sys.argv[1:8]
cfg = yaml.safe_load(Path(base_path).read_text(encoding="utf-8"))
models = [x.strip() for x in models_csv.replace(",", " ").split() if x.strip()]

for name in models:
    cfg["models"][name] = {
        "display_name": name,
        "registry_name": name,
        "env_profile": "main_vlmeval",
        "model_path": f"api://{name}",
        "runtime": {
            "gpus_per_job": 1,
            "infer_batch_size": 1,
            "max_num_seqs": 1,
            "tp_size": 1,
            "max_model_len": None,
            "estimated_dataset_cost": 1.0,
        },
        "task_env": {
            "VLMEVAL_USE_API_REPLAY_MINIMAL_CONFIG": "1",
            "VLMEVAL_API_MINIMAL_IMPORT": "1",
            "VLMEVAL_VLM_MINIMAL_IMPORT": "1",
            "VLMEVAL_LAZY_INIT": "1",
            "VLMEVAL_INFER_NPROC": str(api_infer_nproc),
            "VLMEVAL_API_MAX_TOKENS": str(api_max_tokens),
            "VLMEVAL_API_TIMEOUT": str(api_timeout),
            "VLMEVAL_API_IMG_SIZE": str(api_img_size),
            "VLMEVAL_API_USAGE_LOG_DEFAULT": "1",
        },
    }

out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_suffix(out.suffix + ".tmp")
tmp.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
tmp.replace(out)
PY

if [[ "${MODE}" == "smoke" ]]; then
  SMOKE_ALLOWLIST="${RUN_DIR}/mathvision_first8.txt"
  printf "1\n2\n3\n4\n5\n6\n7\n8\n" > "${SMOKE_ALLOWLIST}"
else
  SMOKE_ALLOWLIST=""
fi

"${PYTHON}" - "${BASE_MATRIX}" "${MATRIX_CONFIG}" "${RUN_NAME}" "${RESULTS_ROOT}" \
  "${MODELS_CSV}" "${DATASETS_CSV}" "${MODES_CSV}" "${GPU_IDS}" "${EVAL_NPROC}" "${SMOKE_ALLOWLIST}" \
  "${JUDGE_MODEL}" "${JUDGE_API_BASE}" <<'PY'
from pathlib import Path
import sys
import yaml

(
    base_matrix,
    out_matrix,
    run_name,
    results_root,
    models_csv,
    datasets_csv,
    modes_csv,
    gpu_ids,
    eval_nproc,
    smoke_allowlist,
    judge_model,
    judge_api_base,
) = sys.argv[1:13]

def split(raw):
    return [x.strip() for x in raw.replace(",", " ").split() if x.strip()]

matrix = yaml.safe_load(Path(base_matrix).read_text(encoding="utf-8"))
matrix["name"] = run_name
matrix["results_root"] = results_root
matrix["node_gpu_ids"] = gpu_ids
matrix["models"] = split(models_csv)
matrix["datasets"] = split(datasets_csv)
matrix["replay_modes"] = split(modes_csv)
matrix["image_transforms"] = ["baseline"]
matrix["policies"] = {"default": {"replay_prompt_template_name": "identity"}}
matrix["resume_infer_default"] = False
matrix["evaluation"] = {
    "launch_mode": "fg",
    "nproc": int(eval_nproc),
    "judge": judge_model,
    "openai_api_key": "",
    "openai_api_base": judge_api_base,
}
matrix["replay"]["template_on_last_replay_text"] = 1
matrix["replay"]["image_copy_mode"] = "reuse_path"
matrix["trace"]["samples"] = 1
matrix["trace"]["prompt_audit_print"] = 0
if smoke_allowlist:
    matrix["dataset_index_allowlists"] = {"MathVision": smoke_allowlist}

out = Path(out_matrix)
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_suffix(out.suffix + ".tmp")
tmp.write_text(yaml.safe_dump(matrix, sort_keys=False, allow_unicode=True), encoding="utf-8")
tmp.replace(out)
PY

echo "[LAUNCH] mode=${MODE}"
echo "[LAUNCH] run_name=${RUN_NAME}"
echo "[LAUNCH] results_root=${RESULTS_ROOT}"
echo "[LAUNCH] models=${MODELS_CSV}"
echo "[LAUNCH] datasets=${DATASETS_CSV} modes=${MODES_CSV}"
echo "[LAUNCH] gpu_slots=${GPU_IDS} scheduler=${SCHEDULER}"
echo "[LAUNCH] api_infer_nproc=${API_INFER_NPROC}"
echo "[LAUNCH] api_max_tokens=${API_MAX_TOKENS}"
echo "[LAUNCH] api_timeout=${API_TIMEOUT}"
echo "[LAUNCH] api_img_size=${API_IMG_SIZE}"
echo "[LAUNCH] judge=${JUDGE_MODEL} judge_api_base=${JUDGE_API_BASE}"
echo "[LAUNCH] model_config=${MODEL_CONFIG}"
echo "[LAUNCH] matrix_config=${MATRIX_CONFIG}"
echo "[LAUNCH] resume_infer=${RESUME_INFER:-0}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[LAUNCH] DRY_RUN=1, configs generated only."
  exit 0
fi

extra_args=()
if [[ "${RESUME_INFER:-0}" == "1" ]]; then
  extra_args+=(--resume-infer)
fi

cd "${VLMEVAL_ROOT}"
exec bash scripts/run_benchmark.sh \
  --matrix-config "${MATRIX_CONFIG}" \
  --model-config "${MODEL_CONFIG}" \
  --nodes 1 \
  --node-rank 0 \
  --gpu-ids "${GPU_IDS}" \
  --models "${MODELS_CSV}" \
  --datasets "${DATASETS_CSV}" \
  --modes "${MODES_CSV}" \
  --transforms baseline \
  --policies default \
  --scheduler "${SCHEDULER}" \
  "${extra_args[@]}"
