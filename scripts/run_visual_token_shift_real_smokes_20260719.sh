#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export MODEL_ROOT="${MODEL_ROOT:-/user/zyc1781/models}"
export TOKEN_ROLL_PYTHON="${TOKEN_ROLL_PYTHON:-/root/.venvs/lmms-engine/bin/python}"
export TOKEN_ROLL_PYDEPS="${TOKEN_ROLL_PYDEPS:-/root/.venvs/vlmevalkit-token-roll-pydeps}"
export MINICPM_TOKEN_ROLL_PYDEPS="${MINICPM_TOKEN_ROLL_PYDEPS:-/root/.venvs/minicpmo-token-roll-pydeps}"
export PYTHONPATH="${REPO_ROOT}:${TOKEN_ROLL_PYDEPS}${PYTHONPATH:+:${PYTHONPATH}}"
export OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/runs/visual_token_shift_real_smoke_20260719_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_ROOT}"

run_one() {
  local devices="$1"
  local family="$2"
  local model_dir="$3"
  local name="$4"
  local family_pydeps=""
  if [[ "${family}" == "minicpm" ]]; then
    family_pydeps="${MINICPM_TOKEN_ROLL_PYDEPS}:"
  fi
  PYTHONPATH="${REPO_ROOT}:${family_pydeps}${TOKEN_ROLL_PYDEPS}${PYTHONPATH:+:${PYTHONPATH}}" \
    CUDA_VISIBLE_DEVICES="${devices}" "${TOKEN_ROLL_PYTHON}" \
    scripts/smoke_visual_token_shift_real_model_20260719.py \
    --family "${family}" \
    --model-path "${MODEL_ROOT}/${model_dir}" \
    --dump-dir "${OUT_ROOT}/${name}" \
    > "${OUT_ROOT}/${name}.log" 2>&1
}

run_one "0,1" qwen Qwen2.5-VL-32B-Instruct qwen25vl_32b &
pid_qwen32=$!
run_one "2" qwen Qwen2.5-VL-3B-Instruct qwen25vl_3b &
pid_qwen3=$!
run_one "3" minicpm MiniCPM-o-4_5 minicpm_o_45 &
pid_minicpm=$!

failed=0
for pid in "${pid_qwen32}" "${pid_qwen3}" "${pid_minicpm}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

for summary in "${OUT_ROOT}"/*/smoke_summary.json; do
  "${TOKEN_ROLL_PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], d["all_passed"], d["checks"])' "${summary}"
done
exit "${failed}"
