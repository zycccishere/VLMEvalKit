#!/usr/bin/env bash
set -euo pipefail

# Non-inference prompt inspection for the three repaired datasets.
# Dumps JSON payloads showing the final multimodal content order and text blocks
# right before model-side serialization / tokenization.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export LMUData="${LMUData:-$HOME/LMUData}"
export OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/prompt_checks/$(date +%Y%m%d)/three_sets_prompt_preview}"

mkdir -p "${OUT_DIR}/direct" "${OUT_DIR}/default"

cd "${REPO_ROOT}"
if [[ -f /opt/miniconda3/bin/activate ]]; then
    # shellcheck source=/dev/null
    source /opt/miniconda3/bin/activate
fi
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME:-vlmevalkit}"
fi

python3 "${SCRIPT_DIR}/inspect_prompt_build_three_sets.py" \
    --setting direct \
    --rows 0 1 \
    --print-full-content \
    --output-dir "${OUT_DIR}/direct" \
    > /dev/null

python3 "${SCRIPT_DIR}/inspect_prompt_build_three_sets.py" \
    --setting default \
    --rows 0 1 \
    --print-full-content \
    --output-dir "${OUT_DIR}/default" \
    > /dev/null

python3 - <<'PY' "${OUT_DIR}/direct" "${OUT_DIR}/direct_summary.json" "${OUT_DIR}/default" "${OUT_DIR}/default_summary.json"
import json
import os
import sys

def build_summary(src_dir: str, out_file: str) -> None:
    rows = []
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(src_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

build_summary(sys.argv[1], sys.argv[2])
build_summary(sys.argv[3], sys.argv[4])
PY

echo "Prompt previews written to:"
echo "  ${OUT_DIR}/direct"
echo "  ${OUT_DIR}/default"
