#!/usr/bin/env bash

# Common guard helpers for standard run scripts.
# Requires bash and a working python environment with vlmeval importable.

STANDARD_GUARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

standard_guard_init() {
    shopt -s nullglob
    declare -gA DATASET_EXPECTED_COUNTS
    log_root_dir="${work_dir}/_logs"
    infer_log_dir="${log_root_dir}/infer"
    eval_log_dir="${log_root_dir}/eval"
    answer_format_log_dir="${log_root_dir}/answer_format"
    mkdir -p "${log_root_dir}"
    mkdir -p "${infer_log_dir}"
    mkdir -p "${eval_log_dir}"
    mkdir -p "${answer_format_log_dir}"
}

get_expected_count() {
    local dataset_name="$1"
    if [ -n "${DATASET_EXPECTED_COUNTS[$dataset_name]+x}" ]; then
        echo "${DATASET_EXPECTED_COUNTS[$dataset_name]}"
        return 0
    fi
    local raw
raw=$(python - "$dataset_name" <<'PY'
import io
import sys
import contextlib

name = sys.argv[1]

buf = io.StringIO()
dataset = None
err = None
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    try:
        from vlmeval.dataset import build_dataset
        dataset = build_dataset(name)
    except Exception as e:
        err = e

if dataset is None or err is not None:
    captured = buf.getvalue().strip()
    if captured:
        print(captured, file=sys.stderr)
    if err is not None:
        print(f"[get_expected_count] build_dataset({name}) failed: {err}", file=sys.stderr)
    print(-1)
    raise SystemExit(0)

try:
    n = len(dataset)
except Exception:
    data = getattr(dataset, 'data', None)
    n = len(data) if data is not None else -1
print(int(n))
PY
)
    local count
    count=$(printf '%s\n' "$raw" | awk '/^-?[0-9]+$/{v=$0} END{if(v=="") print "-1"; else print v}')
    DATASET_EXPECTED_COUNTS[$dataset_name]="$count"
    echo "$count"
}

infer_file_path() {
    local model_name="$1"
    local dataset_name="$2"
    local model_dir="${work_dir}/${model_name}"
    local xlsx_file="${model_dir}/${model_name}_${dataset_name}.xlsx"
    local tsv_file="${model_dir}/${model_name}_${dataset_name}.tsv"
    if [ -f "$xlsx_file" ]; then
        echo "$xlsx_file"
        return 0
    fi
    if [ -f "$tsv_file" ]; then
        echo "$tsv_file"
        return 0
    fi
    echo ""
}

count_table_rows() {
    local file_path="$1"
    python - "$file_path" <<'PY'
import sys
import csv

path = sys.argv[1]
if not path:
    print(-1)
    raise SystemExit(0)

if path.endswith('.tsv'):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    print(max(len(rows) - 1, 0))
    raise SystemExit(0)

if path.endswith('.xlsx'):
    try:
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        print(max(ws.max_row - 1, 0))
    except Exception:
        print(-1)
    raise SystemExit(0)

print(-1)
PY
}

infer_artifacts_exist() {
    local model_name="$1"
    local dataset_name="$2"
    local model_dir="${work_dir}/${model_name}"
    local all_files=("${model_dir}/${model_name}_${dataset_name}"*)
    [ ${#all_files[@]} -gt 0 ]
}

eval_artifacts_exist() {
    local model_name="$1"
    local dataset_name="$2"
    local model_dir="${work_dir}/${model_name}"
    local all_files=("${model_dir}/${model_name}_${dataset_name}"*)
    local f
    for f in "${all_files[@]}"; do
        case "$f" in
            *.xlsx|*.tsv) ;;
            *_answer_format_report.json|*_answer_format_failures.jsonl) ;;
            *) return 0 ;;
        esac
    done
    return 1
}

infer_complete() {
    local model_name="$1"
    local dataset_name="$2"
    local expected="$3"
    if [ "$expected" -lt 0 ]; then
        return 1
    fi
    local infer_file
    infer_file=$(infer_file_path "$model_name" "$dataset_name")
    if [ -z "$infer_file" ]; then
        return 1
    fi
    local ok
    ok=$(python - "$infer_file" "$expected" <<'PY'
import sys
import math
import pandas as pd

path = sys.argv[1]
expected = int(sys.argv[2])

def is_blank(v):
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    return str(v).strip() == ""

try:
    if path.endswith(".tsv"):
        data = pd.read_csv(path, sep="\t")
    elif path.endswith(".xlsx"):
        data = pd.read_excel(path)
    else:
        print(0)
        raise SystemExit(0)
except Exception:
    print(0)
    raise SystemExit(0)

if len(data) != expected:
    print(0)
    raise SystemExit(0)

cols = [c for c in ["prediction", "description", "detailed_prediction"] if c in data.columns]
if not cols:
    print(1)
    raise SystemExit(0)

for _, row in data.iterrows():
    values = [row[c] for c in cols]
    if all(is_blank(v) for v in values):
        print(0)
        raise SystemExit(0)
    desc = str(row["description"]) if "description" in data.columns and not is_blank(row["description"]) else ""
    if desc.startswith("[FAILED_INFER]") or "Failed to obtain answer via API." in desc:
        print(0)
        raise SystemExit(0)

print(1)
PY
)
    [ "$ok" = "1" ]
}

eval_complete() {
    local model_name="$1"
    local dataset_name="$2"
    local expected="$3"
    local model_dir="${work_dir}/${model_name}"
    local all_files=("${model_dir}/${model_name}_${dataset_name}"*)
    local eval_files=()
    local f
    for f in "${all_files[@]}"; do
        case "$f" in
            *.xlsx|*.tsv) ;;
            *_answer_format_report.json|*_answer_format_failures.jsonl) ;;
            *) eval_files+=("$f") ;;
        esac
    done

    if [ ${#eval_files[@]} -eq 0 ]; then
        return 1
    fi

    local has_sample_pkl=0
    local has_summary_file=0
    for f in "${eval_files[@]}"; do
        case "$f" in
            *_result.pkl)
                has_sample_pkl=1
                local ok
                ok=$(python - "$f" "$expected" <<'PY'
import sys
import pickle

path = sys.argv[1]
expected = int(sys.argv[2])
try:
    with open(path, 'rb') as fp:
        obj = pickle.load(fp)
    n = len(obj) if hasattr(obj, '__len__') else -1
    print(1 if n == expected else 0)
except Exception:
    print(0)
PY
)
                [ "$ok" = "1" ] || return 1
                ;;
            *.csv)
                local ok_csv
                ok_csv=$(python - "$f" <<'PY'
import sys
import csv

path = sys.argv[1]
try:
    with open(path, 'r', encoding='utf-8', newline='') as fp:
        rows = list(csv.reader(fp))
    if len(rows) < 2:
        print(0)
    else:
        flat = ','.join([','.join(r) for r in rows]).lower()
        print(1 if 'overall' in flat else 0)
except Exception:
    print(0)
PY
)
                [ "$ok_csv" = "1" ] || return 1
                has_summary_file=1
                ;;
            *.json)
                if [ -s "$f" ]; then
                    has_summary_file=1
                else
                    return 1
                fi
                ;;
        esac
    done

    # If per-sample pkl exists, it already passed strict count check.
    # Otherwise require at least one non-empty summary artifact.
    if [ "$has_sample_pkl" -eq 1 ]; then
        return 0
    fi
    [ "$has_summary_file" -eq 1 ]
}

cleanup_all_artifacts() {
    local model_name="$1"
    local dataset_name="$2"
    local model_dir="${work_dir}/${model_name}"
    rm -f "${model_dir}/${model_name}_${dataset_name}"*
}

cleanup_eval_artifacts() {
    local model_name="$1"
    local dataset_name="$2"
    local model_dir="${work_dir}/${model_name}"
    local all_files=("${model_dir}/${model_name}_${dataset_name}"*)
    local f
    for f in "${all_files[@]}"; do
        case "$f" in
            *.xlsx|*.tsv|*_answer_format_report.json|*_answer_format_failures.jsonl) ;;
            *) rm -f "$f" ;;
        esac
    done
}

launch_eval_bg() {
    local model_name="$1"
    local dataset_name="$2"
    local eval_log="${eval_log_dir}/${model_name}_${dataset_name}_$(date +%Y%m%d%H%M%S).log"
    local judge_model="${JUDGE_MODEL:-gpt-4o-mini}"
    local judge_nproc="${JUDGE_NPROC:-8}"
    echo "Starting evaluation in background with model $model_name on dataset $dataset_name"
    echo "[START][EVAL-BG] judge=${judge_model} nproc=${judge_nproc}"
    python run.py --data "$dataset_name" --model "$model_name" --work-dir "${work_dir}" --mode eval --nproc "${judge_nproc}" --verbose --judge "${judge_model}" > "${eval_log}" 2>&1 &
    echo "[START][EVAL-BG] $model_name x $dataset_name: pid=$! log=${eval_log}"
}

launch_infer_fg() {
    local model_name="$1"
    local dataset_name="$2"
    local batch_size="$3"
    local infer_log="${infer_log_dir}/${model_name}_${dataset_name}_$(date +%Y%m%d%H%M%S).log"
    echo "Starting inference with model $model_name on dataset $dataset_name"
    echo "[START][INFER-FG] $model_name x $dataset_name: log=${infer_log}"
    python run.py --data "$dataset_name" --model "$model_name" --work-dir "${work_dir}" --mode infer --verbose --batch-size "$batch_size" 2>&1 | tee "${infer_log}"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "[FAIL][INFER] $model_name x $dataset_name: exit_code=$rc log=${infer_log}"
        return "$rc"
    fi
    echo "[DONE][INFER] $model_name x $dataset_name"
    return 0
}

run_answer_format_postprocess() {
    local model_name="$1"
    local dataset_name="$2"
    local enabled="${ANSWER_FORMAT_ENABLE:-1}"
    case "$(echo "$enabled" | tr '[:upper:]' '[:lower:]')" in
        0|false|no|off)
            return 0
            ;;
    esac

    local infer_file
    infer_file=$(infer_file_path "$model_name" "$dataset_name")
    if [ -z "$infer_file" ] || [ ! -f "$infer_file" ]; then
        echo "[SKIP][FORMAT] $model_name x $dataset_name: infer file missing."
        return 0
    fi

    local model_dir="${work_dir}/${model_name}"
    local report_json="${model_dir}/${model_name}_${dataset_name}_answer_format_report.json"
    local fail_jsonl="${model_dir}/${model_name}_${dataset_name}_answer_format_failures.jsonl"
    local fmt_log="${answer_format_log_dir}/${model_name}_${dataset_name}_$(date +%Y%m%d%H%M%S).log"

    local require_boxed="${ANSWER_FORMAT_REQUIRE_BOXED:-0}"
    local response_col="${ANSWER_FORMAT_RESPONSE_COL:-prediction}"
    local fallback_col="${ANSWER_FORMAT_FALLBACK_COL:-detailed_prediction}"
    local max_fails="${ANSWER_FORMAT_MAX_FAILS:-50}"

    echo "[START][FORMAT] $model_name x $dataset_name: infer_file=$infer_file report=$report_json"
    python "${STANDARD_GUARD_DIR}/postprocess_answer_format.py" \
        --pred-file "$infer_file" \
        --out-json "$report_json" \
        --out-fail-jsonl "$fail_jsonl" \
        --response-col "$response_col" \
        --fallback-col "$fallback_col" \
        --require-boxed "$require_boxed" \
        --max-fails "$max_fails" > "$fmt_log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[FAIL][FORMAT] $model_name x $dataset_name: exit_code=$rc log=$fmt_log"
        return "$rc"
    fi
    echo "[DONE][FORMAT] $model_name x $dataset_name: report=$report_json log=$fmt_log"
    return 0
}

# Run RLHF-V MMHal or ObjHal eval after inference (convert pred to answer jsonl + GPT review).
# Requires: work_dir, model_name, dataset_name (MMHal or ObjHal), OPENAI_API_KEY; for ObjHal also COCO2014_ANNOTATIONS.
launch_hal_eval_after_infer() {
    local model_name="$1"
    local dataset_name="$2"
    local hal_log="${answer_format_log_dir:-${work_dir}/_logs/answer_format}/${model_name}_${dataset_name}_hal_$(date +%Y%m%d%H%M%S).log"
    local run_in_bg="${HAL_EVAL_RUN_IN_BG:-1}"
    if [[ "${run_in_bg}" == "1" || "${run_in_bg}" == "true" || "${run_in_bg}" == "TRUE" ]]; then
        echo "[START][HAL-EVAL-BG] $model_name x $dataset_name -> run_hal_eval_after_infer.py"
        python "${STANDARD_GUARD_DIR}/run_hal_eval_after_infer.py" \
            --work-dir "${work_dir}" \
            --model-name "${model_name}" \
            --dataset "${dataset_name}" \
            ${OPENAI_API_KEY:+--openai-key "${OPENAI_API_KEY}"} \
            ${COCO2014_ANNOTATIONS:+--coco-annotations "${COCO2014_ANNOTATIONS}"} \
            >> "${hal_log}" 2>&1 &
        echo "[START][HAL-EVAL-BG] $model_name x $dataset_name: pid=$! log=${hal_log}"
        return 0
    fi

    echo "[START][HAL-EVAL] $model_name x $dataset_name -> run_hal_eval_after_infer.py"
    python "${STANDARD_GUARD_DIR}/run_hal_eval_after_infer.py" \
        --work-dir "${work_dir}" \
        --model-name "${model_name}" \
        --dataset "${dataset_name}" \
        ${OPENAI_API_KEY:+--openai-key "${OPENAI_API_KEY}"} \
        ${COCO2014_ANNOTATIONS:+--coco-annotations "${COCO2014_ANNOTATIONS}"} \
        >> "${hal_log}" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[FAIL][HAL-EVAL] $model_name x $dataset_name: exit_code=$rc log=${hal_log}"
        return "$rc"
    fi
    echo "[DONE][HAL-EVAL] $model_name x $dataset_name: log=${hal_log}"
    return 0
}
