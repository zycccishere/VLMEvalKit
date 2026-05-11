set -x

export PATH=/usr/local/cuda/bin:$PATH
eval "$(conda shell.bash hook)"
source /opt/miniconda3/bin/activate
conda activate vlmevalkit_s2_baseline

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1
export timestamp=`date +"%Y%m%d%H%M%S"`
export OLD_VERSION='False'
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export FORCE_LOCAL=True

MODEL_QWEN25="Qwen2.5-VL-7B-Instruct-Replay"
MODEL_QWEN2="Qwen2-VL-7B-Instruct-Replay"

DEFAULT_DATALIST="SEEDBench2_Plus MathVista_MINI MMStar AI2D_TEST MMVet OCRBench MMMU_DEV_VAL MathVision DynaMath ObjHal MMHal"
DATALIST=${DATALIST:-$DEFAULT_DATALIST}
SAVE_PREFIX_INPUT=$1
export MODEL_PATH_QWEN25=$2
export MODEL_PATH_QWEN2=$3

export REPLAY_MODE=none
export REPLAY_TIMES=1
export REPLAY_DEBUG=${REPLAY_DEBUG:-0}
export REPLAY_LIMIT_MM_PER_PROMPT=${REPLAY_LIMIT_MM_PER_PROMPT:-1}
export REPLAY_IMAGE_COPY_MODE=${REPLAY_IMAGE_COPY_MODE:-reuse_path}

# Answer-format postprocess defaults (no extra export needed before running).
export ANSWER_FORMAT_ENABLE=${ANSWER_FORMAT_ENABLE:-1}
export ANSWER_FORMAT_REQUIRE_BOXED=${ANSWER_FORMAT_REQUIRE_BOXED:-0}
export ANSWER_FORMAT_RESPONSE_COL=${ANSWER_FORMAT_RESPONSE_COL:-prediction}
export ANSWER_FORMAT_FALLBACK_COL=${ANSWER_FORMAT_FALLBACK_COL:-detailed_prediction}
export ANSWER_FORMAT_MAX_FAILS=${ANSWER_FORMAT_MAX_FAILS:-50}
# Prompt template control for directly-answer experiments.
# Built-ins: identity | directly_answer
export REPLAY_PROMPT_TEMPLATE_NAME=${REPLAY_PROMPT_TEMPLATE_NAME:-directly_answer}
# Optional: when replay is enabled, apply template only to the last replayed text.
export REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT=${REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT:-0}
# Optional overrides:
#   export REPLAY_PROMPT_TEMPLATE_FILE=/path/to/template.txt
#   export REPLAY_PROMPT_TEMPLATE='...{problem}...'

save_root="${SAVE_ROOT:-/path/to/vlmevalkit}"
date_tag="${EXP_DATE_TAG:-$(date +%Y%m%d)}"
group_tag="${EXP_GROUP_TAG:-default}"
if [ -n "$SAVE_PREFIX_INPUT" ]; then
    SAVE_PREFIX="$SAVE_PREFIX_INPUT"
else
    # Organized default: runs/standard/<date>/<group>/<replay_mode>/output
    SAVE_PREFIX="runs/standard/${date_tag}/${group_tag}/${REPLAY_MODE}"
fi
work_dir="$save_root/$SAVE_PREFIX/output"
echo "save prefix: $SAVE_PREFIX"
echo "work directory: $work_dir"
echo "models: $MODEL_QWEN25 and $MODEL_QWEN2"
echo "datasets: $DATALIST"
echo "Replay settings: mode=$REPLAY_MODE times=$REPLAY_TIMES debug=$REPLAY_DEBUG limit_mm=$REPLAY_LIMIT_MM_PER_PROMPT"
echo "Prompt template: name=$REPLAY_PROMPT_TEMPLATE_NAME file=${REPLAY_PROMPT_TEMPLATE_FILE:-<none>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_standard_guard.sh"
standard_guard_init

for DATASET in $DATALIST;
do
    for MODELNAME in $MODEL_QWEN25 $MODEL_QWEN2;
    do
        expected_count=$(get_expected_count "$DATASET")
        echo "Expected samples for $DATASET: $expected_count"
        if [ "$expected_count" -lt 0 ]; then
            echo "[SKIP][DATASET] $MODELNAME x $DATASET: dataset unavailable or build failed."
            continue
        fi

        if infer_complete "$MODELNAME" "$DATASET" "$expected_count"; then
            echo "[SKIP][INFER] $MODELNAME x $DATASET: infer result is complete."
        else
            if infer_artifacts_exist "$MODELNAME" "$DATASET"; then
                echo "[CLEAN][INFER+EVAL] $MODELNAME x $DATASET: infer incomplete, remove stale artifacts."
                cleanup_all_artifacts "$MODELNAME" "$DATASET"
            fi
            if ! launch_infer_fg "$MODELNAME" "$DATASET" 32; then
                echo "[SKIP][EVAL] $MODELNAME x $DATASET: infer failed."
                continue
            fi
        fi

        if infer_complete "$MODELNAME" "$DATASET" "$expected_count"; then
            run_answer_format_postprocess "$MODELNAME" "$DATASET"
            if eval_complete "$MODELNAME" "$DATASET" "$expected_count"; then
                echo "[SKIP][EVAL] $MODELNAME x $DATASET: eval result is complete."
            else
                if eval_artifacts_exist "$MODELNAME" "$DATASET"; then
                    echo "[CLEAN][EVAL] $MODELNAME x $DATASET: eval incomplete, remove stale eval artifacts."
                    cleanup_eval_artifacts "$MODELNAME" "$DATASET"
                fi
                launch_eval_bg "$MODELNAME" "$DATASET"
            fi
        else
            echo "[SKIP][EVAL] $MODELNAME x $DATASET: infer is still incomplete."
        fi
    done
done
