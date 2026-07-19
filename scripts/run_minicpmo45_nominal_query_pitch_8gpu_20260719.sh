#!/usr/bin/env bash
set -euo pipefail

export VLMEVAL_API_MINIMAL_IMPORT=1
export VLMEVAL_VLM_MINIMAL_IMPORT=1
export VLMEVAL_LAZY_INIT=1
export VLMEVAL_USE_MINICPM45_MINIMAL_CONFIG=1
export REPLAY_IMAGE_TRANSFORM_STRICT=1
export MINICPM45_USE_VLLM="${MINICPM45_USE_VLLM:-1}"

MODE="${1:-infer}"
if [[ $# -gt 0 ]]; then
  shift
fi

INFER_CONFIG="configs/matrix_minicpmo45_iqiq_image2_nominal_query_pitch_infer_20260719.yaml"
EVAL_CONFIG="configs/matrix_minicpmo45_iqiq_image2_nominal_query_pitch_posthoc_eval_20260719.yaml"

run_matrix() {
  bash scripts/run_benchmark.sh --matrix-config "$1" "${@:2}"
}

case "$MODE" in
  infer)
    run_matrix "$INFER_CONFIG" "$@"
    ;;
  eval)
    run_matrix "$EVAL_CONFIG" "$@"
    ;;
  all)
    run_matrix "$INFER_CONFIG" "$@"
    run_matrix "$EVAL_CONFIG" "$@"
    ;;
  plan)
    run_matrix "$INFER_CONFIG" --plan-only "$@"
    ;;
  *)
    echo "Usage: $0 {infer|eval|all|plan} [run_benchmark args...]" >&2
    exit 2
    ;;
esac
