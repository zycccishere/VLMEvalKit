#!/usr/bin/env bash
set -euo pipefail

cd /path/to/vlmevalkit

mode="${1:-both}"
if [[ "${mode}" == -* ]]; then
  mode="both"
elif [[ "${mode}" == "symmetry" || "${mode}" == "real_half_patch" || "${mode}" == "both" ]]; then
  shift || true
else
  mode="both"
fi

case "${mode}" in
  symmetry)
    bash scripts/run_qwen25vl32b_image2_symmetry_full_20260412.sh "$@"
    ;;
  real_half_patch)
    bash scripts/run_qwen25vl32b_image2_real_half_patch_full_20260412.sh "$@"
    ;;
  both)
    bash scripts/run_qwen25vl32b_image2_symmetry_full_20260412.sh "$@"
    bash scripts/run_qwen25vl32b_image2_real_half_patch_full_20260412.sh "$@"
    ;;
  *)
    echo "[FATAL] unsupported mode: ${mode}" >&2
    echo "Usage: $0 [both|symmetry|real_half_patch] [run_benchmark args...]" >&2
    exit 1
    ;;
esac
