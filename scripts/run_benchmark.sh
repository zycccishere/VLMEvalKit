#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTROL_PYTHON="${CONTROL_PYTHON:-python}"

if [[ "${CONTROL_PYTHON}" == */* ]]; then
    if [[ ! -x "${CONTROL_PYTHON}" ]]; then
        echo "[FATAL] control python not found: ${CONTROL_PYTHON}" >&2
        exit 1
    fi
elif ! command -v "${CONTROL_PYTHON}" >/dev/null 2>&1; then
    echo "[FATAL] control python not found on PATH: ${CONTROL_PYTHON}" >&2
    exit 1
fi

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

export LMUData="${LMUData:-${REPO_ROOT}/LMUData}"
mkdir -p "${LMUData}"

exec "${CONTROL_PYTHON}" "${SCRIPT_DIR}/run_benchmark.py" "$@"
