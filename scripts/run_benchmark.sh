#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

if [[ -z "${CONTROL_PYTHON:-}" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        CONTROL_PYTHON="${VIRTUAL_ENV}/bin/python"
    elif [[ -n "${CONDA_ROOT:-}" && -x "${CONDA_ROOT}/envs/vlmevalkit/bin/python" ]]; then
        CONTROL_PYTHON="${CONDA_ROOT}/envs/vlmevalkit/bin/python"
    elif [[ -n "${CONDA_ROOT:-}" && -x "${CONDA_ROOT}/bin/python" ]]; then
        CONTROL_PYTHON="${CONDA_ROOT}/bin/python"
    else
        CONTROL_PYTHON=python3
    fi
fi

if [[ "${CONTROL_PYTHON}" == */* ]]; then
    if [[ ! -x "${CONTROL_PYTHON}" ]]; then
        echo "[FATAL] control python not found: ${CONTROL_PYTHON}" >&2
        exit 1
    fi
elif ! command -v "${CONTROL_PYTHON}" >/dev/null 2>&1; then
    echo "[FATAL] control python not found on PATH: ${CONTROL_PYTHON}" >&2
    exit 1
fi

if [[ -z "${LMUData:-}" ]]; then
    echo "[FATAL] LMUData is not set. Export LMUData or define it in ${REPO_ROOT}/.env." >&2
    exit 1
fi
export LMUData
if [[ ! -d "${LMUData}" ]]; then
    echo "[FATAL] LMUData not found: ${LMUData}" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${CONTROL_PYTHON}" -m vlmeval.cli.run_benchmark "$@"
