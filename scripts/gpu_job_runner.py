#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CANONICAL_RUNNER = Path("/path/to/LLaVA/scripts/gpu_job_runner.py")


def main() -> int:
    if not CANONICAL_RUNNER.exists():
        raise SystemExit(
            f"Canonical runner not found: {CANONICAL_RUNNER}. "
            "Run the LLaVA-side gpu_job_runner.py directly."
        )

    cmd = [sys.executable, str(CANONICAL_RUNNER), *sys.argv[1:]]
    env = os.environ.copy()
    env.setdefault("GPU_JOB_RUNNER_ALIAS_SOURCE", str(Path(__file__).resolve()))
    return subprocess.run(cmd, check=False, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
