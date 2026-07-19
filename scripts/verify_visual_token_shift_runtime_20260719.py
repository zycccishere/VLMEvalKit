#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

from safetensors import safe_open


MODEL_DIRS = (
    "Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-3B-Instruct",
    "MiniCPM-o-4_5",
)


def verify_snapshot(root: Path) -> dict:
    required_metadata = ("config.json", "tokenizer_config.json")
    missing_metadata = [name for name in required_metadata if not (root / name).is_file()]
    indexes = sorted(root.glob("*.safetensors.index.json"))
    weight_files: set[str] = set()
    for path in indexes:
        weight_files.update(json.loads(path.read_text(encoding="utf-8"))["weight_map"].values())
    missing_weights = [
        name
        for name in sorted(weight_files)
        if not (root / name).is_file() or (root / name).stat().st_size < 1024
    ]
    invalid_weights = []
    for name in sorted(weight_files):
        path = root / name
        if name in missing_weights:
            continue
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                if not handle.keys():
                    invalid_weights.append(f"{name}: no tensor keys")
        except Exception as exc:
            invalid_weights.append(f"{name}: {type(exc).__name__}: {exc}")
    single = root / "model.safetensors"
    single_ok = single.is_file() and single.stat().st_size > 100_000_000
    if single_ok:
        try:
            with safe_open(single, framework="pt", device="cpu") as handle:
                single_ok = bool(handle.keys())
        except Exception:
            single_ok = False
    weights_ok = (bool(weight_files) and not missing_weights and not invalid_weights) or single_ok
    return {
        "ok": not missing_metadata and weights_ok,
        "missing_metadata": missing_metadata,
        "indexed_weight_files": len(weight_files),
        "missing_weights": missing_weights,
        "invalid_weights": invalid_weights,
        "single_model_safetensors": single_ok,
    }


def verify_openai_auth(required: bool) -> dict:
    key = (
        os.environ.get("OPENAI_API_KEY_JUDGE", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    base = (
        os.environ.get("OPENAI_API_BASE_JUDGE", "").strip()
        or os.environ.get("OPENAI_API_BASE", "").strip()
        or "https://api.openai.com"
    ).rstrip("/")
    result = {
        "required": required,
        "credentials_present": bool(key),
        "authenticated": None,
        "http_status": None,
        "ok": not required,
    }
    if not required:
        return result
    if not key:
        result["ok"] = False
        return result
    url = base + ("/models" if base.endswith("/v1") else "/v1/models")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result["http_status"] = response.status
            result["authenticated"] = response.status == 200
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["authenticated"] = False
    except Exception as exc:
        result["authenticated"] = False
        result["error_type"] = type(exc).__name__
    result["ok"] = result["authenticated"] is True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--minicpm-transformers-pydeps", type=Path, required=True)
    parser.add_argument("--require-openai-auth", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("VLMEVAL_API_MINIMAL_IMPORT", "1")
    os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")
    os.environ.setdefault("VLMEVAL_LAZY_INIT", "1")

    modules = (
        "torch",
        "transformers",
        "accelerate",
        "qwen_vl_utils",
        "validators",
        "sty",
        "openpyxl",
        "xlsxwriter",
        "matplotlib",
        "portalocker",
        "tabulate",
        "imageio",
        "openai",
        "dotenv",
        "vlmeval.dataset",
        "vlmeval.config_qwen_minimal",
        "vlmeval.config_minicpm45_minimal",
        "vlmeval.vlm.replay_visual_token_shift",
    )
    module_status = {}
    for name in modules:
        importlib.import_module(name)
        module_status[name] = True

    import torch
    import transformers
    qwen_runtime_ok = transformers.__version__ == "5.5.0"

    minicpm_probe_env = os.environ.copy()
    minicpm_probe_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(args.minicpm_transformers_pydeps),
            minicpm_probe_env.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    minicpm_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, torch, transformers; "
                "print(json.dumps({'torch': torch.__version__, "
                "'transformers': transformers.__version__}))"
            ),
        ],
        env=minicpm_probe_env,
        check=True,
        capture_output=True,
        text=True,
    )
    minicpm_versions = json.loads(minicpm_probe.stdout.strip())
    minicpm_runtime_ok = minicpm_versions["transformers"] == "4.51.0"

    minicpm_root = args.model_root / "MiniCPM-o-4_5"
    remote_code_hashes = {}
    for name in ("configuration_minicpmo.py", "modeling_minicpmo.py", "processing_minicpmo.py"):
        path = minicpm_root / name
        if path.is_file():
            remote_code_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    snapshots = {name: verify_snapshot(args.model_root / name) for name in MODEL_DIRS}
    openai_auth = verify_openai_auth(args.require_openai_auth)
    payload = {
        "all_passed": (
            all(s["ok"] for s in snapshots.values())
            and all(module_status.values())
            and qwen_runtime_ok
            and minicpm_runtime_ok
            and openai_auth["ok"]
        ),
        "modules": module_status,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "qwen_runtime_ok": qwen_runtime_ok,
        "minicpm_versions": minicpm_versions,
        "minicpm_runtime_ok": minicpm_runtime_ok,
        "openai_auth": openai_auth,
        "minicpm_remote_code_sha256": remote_code_hashes,
        "snapshots": snapshots,
    }
    print(json.dumps(payload, indent=2))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
