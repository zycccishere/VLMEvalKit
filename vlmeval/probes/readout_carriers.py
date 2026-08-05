from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from vlmeval.probes.readout_v2 import (
    ANSWER_PREFIX,
    all_single_choice_labels,
    append_jsonl,
    build_runtime,
    candidate_token_plan,
    checkpoint_file_stats,
    checkpoint_identity_with_stable_stats,
    derive_prefill_boundary,
    longest_common_prefix,
    qwen_image_spans,
    row_choice_labels,
    row_option_texts,
    sha256_file,
    sha256_json,
    verify_checkpoint_identity_quick,
    write_json,
)
from vlmeval.probes.standard_entry_parity import (
    build_standard_prompt,
    patched_environ,
    refresh_replay_runtime,
    repo_git_snapshot,
    summarize_prompt_items,
)


FROZEN_SOURCE_SHA256 = {
    "all_single_choice": "fd8641c7c80ad2d6329624a8ee5fab9277e513d423487622184340bbe61e57e2",
    "fixed_choice": "126434b2ef4a859314a11fab78339015cd882c8cc1dcb9c00ee09d9f78490ee7",
    "mmstar_ai2d": "a76276314d99ff30ea85f88855ad95aec165e0ef4d46ee69086baaa618eb576c",
}
DATASET_SOURCES = {
    "DynaMath": "all_single_choice",
    "WeMath": "all_single_choice",
    "MMBench_DEV_EN_V11": "fixed_choice",
    "MMStar": "mmstar_ai2d",
    "AI2D_TEST": "mmstar_ai2d",
}
EXPECTED_COUNTS = {
    "DynaMath": 1736,
    "WeMath": 1732,
    "MMBench_DEV_EN_V11": 1110,
    "MMStar": 1490,
    "AI2D_TEST": 3060,
}
DEFAULT_DATASETS = tuple(DATASET_SOURCES)
NOISE_CARRIER_SEEDS = {
    "noise_image_s0": 17,
    "noise_image_s1": 29,
    "noise_image_s2": 43,
}
SHUFFLED_LOREM_SEEDS = {
    "shuffled_lorem_s0": 17,
    "shuffled_lorem_s1": 29,
    "shuffled_lorem_s2": 43,
}
COLOR_VALUES = {
    "blank_image": (255, 255, 255),
    "yellow_image": (255, 255, 0),
}
VISUAL_CARRIERS = (*COLOR_VALUES, *NOISE_CARRIER_SEEDS)
TEXT_CARRIERS = (
    "dot_text",
    "space_text",
    "ordered_lorem",
    *SHUFFLED_LOREM_SEEDS,
)
CARRIERS = (*VISUAL_CARRIERS, *TEXT_CARRIERS)
MASK_CONDITIONS = ("aware", "no_write", "position_null")
PRIMARY_CONDITIONS = ("blind", *CARRIERS, "full")
ENTRY_ENV_KEYS = (
    "REPLAY_MODE",
    "REPLAY_PROMPT_TEMPLATE_NAME",
    "REPLAY_TIMES",
    "REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT",
)
RECORD_FIELDS = (
    "dataset",
    "row_position",
    "sample_index",
    "answer_key",
    "choice_labels",
    "choice_count",
    "option_text_sha256",
    "circular_group_size",
)
MODEL_FAMILIES = {
    "qwen25vl_3b": "qwen25vl",
    "qwen25vl_7b": "qwen25vl",
    "qwen25vl_32b": "qwen25vl",
    "minicpm_v_45": "minicpmv45",
    "minicpm_o_45": "minicpmo45",
}
MINICPM_FAMILIES = frozenset({"minicpmv45", "minicpmo45"})
LOREM_TEXT = (
    " Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua."
)
LOGIT_PARITY_ATOL = 0.05
LOGIT_PARITY_RTOL = 0.005
CANDIDATE_PARITY_ATOL = 0.02
BLOCKED_PREFIX_ATOL = 1e-5
CORRUPTION_POSITIVE_MIN = 1e-4
SCHEMA = "topic-image-replay/readout-random-carriers/v1"
RECORD_SCHEMA = "topic-image-replay/readout-random-carrier-record/v1"


@dataclass
class PreparedSequence:
    inputs: dict[str, Any]
    prefill_len: int
    readout_indices: list[int]
    prompt_text: str
    generation_text: str
    token_roles: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_images: dict[str, Image.Image] = field(default_factory=dict)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _parse_csv(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _flatten_ints(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [item for nested in value for item in _flatten_ints(nested)]
    return [int(value)]


def _model_family(model_key: str) -> str:
    try:
        return MODEL_FAMILIES[model_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported readout-carrier model: {model_key}") from exc


def _is_minicpm_family(family: str) -> bool:
    return family in MINICPM_FAMILIES


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _record_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(record["dataset"]),
        int(record["row_position"]),
        str(record["sample_index"]),
    )


def _file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _repo_full_identity(repo_root: str | Path) -> dict[str, str]:
    root = Path(repo_root).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).rstrip("\n")
    if len(commit) != 40:
        raise RuntimeError(f"Expected a full 40-character git commit, got {commit!r}")
    return {"commit": commit, "status_short": status}


def _canonical_source_record(record: dict[str, Any]) -> dict[str, Any]:
    out = {key: copy.deepcopy(record.get(key)) for key in RECORD_FIELDS}
    out["row_position"] = int(out["row_position"])
    out["sample_index"] = str(out["sample_index"])
    out["answer_key"] = str(out["answer_key"])
    out["choice_labels"] = [str(item) for item in out["choice_labels"]]
    out["choice_count"] = int(out["choice_count"])
    if out["choice_count"] != len(out["choice_labels"]):
        raise RuntimeError(f"Choice-count mismatch in frozen record: {out}")
    if out["answer_key"] not in out["choice_labels"]:
        raise RuntimeError(f"Gold answer missing from choices in frozen record: {out}")
    return out


def build_frozen_manifest(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    datasets = _parse_csv(args.datasets)
    unknown = set(datasets) - set(DEFAULT_DATASETS)
    if unknown:
        raise ValueError(f"Unsupported frozen datasets: {sorted(unknown)}")

    source_paths = {
        "all_single_choice": Path(args.all_single_choice_manifest).resolve(),
        "fixed_choice": Path(args.fixed_choice_manifest).resolve(),
        "mmstar_ai2d": Path(args.mmstar_ai2d_manifest).resolve(),
    }
    lmu_data = Path(args.lmu_data).resolve()
    sources: dict[str, dict[str, Any]] = {}
    source_payloads: dict[str, dict[str, Any]] = {}
    for name, path in source_paths.items():
        observed = sha256_file(path)
        expected = FROZEN_SOURCE_SHA256[name]
        if observed != expected:
            raise RuntimeError(
                f"Frozen source hash mismatch for {name}: {observed} != {expected}"
            )
        source_payloads[name] = _load_json(path)
        sources[name] = {"path": str(path), "sha256": observed}

    source_data = {}
    for dataset in datasets:
        path = lmu_data / f"{dataset}.tsv"
        if not path.is_file():
            raise FileNotFoundError(f"Frozen dataset TSV is missing: {path}")
        observed = _file_identity(path)
        source_name = DATASET_SOURCES[dataset]
        accepted = source_payloads[source_name].get("source_data", {}).get(dataset)
        if accepted is not None:
            if (
                int(accepted["size"]) != observed["size"]
                or str(accepted["sha256"]) != observed["sha256"]
            ):
                raise RuntimeError(
                    f"Accepted TSV identity mismatch for {dataset}: "
                    f"{observed} != {accepted}"
                )
        observed["accepted_identity_available"] = accepted is not None
        observed["accepted_source_manifest"] = source_name
        source_data[dataset] = observed

    selected: list[dict[str, Any]] = []
    summaries = []
    seen: set[tuple[str, int, str]] = set()
    for dataset in datasets:
        source_name = DATASET_SOURCES[dataset]
        rows = [
            _canonical_source_record(record)
            for record in source_payloads[source_name]["records"]
            if record.get("dataset") == dataset
        ]
        expected_count = EXPECTED_COUNTS[dataset]
        if len(rows) != expected_count:
            raise RuntimeError(
                f"Frozen count mismatch for {dataset}: {len(rows)} != {expected_count}"
            )
        rows.sort(key=lambda row: (row["row_position"], row["sample_index"]))
        source_count = len(rows)
        if args.samples_per_dataset is not None:
            sample_count = int(args.samples_per_dataset)
            if not 0 < sample_count <= source_count:
                raise ValueError(
                    f"Invalid samples-per-dataset for {dataset}: {sample_count}"
                )
            seed_bytes = hashlib.sha256(
                f"{int(args.selection_seed)}:{dataset}".encode("utf-8")
            ).digest()[:8]
            dataset_seed = int.from_bytes(seed_bytes, "little")
            indices = np.random.default_rng(dataset_seed).choice(
                source_count, size=sample_count, replace=False
            )
            rows = [rows[int(index)] for index in sorted(indices.tolist())]
        for dataset_position, row in enumerate(rows):
            key = _record_key(row)
            if key in seen:
                raise RuntimeError(f"Duplicate frozen record key: {key}")
            seen.add(key)
            row["dataset_position"] = dataset_position
            row["experiment_position"] = len(selected)
            row["shard"] = len(selected) % int(args.num_shards)
            selected.append(row)
        summaries.append(
            {
                "dataset": dataset,
                "selected_rows": len(rows),
                "source_rows": source_count,
                "choice_count_histogram": dict(
                    sorted(Counter(row["choice_count"] for row in rows).items())
                ),
                "answer_histogram": dict(
                    sorted(Counter(row["answer_key"] for row in rows).items())
                ),
                "records_sha256": sha256_json(rows),
                "source_manifest": source_name,
            }
        )

    model_identity, model_file_stats = checkpoint_identity_with_stable_stats(
        args.model_path
    )
    implementation_path = Path(__file__).resolve()
    repo_identity = _repo_full_identity(repo_root)
    if repo_identity["status_short"]:
        raise RuntimeError(
            "Frozen carrier manifests must be built from a clean repository: "
            f"{repo_identity['status_short']}"
        )
    matrix_identity = _file_identity(args.matrix_config)
    models_config_identity = _file_identity(repo_root / "configs" / "models.yaml")
    payload = {
        "schema": SCHEMA,
        "created_at": time.time(),
        "repo_root": str(repo_root),
        "repo_snapshot": repo_git_snapshot(repo_root),
        "repo_commit": repo_identity["commit"],
        "implementation_sha256": sha256_file(implementation_path),
        "model_key": args.model_key,
        "model_family": _model_family(args.model_key),
        "model_identity": model_identity,
        "model_identity_sha256": model_identity["content_sha256"],
        "model_file_stats": model_file_stats,
        "matrix_config": matrix_identity,
        "models_config": models_config_identity,
        "answer_prefix": ANSWER_PREFIX,
        "primary_conditions": list(PRIMARY_CONDITIONS),
        "carrier_conditions": list(MASK_CONDITIONS),
        "carrier_primary_mask": "aware",
        "blind_semantics": "literal context-free Answer prefix without chat framing",
        "carrier_token_count_semantics": "per-sample projected visual-core count",
        "samples_per_dataset": args.samples_per_dataset,
        "selection_seed": int(args.selection_seed),
        "frozen_sources": sources,
        "source_data": source_data,
        "source_data_sha256": sha256_json(source_data),
        "num_shards": int(args.num_shards),
        "datasets": summaries,
        "record_count": len(selected),
        "records_sha256": sha256_json(selected),
        "records": selected,
    }
    write_json(Path(args.output), payload)
    return payload


def _verify_repo_contract(repo_root: Path, manifest: dict[str, Any]) -> None:
    identity = _repo_full_identity(repo_root)
    if identity["commit"] != manifest["repo_commit"]:
        raise RuntimeError(
            f"Runtime git commit changed: {identity['commit']} != {manifest['repo_commit']}"
        )
    if identity["status_short"]:
        raise RuntimeError(f"Runtime repository is dirty: {identity['status_short']}")


def verify_run_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError(
            f"Unexpected carrier manifest schema: {manifest.get('schema')}"
        )
    _verify_repo_contract(repo_root, manifest)
    if sha256_file(Path(__file__).resolve()) != manifest["implementation_sha256"]:
        raise RuntimeError("Carrier implementation changed after manifest creation")
    if (
        _file_identity(args.matrix_config)["sha256"]
        != manifest["matrix_config"]["sha256"]
    ):
        raise RuntimeError("Matrix config changed after manifest creation")
    if (
        _file_identity(repo_root / "configs" / "models.yaml")["sha256"]
        != manifest["models_config"]["sha256"]
    ):
        raise RuntimeError("Models config changed after manifest creation")
    checkpoint_identity, checkpoint_stats = checkpoint_identity_with_stable_stats(
        args.model_path
    )
    if checkpoint_identity != manifest["model_identity"]:
        raise RuntimeError(
            "Full checkpoint SHA-256 identity changed after smoke manifest"
        )
    for dataset, expected in manifest["source_data"].items():
        observed = _file_identity(Path(args.lmu_data).resolve() / f"{dataset}.tsv")
        if (
            observed["size"] != expected["size"]
            or observed["sha256"] != expected["sha256"]
        ):
            raise RuntimeError(
                f"Dataset source changed after manifest creation: {dataset}"
            )
    payload = {
        "schema": "topic-image-replay/readout-carrier-run-contract/v1",
        "passed": True,
        "created_at": time.time(),
        "manifest_sha256": sha256_file(manifest_path),
        "repo_commit": manifest["repo_commit"],
        "implementation_sha256": manifest["implementation_sha256"],
        "model_key": manifest["model_key"],
        "model_identity_sha256": manifest["model_identity_sha256"],
        "checkpoint_file_stats": checkpoint_stats,
        "source_data_sha256": manifest["source_data_sha256"],
        "matrix_config_sha256": manifest["matrix_config"]["sha256"],
        "models_config_sha256": manifest["models_config"]["sha256"],
    }
    write_json(Path(args.output), payload)
    return payload


def _validate_run_contract_attestation(
    path: str | Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    model_path: str,
) -> None:
    payload = _load_json(path)
    expected = {
        "manifest_sha256": sha256_file(manifest_path),
        "repo_commit": manifest["repo_commit"],
        "implementation_sha256": manifest["implementation_sha256"],
        "model_key": manifest["model_key"],
        "model_identity_sha256": manifest["model_identity_sha256"],
        "source_data_sha256": manifest["source_data_sha256"],
        "matrix_config_sha256": manifest["matrix_config"]["sha256"],
        "models_config_sha256": manifest["models_config"]["sha256"],
    }
    if payload.get("passed") is not True:
        raise RuntimeError(f"Run-contract attestation did not pass: {path}")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Run-contract attestation mismatch for {key}: {path}")
    if checkpoint_file_stats(model_path) != payload.get("checkpoint_file_stats"):
        raise RuntimeError(
            "Checkpoint file size/mtime changed after centralized full hashing"
        )


def _load_probe_model(env: dict[str, str], registry_name: str, family: str) -> Any:
    with patched_environ(env):
        os.environ.pop("WORLD_SIZE", None)
        for name in (
            "vlmeval.config_runtime",
            "vlmeval.config_qwen_minimal",
            "vlmeval.config_minicpm45_minimal",
        ):
            sys.modules.pop(name, None)
        cfg = importlib.import_module("vlmeval.config_runtime")
        if registry_name not in cfg.supported_VLM:
            raise KeyError(
                f"{registry_name} not in runtime registry: {sorted(cfg.supported_VLM)}"
            )
        wrapper = cfg.supported_VLM[registry_name](use_vllm=False)
        if family == "qwen25vl":
            refresh_replay_runtime(wrapper, env)
        if hasattr(wrapper, "model") and hasattr(wrapper.model, "eval"):
            wrapper.model.eval()
        return wrapper


def _token_id(tokenizer: Any, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if (
        not isinstance(value, int)
        or value < 0
        or value == getattr(tokenizer, "unk_token_id", None)
    ):
        raise RuntimeError(f"Tokenizer has no usable token id for {token!r}")
    return int(value)


def _literal_token_id(tokenizer: Any, literal: str) -> int:
    ids = list(tokenizer(literal, add_special_tokens=False).input_ids)
    if len(ids) != 1:
        raise RuntimeError(f"Literal {literal!r} is not exactly one token: {ids}")
    decoded = tokenizer.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != literal:
        raise RuntimeError(
            f"Literal token is not reversibly decoded: {literal!r} -> {ids} -> {decoded!r}"
        )
    if int(ids[0]) in set(getattr(tokenizer, "all_special_ids", []) or []):
        raise RuntimeError(f"Literal {literal!r} resolved to a special token: {ids[0]}")
    return int(ids[0])


def _token_ids_sha256(token_ids: Iterable[int]) -> str:
    values = np.asarray(list(token_ids), dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _lorem_carrier_token_ids(
    tokenizer: Any, target_n: int, carrier: str
) -> tuple[list[int], dict[str, Any]]:
    source_ids = [
        int(item) for item in tokenizer(LOREM_TEXT, add_special_tokens=False).input_ids
    ]
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if not source_ids or any(token_id in special_ids for token_id in source_ids):
        raise RuntimeError("Lorem carrier produced an empty or special-token sequence")
    ordered = (source_ids * math.ceil(target_n / len(source_ids)))[:target_n]
    token_ids = list(ordered)
    shuffle_seed = SHUFFLED_LOREM_SEEDS.get(carrier)
    if shuffle_seed is not None:
        permutation = np.random.default_rng(shuffle_seed).permutation(target_n)
        token_ids = [ordered[int(index)] for index in permutation]
    elif carrier != "ordered_lorem":
        raise ValueError(f"Unknown Lorem carrier: {carrier}")
    return token_ids, {
        "text_source": "canonical_lorem_ipsum",
        "source_text": LOREM_TEXT,
        "shuffle_seed": shuffle_seed,
        "ordered_token_ids_sha256": _token_ids_sha256(ordered),
    }


def _splice_ids_2d(
    inputs: dict[str, Any],
    start: int,
    end: int,
    token_ids: list[int],
    *,
    recompute_position_ids: bool,
) -> dict[str, Any]:
    out = dict(inputs)
    old_ids = inputs["input_ids"]
    if old_ids.ndim != 2 or old_ids.shape[0] != 1:
        raise RuntimeError(
            f"Expected one unpadded sequence, got {tuple(old_ids.shape)}"
        )
    old_len = int(old_ids.shape[-1])
    if not 0 <= start <= end <= old_len:
        raise RuntimeError(
            f"Invalid token splice [{start}, {end}) for length {old_len}"
        )
    extra = torch.tensor([token_ids], dtype=old_ids.dtype, device=old_ids.device)
    out["input_ids"] = torch.cat([old_ids[:, :start], extra, old_ids[:, end:]], dim=-1)
    for key, value in inputs.items():
        if key == "input_ids" or not isinstance(value, torch.Tensor):
            continue
        if value.ndim == 2 and int(value.shape[-1]) == old_len:
            if key == "attention_mask":
                fill = torch.ones(
                    (value.shape[0], len(token_ids)),
                    dtype=value.dtype,
                    device=value.device,
                )
                out[key] = torch.cat([value[:, :start], fill, value[:, end:]], dim=-1)
            elif key == "position_ids" and recompute_position_ids:
                continue
            else:
                raise RuntimeError(
                    f"Sequence tensor {key} requires explicit insertion semantics: {tuple(value.shape)}"
                )
    if recompute_position_ids and "position_ids" in inputs:
        attention = out.get("attention_mask")
        if attention is None or not torch.all(attention == 1):
            raise RuntimeError("MiniCPM readout path requires one unpadded sequence")
        out["position_ids"] = torch.arange(
            out["input_ids"].shape[-1],
            dtype=torch.long,
            device=out["input_ids"].device,
        ).unsqueeze(0)
    return out


def _append_ids(
    inputs: dict[str, Any], token_ids: list[int], *, recompute_position_ids: bool
) -> dict[str, Any]:
    position = int(inputs["input_ids"].shape[-1])
    return _splice_ids_2d(
        inputs,
        position,
        position,
        token_ids,
        recompute_position_ids=recompute_position_ids,
    )


def _single_edit_spec(
    base_ids: torch.Tensor,
    expanded_ids: torch.Tensor,
    core_start: int,
    core_end_exclusive: int,
) -> dict[str, Any]:
    base = [int(item) for item in base_ids[0].detach().cpu().tolist()]
    expanded = [int(item) for item in expanded_ids[0].detach().cpu().tolist()]
    prefix = len(longest_common_prefix([base, expanded]))
    suffix = 0
    suffix_limit = min(len(base) - prefix, len(expanded) - prefix)
    while suffix < suffix_limit and base[-1 - suffix] == expanded[-1 - suffix]:
        suffix += 1
    base_middle_end = len(base) - suffix if suffix else len(base)
    expanded_middle_end = len(expanded) - suffix if suffix else len(expanded)
    if not (prefix <= core_start < core_end_exclusive <= expanded_middle_end):
        raise RuntimeError(
            "Readout core is outside the expanded carrier envelope: "
            f"prefix={prefix} core=[{core_start},{core_end_exclusive}) "
            f"edit_end={expanded_middle_end}"
        )
    return {
        "edit_start": prefix,
        "source_end": base_middle_end,
        "source_ids": base[prefix:base_middle_end],
        "expanded_ids": expanded[prefix:expanded_middle_end],
        "prefix_envelope_ids": expanded[prefix:core_start],
        "suffix_envelope_ids": expanded[core_end_exclusive:expanded_middle_end],
        "core_offset": core_start - prefix,
        "suffix_length": suffix,
        "source_prefix_sha256": hashlib.sha256(
            np.asarray(base[:prefix], dtype=np.int64).tobytes()
        ).hexdigest(),
    }


def _insert_matched_text_ids_carrier(
    base: PreparedSequence,
    reference: PreparedSequence,
    *,
    core_start: int,
    core_end_exclusive: int,
    carrier_token_ids: list[int],
    carrier: str,
    family: str,
    carrier_metadata: dict[str, Any] | None = None,
) -> PreparedSequence:
    spec = _single_edit_spec(
        base.inputs["input_ids"],
        reference.inputs["input_ids"],
        core_start,
        core_end_exclusive,
    )
    target_n = core_end_exclusive - core_start
    if len(carrier_token_ids) != target_n:
        raise RuntimeError(
            f"Carrier {carrier} has {len(carrier_token_ids)} tokens, expected {target_n}"
        )
    replacement = (
        list(spec["prefix_envelope_ids"])
        + [int(item) for item in carrier_token_ids]
        + list(spec["suffix_envelope_ids"])
    )
    if len(replacement) != len(spec["expanded_ids"]):
        raise AssertionError("Matched text carrier changed expanded-envelope length")
    if int(spec["source_end"]) > int(base.prefill_len):
        raise RuntimeError("Carrier edit crosses the standard IQ prefill boundary")
    if int(spec["edit_start"]) + len(replacement) > int(reference.prefill_len):
        raise RuntimeError("Carrier edit crosses the IQI prefill boundary")
    out = copy.copy(base)
    out.inputs = _splice_ids_2d(
        base.inputs,
        int(spec["edit_start"]),
        int(spec["source_end"]),
        replacement,
        recompute_position_ids=_is_minicpm_family(family),
    )
    out.prefill_len = int(base.prefill_len) - len(spec["source_ids"]) + len(replacement)
    readout_start = int(spec["edit_start"]) + len(spec["prefix_envelope_ids"])
    out.readout_indices = list(range(readout_start, readout_start + target_n))
    out.token_roles = list(base.token_roles)
    inserted_roles = (
        ["carrier_envelope"] * len(spec["prefix_envelope_ids"])
        + [f"readout:{carrier}"] * target_n
        + ["carrier_envelope"] * len(spec["suffix_envelope_ids"])
    )
    out.token_roles[int(spec["edit_start"]) : int(spec["source_end"])] = inserted_roles
    out_ids = out.inputs["input_ids"][0]
    reference_ids = reference.inputs["input_ids"][0]
    if not torch.equal(
        out_ids[:core_start], reference_ids[:core_start]
    ) or not torch.equal(
        out_ids[core_end_exclusive:], reference_ids[core_end_exclusive:]
    ):
        raise RuntimeError("Matched text carrier differs from IQI outside its core")
    out.metadata = copy.deepcopy(base.metadata)
    out.artifact_images = dict(base.artifact_images)
    out.metadata.update(
        {
            "carrier": carrier,
            "carrier_kind": "text_token_ids_in_matched_visual_envelope",
            "carrier_token_ids_sha256": _token_ids_sha256(carrier_token_ids),
            "carrier_token_unique_count": len(set(carrier_token_ids)),
            "edit_position": int(spec["edit_start"]),
            "replaced_source_token_count": len(spec["source_ids"]),
            "replacement_token_count": len(replacement),
            "prefix_envelope_token_count": len(spec["prefix_envelope_ids"]),
            "suffix_envelope_token_count": len(spec["suffix_envelope_ids"]),
            "source_prefix_sha256": spec["source_prefix_sha256"],
            "rendered_text_excludes_injected_carrier": True,
        }
    )
    out.metadata.update(carrier_metadata or {})
    return out


def _insert_matched_text_carrier(
    base: PreparedSequence,
    reference: PreparedSequence,
    *,
    core_start: int,
    core_end_exclusive: int,
    literal_token_id: int,
    carrier: str,
    family: str,
) -> PreparedSequence:
    target_n = core_end_exclusive - core_start
    return _insert_matched_text_ids_carrier(
        base,
        reference,
        core_start=core_start,
        core_end_exclusive=core_end_exclusive,
        carrier_token_ids=[int(literal_token_id)] * target_n,
        carrier=carrier,
        family=family,
        carrier_metadata={"literal_token_id": int(literal_token_id)},
    )


def _make_qwen_messages(
    wrapper: Any, content: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    messages = []
    if getattr(wrapper, "system_prompt", None) is not None:
        messages.append({"role": "system", "content": wrapper.system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


def _prepare_qwen_content(
    wrapper: Any,
    content: list[dict[str, Any]],
    *,
    expected_images: int,
) -> PreparedSequence:
    from qwen_vl_utils import process_vision_info

    messages = _make_qwen_messages(wrapper, content)
    prompt_text_obj = wrapper.processor.apply_chat_template(
        [messages], tokenize=False, add_generation_prompt=False
    )
    generation_text_obj = wrapper.processor.apply_chat_template(
        [messages], tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info([messages])
    inputs = wrapper.processor(
        text=generation_text_obj,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    inputs = dict(inputs)
    prefill_len, boundary_meta = derive_prefill_boundary(
        wrapper,
        messages,
        prompt_text_obj,
        generation_text_obj,
        inputs,
        images,
        videos,
    )
    if not torch.all(inputs["attention_mask"] == 1):
        raise RuntimeError("Qwen carrier path unexpectedly contains padding")
    spans = qwen_image_spans(inputs["input_ids"], wrapper.processor.tokenizer)
    if len(spans) != expected_images:
        raise RuntimeError(f"Expected {expected_images} Qwen images, got {spans}")
    prompt_text = (
        prompt_text_obj[0] if isinstance(prompt_text_obj, list) else prompt_text_obj
    )
    generation_text = (
        generation_text_obj[0]
        if isinstance(generation_text_obj, list)
        else generation_text_obj
    )
    roles = [
        "prefill" if idx < prefill_len else "decode_prefix"
        for idx in range(inputs["input_ids"].shape[-1])
    ]
    tokenizer = wrapper.processor.tokenizer
    return PreparedSequence(
        inputs=inputs,
        prefill_len=int(prefill_len),
        readout_indices=[],
        prompt_text=str(prompt_text),
        generation_text=str(generation_text),
        token_roles=roles,
        metadata={
            "model_family": "qwen25vl",
            "image_spans": spans,
            "image_grid_thw": (
                inputs.get("image_grid_thw").detach().cpu().tolist()
                if isinstance(inputs.get("image_grid_thw"), torch.Tensor)
                else None
            ),
            "boundary": boundary_meta,
            "image_token_id": _token_id(tokenizer, "<|image_pad|>"),
            "vision_start_token_id": _token_id(tokenizer, "<|vision_start|>"),
            "vision_end_token_id": _token_id(tokenizer, "<|vision_end|>"),
            "spatial_merge_size": int(
                wrapper.model.config.vision_config.spatial_merge_size
            ),
        },
    )


def _qwen_replay_content(
    wrapper: Any, dataset: Any, dataset_name: str, row: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prompt = build_standard_prompt(wrapper, dataset, dataset_name, row)
    replayed = wrapper._prepare_content(prompt, dataset=dataset_name)
    types = [str(item.get("type")) for item in replayed]
    if types != ["image", "text", "image"]:
        raise RuntimeError(f"Expected exact Qwen IQI order, got {types}")
    base = [copy.deepcopy(replayed[0]), copy.deepcopy(replayed[1])]
    return replayed, base, prompt


def _natural_noise_image(size: tuple[int, int], seed: int) -> Image.Image:
    width, height = (int(size[0]), int(size[1]))
    side = 256
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((side, side))
    fy = np.fft.fftfreq(side)[:, None]
    fx = np.fft.fftfreq(side)[None, :]
    frequency = np.sqrt(fx * fx + fy * fy)
    frequency[0, 0] = 1.0
    filtered = np.fft.ifft2(np.fft.fft2(white) / frequency).real
    filtered -= filtered.mean()
    filtered /= filtered.std()
    grayscale = np.clip(127.5 + 40.0 * filtered, 24.0, 231.0).astype(np.uint8)
    rgb = np.repeat(grayscale[:, :, None], 3, axis=2)
    base = Image.fromarray(rgb, mode="RGB")
    return base.resize((width, height), Image.Resampling.BICUBIC)


def _image_sha256(image: Image.Image) -> str:
    return hashlib.sha256(
        np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes()
    ).hexdigest()


def _visual_carrier_image(original: Image.Image, carrier: str) -> Image.Image:
    if carrier in COLOR_VALUES:
        return Image.new("RGB", original.size, color=COLOR_VALUES[carrier])
    return _natural_noise_image(original.size, NOISE_CARRIER_SEEDS[carrier])


def _visual_carrier_metadata(carrier: str, image: Image.Image) -> dict[str, Any]:
    metadata = {
        "raw_carrier_sha256": _image_sha256(image),
        "raw_carrier_size": list(image.size),
    }
    if carrier in COLOR_VALUES:
        metadata["raw_carrier_rgb"] = list(COLOR_VALUES[carrier])
    else:
        metadata.update(
            {
                "noise_seed": int(NOISE_CARRIER_SEEDS[carrier]),
                "noise_family": "grayscale_power_spectrum_1_over_f_squared",
            }
        )
    return metadata


def _qwen_visual_content(
    replayed: list[dict[str, Any]], carrier: str
) -> tuple[list[dict[str, Any]], Image.Image, Image.Image]:
    from qwen_vl_utils import fetch_image

    original = fetch_image(replayed[2]).convert("RGB")
    replacement = _visual_carrier_image(original, carrier)
    content = copy.deepcopy(replayed)
    content[2] = dict(content[2])
    content[2]["image"] = replacement
    return content, original, replacement


def _prepare_qwen_sequences(
    wrapper: Any, dataset: Any, dataset_name: str, row: Any
) -> dict[str, PreparedSequence]:
    replayed, base, standard_prompt = _qwen_replay_content(
        wrapper, dataset, dataset_name, row
    )
    reference = _prepare_qwen_content(wrapper, replayed, expected_images=2)
    full = _prepare_qwen_content(wrapper, base, expected_images=1)
    prompt_summary = summarize_prompt_items(standard_prompt)
    full.metadata.update(
        {
            "standard_prompt": prompt_summary,
            "standard_prompt_sha256": sha256_json(prompt_summary),
        }
    )
    sequences: dict[str, PreparedSequence] = {"full": full}
    reference_grid = reference.metadata["image_grid_thw"]
    reference_core = reference.metadata["image_spans"][1]
    edit_spec = _single_edit_spec(
        full.inputs["input_ids"],
        reference.inputs["input_ids"],
        int(reference_core["core_start"]),
        int(reference_core["core_end"]) + 1,
    )

    for carrier in VISUAL_CARRIERS:
        content, original, replacement = _qwen_visual_content(replayed, carrier)
        prepared = _prepare_qwen_content(wrapper, content, expected_images=2)
        if prepared.metadata["image_grid_thw"] != reference_grid:
            raise RuntimeError(
                f"Qwen {carrier} changed image_grid_thw: "
                f"{prepared.metadata['image_grid_thw']} != {reference_grid}"
            )
        span = prepared.metadata["image_spans"][1]
        prepared.readout_indices = list(
            range(int(span["core_start"]), int(span["core_end"]) + 1)
        )
        for idx in prepared.readout_indices:
            prepared.token_roles[idx] = f"readout:{carrier}"
        prepared.metadata.update(
            {
                "carrier": carrier,
                "carrier_kind": "projected_visual_core",
                "reference_image_grid_thw": reference_grid,
                "reference_readout_span": reference.metadata["image_spans"][1],
                "raw_source_size": list(original.size),
                "standard_prompt": prompt_summary,
                "standard_prompt_sha256": sha256_json(prompt_summary),
                "source_prefix_sha256": edit_spec["source_prefix_sha256"],
                "edit_position": edit_spec["edit_start"],
                "replaced_source_token_count": len(edit_spec["source_ids"]),
                "replacement_token_count": len(edit_spec["expanded_ids"]),
                "prefix_envelope_token_count": len(edit_spec["prefix_envelope_ids"]),
                "suffix_envelope_token_count": len(edit_spec["suffix_envelope_ids"]),
            }
        )
        prepared.metadata.update(_visual_carrier_metadata(carrier, replacement))
        prepared.artifact_images[carrier] = replacement
        prepared.artifact_images["source_readout_image"] = original
        sequences[carrier] = prepared

    target_n = len(sequences["blank_image"].readout_indices)
    if target_n != len(sequences["yellow_image"].readout_indices):
        raise RuntimeError("Qwen blank/yellow projected token counts differ")
    tokenizer = wrapper.processor.tokenizer
    for carrier, literal in (("dot_text", "."), ("space_text", " ")):
        token = _literal_token_id(tokenizer, literal)
        prepared = _insert_matched_text_carrier(
            full,
            reference,
            core_start=int(reference_core["core_start"]),
            core_end_exclusive=int(reference_core["core_end"]) + 1,
            literal_token_id=token,
            carrier=carrier,
            family="qwen25vl",
        )
        prepared.metadata.update(
            {
                "literal": literal,
                "standard_prompt": prompt_summary,
                "standard_prompt_sha256": sha256_json(prompt_summary),
            }
        )
        sequences[carrier] = prepared

    for carrier in ("ordered_lorem", *SHUFFLED_LOREM_SEEDS):
        token_ids, metadata = _lorem_carrier_token_ids(tokenizer, target_n, carrier)
        prepared = _insert_matched_text_ids_carrier(
            full,
            reference,
            core_start=int(reference_core["core_start"]),
            core_end_exclusive=int(reference_core["core_end"]) + 1,
            carrier_token_ids=token_ids,
            carrier=carrier,
            family="qwen25vl",
            carrier_metadata=metadata,
        )
        prepared.metadata.update(
            {
                "standard_prompt": prompt_summary,
                "standard_prompt_sha256": sha256_json(prompt_summary),
            }
        )
        sequences[carrier] = prepared

    _validate_matched_readout_counts(sequences)
    return sequences


def _minicpm_content_to_prompt(
    wrapper: Any,
    content: list[Any],
    *,
    family: str,
    add_generation_prompt: bool,
) -> tuple[str, list[Image.Image]]:
    images: list[Image.Image] = []
    parts: list[str] = []
    for item in content:
        if isinstance(item, Image.Image):
            images.append(item.convert("RGB"))
            parts.append(
                "(<image>./</image>)"
                if family == "minicpmv45"
                else "<image>./</image>"
            )
        elif isinstance(item, str):
            parts.append(item)
        else:
            raise TypeError(f"Unsupported MiniCPM content item: {type(item).__name__}")
    messages = [{"role": "user", "content": "\n".join(parts)}]
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    if family == "minicpmo45":
        template_kwargs["use_tts_template"] = False
    prompt = wrapper.processor.tokenizer.apply_chat_template(
        messages, **template_kwargs
    )
    return str(prompt), images


def _minicpm_processor_call(
    wrapper: Any,
    prompt: str,
    images: list[Image.Image],
    *,
    family: str,
) -> dict[str, Any]:
    model = wrapper.model
    if family == "minicpmo45":
        model.prepare_processor(processor=wrapper.processor, tokenizer=wrapper.tokenizer)
        batch = wrapper.processor(
            [prompt],
            [images],
            [[]],
            [[]],
            max_slice_nums=1,
            use_image_id=True,
            stream_input=False,
            return_tensors="pt",
            max_length=8192,
        ).to(model.device)
    else:
        batch = wrapper.processor(
            [prompt],
            [images],
            max_slice_nums=1,
            use_image_id=True,
            return_tensors="pt",
            max_length=8192,
        ).to(model.device)
    out = dict(batch)
    out.pop("image_sizes", None)
    return out


def _prepare_minicpm_content(
    wrapper: Any,
    content: list[Any],
    *,
    family: str,
    expected_images: int,
) -> PreparedSequence:
    prompt_text, prompt_images = _minicpm_content_to_prompt(
        wrapper, content, family=family, add_generation_prompt=False
    )
    generation_text, generation_images = _minicpm_content_to_prompt(
        wrapper, content, family=family, add_generation_prompt=True
    )
    if (
        len(generation_images) != expected_images
        or len(prompt_images) != expected_images
    ):
        raise RuntimeError(
            f"Expected {expected_images} MiniCPM images, got "
            f"{len(prompt_images)}/{len(generation_images)}"
        )
    inputs = _minicpm_processor_call(
        wrapper, generation_text, generation_images, family=family
    )
    prompt_inputs = _minicpm_processor_call(
        wrapper, prompt_text, prompt_images, family=family
    )
    prompt_ids = prompt_inputs["input_ids"][0].detach().cpu().tolist()
    generation_ids = inputs["input_ids"][0].detach().cpu().tolist()
    if generation_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "MiniCPM generation prompt is not prefixed by its user prompt"
        )
    if not torch.all(inputs["attention_mask"] == 1):
        raise RuntimeError("MiniCPM carrier path unexpectedly contains padding")
    expected_positions = torch.arange(
        inputs["input_ids"].shape[-1],
        device=inputs["input_ids"].device,
        dtype=torch.long,
    ).unsqueeze(0)
    position_ids = inputs.get("position_ids")
    if position_ids is not None and not torch.equal(
        position_ids.long(), expected_positions
    ):
        raise RuntimeError("MiniCPM processor returned unexpected position_ids")
    inputs["position_ids"] = expected_positions
    bounds_raw = inputs.get("image_bound")
    if not isinstance(bounds_raw, list) or len(bounds_raw) != 1:
        raise RuntimeError(
            f"Unexpected MiniCPM image_bound container: {type(bounds_raw)}"
        )
    bounds_tensor = bounds_raw[0]
    bounds = (
        bounds_tensor.detach().cpu().tolist()
        if isinstance(bounds_tensor, torch.Tensor)
        else list(bounds_tensor)
    )
    if len(bounds) != expected_images:
        raise RuntimeError(
            f"MiniCPM config must yield one image bound per source image; got {bounds}"
        )
    query_num = int(wrapper.model.config.query_num)
    bound_lengths = [int(end) - int(start) for start, end in bounds]
    if any(length != query_num for length in bound_lengths):
        raise RuntimeError(
            f"MiniCPM image bounds do not each contain query_num={query_num}: {bounds}"
        )
    roles = [
        "prefill" if idx < len(prompt_ids) else "decode_prefix"
        for idx in range(len(generation_ids))
    ]
    return PreparedSequence(
        inputs=inputs,
        prefill_len=len(prompt_ids),
        readout_indices=[],
        prompt_text=prompt_text,
        generation_text=generation_text,
        token_roles=roles,
        metadata={
            "model_family": family,
            "image_bound": bounds,
            "image_bound_lengths": bound_lengths,
            "tgt_sizes": _nested_tensor_summary_value(inputs.get("tgt_sizes")),
            "temporal_ids": _nested_tensor_summary_value(inputs.get("temporal_ids")),
            "query_num": query_num,
            "max_slice_nums": 1,
            "model_config_max_slice_nums": int(
                wrapper.model.config.slice_config.max_slice_nums
            ),
            "vision_encode_mode": "per_image_batch_one",
        },
    )


def _minicpm_replay_content(
    wrapper: Any, dataset: Any, dataset_name: str, row: Any
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    wrapper.tokenizer.chat_template = wrapper._select_hf_chat_template(dataset_name)
    wrapper.processor.tokenizer = wrapper.tokenizer
    prompt = build_standard_prompt(wrapper, dataset, dataset_name, row)
    replayed_message = wrapper._apply_replay_pipeline(prompt, dataset=dataset_name)
    content = wrapper._message_to_content(replayed_message, dataset=dataset_name)
    content = _normalize_minicpm_iqi_content(content)
    base = [content[0], content[1]]
    return content, base, prompt


def _normalize_minicpm_iqi_content(content: list[Any]) -> list[Any]:
    types = [
        (
            "image"
            if isinstance(item, Image.Image)
            else "text" if isinstance(item, str) else type(item).__name__
        )
        for item in content
    ]
    if len(content) < 3 or types[0] != "image" or types[-1] != "image":
        raise RuntimeError(f"Expected exact MiniCPM IQI order, got {types}")
    if any(item_type != "text" for item_type in types[1:-1]):
        raise RuntimeError(
            f"Expected only MiniCPM question text between images, got {types}"
        )
    return [content[0], "\n".join(content[1:-1]), content[-1]]


def _prepare_minicpm_sequences(
    wrapper: Any, dataset: Any, dataset_name: str, row: Any, *, family: str
) -> dict[str, PreparedSequence]:
    replayed, base, standard_prompt = _minicpm_replay_content(
        wrapper, dataset, dataset_name, row
    )
    reference = _prepare_minicpm_content(
        wrapper, replayed, family=family, expected_images=2
    )
    full = _prepare_minicpm_content(wrapper, base, family=family, expected_images=1)
    prompt_summary = summarize_prompt_items(standard_prompt)
    full.metadata.update(
        {
            "standard_prompt": prompt_summary,
            "standard_prompt_sha256": sha256_json(prompt_summary),
        }
    )
    sequences: dict[str, PreparedSequence] = {"full": full}
    if reference.metadata["max_slice_nums"] != 1:
        raise RuntimeError(
            "MiniCPM carrier experiment is frozen to max_slice_nums=1; "
            f"runtime reported {reference.metadata['max_slice_nums']}"
        )
    second = replayed[2].convert("RGB")
    second_start, second_end = reference.metadata["image_bound"][1]
    edit_spec = _single_edit_spec(
        full.inputs["input_ids"],
        reference.inputs["input_ids"],
        int(second_start),
        int(second_end),
    )
    for carrier in VISUAL_CARRIERS:
        replacement = _visual_carrier_image(second, carrier)
        content = [replayed[0], replayed[1], replacement]
        prepared = _prepare_minicpm_content(
            wrapper, content, family=family, expected_images=2
        )
        if prepared.metadata["image_bound"] != reference.metadata["image_bound"]:
            raise RuntimeError(
                f"MiniCPM {carrier} changed image_bound: "
                f"{prepared.metadata['image_bound']} != {reference.metadata['image_bound']}"
            )
        start, end = prepared.metadata["image_bound"][1]
        prepared.readout_indices = list(range(int(start), int(end)))
        for idx in prepared.readout_indices:
            prepared.token_roles[idx] = f"readout:{carrier}"
        prepared.metadata.update(
            {
                "carrier": carrier,
                "carrier_kind": "global_resampler_query_core",
                "reference_image_bound": reference.metadata["image_bound"],
                "reference_tgt_sizes": reference.metadata["tgt_sizes"],
                "raw_source_size": list(second.size),
                "standard_prompt": prompt_summary,
                "standard_prompt_sha256": sha256_json(prompt_summary),
                "source_prefix_sha256": edit_spec["source_prefix_sha256"],
                "edit_position": edit_spec["edit_start"],
                "replaced_source_token_count": len(edit_spec["source_ids"]),
                "replacement_token_count": len(edit_spec["expanded_ids"]),
                "prefix_envelope_token_count": len(edit_spec["prefix_envelope_ids"]),
                "suffix_envelope_token_count": len(edit_spec["suffix_envelope_ids"]),
            }
        )
        prepared.metadata.update(_visual_carrier_metadata(carrier, replacement))
        prepared.artifact_images[carrier] = replacement
        prepared.artifact_images["source_readout_image"] = second
        sequences[carrier] = prepared

    tokenizer = wrapper.tokenizer
    for carrier, literal in (("dot_text", "."), ("space_text", " ")):
        token = _literal_token_id(tokenizer, literal)
        prepared = _insert_matched_text_carrier(
            full,
            reference,
            core_start=int(second_start),
            core_end_exclusive=int(second_end),
            literal_token_id=token,
            carrier=carrier,
            family=family,
        )
        prepared.metadata.update(
            {
                "literal": literal,
                "standard_prompt": prompt_summary,
                "standard_prompt_sha256": sha256_json(prompt_summary),
            }
        )
        sequences[carrier] = prepared

    target_n = len(sequences["blank_image"].readout_indices)
    for carrier in ("ordered_lorem", *SHUFFLED_LOREM_SEEDS):
        token_ids, metadata = _lorem_carrier_token_ids(tokenizer, target_n, carrier)
        prepared = _insert_matched_text_ids_carrier(
            full,
            reference,
            core_start=int(second_start),
            core_end_exclusive=int(second_end),
            carrier_token_ids=token_ids,
            carrier=carrier,
            family=family,
            carrier_metadata=metadata,
        )
        prepared.metadata.update(
            {
                "standard_prompt": prompt_summary,
                "standard_prompt_sha256": sha256_json(prompt_summary),
            }
        )
        sequences[carrier] = prepared

    _validate_matched_readout_counts(sequences)
    return sequences


def _validate_matched_readout_counts(sequences: dict[str, PreparedSequence]) -> None:
    counts = {carrier: len(sequences[carrier].readout_indices) for carrier in CARRIERS}
    if len(set(counts.values())) != 1 or next(iter(counts.values())) <= 0:
        raise RuntimeError(
            f"Carrier readout token counts are not strictly matched: {counts}"
        )
    readout_spans = {
        carrier: tuple(int(item) for item in sequences[carrier].readout_indices)
        for carrier in CARRIERS
    }
    if len(set(readout_spans.values())) != 1:
        raise RuntimeError(
            f"Carrier readout positions are not strictly matched: {readout_spans}"
        )
    shared_readout = next(iter(readout_spans.values()))
    if shared_readout != tuple(range(shared_readout[0], shared_readout[-1] + 1)):
        raise RuntimeError(f"Carrier readout span is not contiguous: {shared_readout}")
    sequence_lengths = {
        carrier: int(sequences[carrier].inputs["input_ids"].shape[-1])
        for carrier in CARRIERS
    }
    prefill_lengths = {
        carrier: int(sequences[carrier].prefill_len) for carrier in CARRIERS
    }
    readout_to_boundary = {
        carrier: int(sequences[carrier].prefill_len)
        - int(sequences[carrier].readout_indices[-1])
        - 1
        for carrier in CARRIERS
    }
    decode_prefix_lengths = {
        carrier: sequence_lengths[carrier] - prefill_lengths[carrier]
        for carrier in CARRIERS
    }
    invariants = {
        "sequence_lengths": sequence_lengths,
        "prefill_lengths": prefill_lengths,
        "readout_to_boundary": readout_to_boundary,
        "decode_prefix_lengths": decode_prefix_lengths,
    }
    for name, values in invariants.items():
        if len(set(values.values())) != 1:
            raise RuntimeError(f"Carrier structural match failed for {name}: {values}")
    for carrier in CARRIERS:
        seq = sequences[carrier]
        if max(seq.readout_indices) >= seq.prefill_len:
            raise RuntimeError(
                f"Readout span crosses decode boundary for {carrier}: "
                f"{seq.readout_indices[-1]} >= {seq.prefill_len}"
            )


def _carrier_masks(
    seq_len: int, prefill_len: int, readout_indices: list[int]
) -> tuple[torch.Tensor, dict[str, Any]]:
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    readout = sorted(set(int(item) for item in readout_indices))
    if not readout or readout[-1] >= prefill_len:
        raise RuntimeError(f"Invalid readout indices: {readout} prefill={prefill_len}")
    readout_tensor = torch.tensor(readout, dtype=torch.long)

    aware = causal.clone()
    aware[prefill_len:, :prefill_len] = False
    aware[prefill_len:, readout_tensor] = True

    no_write = aware.clone()
    for query in readout:
        no_write[query, :prefill_len] = False
        visible_readout = [key for key in readout if key <= query]
        no_write[query, visible_readout] = True

    position_null = causal.clone()
    position_null[prefill_len:, :prefill_len] = False
    masks = torch.stack([aware, no_write, position_null], dim=0)
    checks = {
        "causal_shape": [seq_len, seq_len],
        "prefill_len": prefill_len,
        "readout_indices": readout,
        "readout_count": len(readout),
        "aware_decode_prefill_visible": int(aware[prefill_len:, :prefill_len].sum()),
        "aware_expected_per_decode_row": len(readout),
        "no_write_readout_to_pre_readout_visible": int(
            no_write[readout_tensor, : readout[0]].sum()
        ),
        "position_null_decode_prefill_visible": int(
            position_null[prefill_len:, :prefill_len].sum()
        ),
        "no_future": {
            name: bool(torch.equal(mask & ~causal, torch.zeros_like(causal)))
            for name, mask in zip(MASK_CONDITIONS, masks)
        },
    }
    decode_rows = seq_len - prefill_len
    if checks["aware_decode_prefill_visible"] != decode_rows * len(readout):
        raise RuntimeError(f"Aware mask exposes unexpected prefill keys: {checks}")
    if checks["no_write_readout_to_pre_readout_visible"] != 0:
        raise RuntimeError(
            f"No-write carrier can still read pre-carrier keys: {checks}"
        )
    if checks["position_null_decode_prefill_visible"] != 0:
        raise RuntimeError(f"Position-null mask exposes prefill keys: {checks}")
    if not all(checks["no_future"].values()):
        raise RuntimeError(f"Carrier mask exposes future keys: {checks}")
    return masks, checks


def _additive_mask(
    allowed: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    out = torch.full(
        allowed.shape,
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    out.masked_fill_(allowed.to(device=device), 0)
    return out.unsqueeze(1)


def _qwen_per_image_features(
    model: Any, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise RuntimeError(f"Unexpected Qwen image grid shape: {image_grid_thw.shape}")
    if torch.any(image_grid_thw <= 0):
        raise RuntimeError(f"Qwen image grid must be positive: {image_grid_thw}")
    patch_counts = [int(value) for value in image_grid_thw.prod(dim=-1).tolist()]
    if sum(patch_counts) != int(pixel_values.shape[0]):
        raise RuntimeError(
            "Qwen image grids do not exactly consume pixel_values: "
            f"{patch_counts} vs {pixel_values.shape[0]}"
        )
    merge_area = int(model.config.vision_config.spatial_merge_size) ** 2
    if any(count % merge_area for count in patch_counts):
        raise RuntimeError(
            f"Qwen image patch counts are not divisible by {merge_area}: "
            f"{patch_counts}"
        )
    pixel_parts = torch.split(pixel_values, patch_counts, dim=0)
    feature_parts = []
    for image_idx, pixel_part in enumerate(pixel_parts):
        image_features = model.get_image_features(
            pixel_part, image_grid_thw[image_idx : image_idx + 1]
        )
        if isinstance(image_features, torch.Tensor):
            image_features = (image_features,)
        if len(image_features) != 1:
            raise RuntimeError(
                f"Qwen per-image encoder returned {len(image_features)} feature parts"
            )
        image_feature = image_features[0]
        expected_features = patch_counts[image_idx] // merge_area
        if int(image_feature.shape[0]) != expected_features:
            raise RuntimeError(
                "Qwen per-image feature length mismatch: "
                f"{image_feature.shape[0]} != {expected_features}"
            )
        feature_parts.append(image_feature)
    return tuple(feature_parts)


@torch.no_grad()
def _prepare_qwen_state(model: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    input_ids = inputs["input_ids"]
    embeds = model.get_input_embeddings()(input_ids)
    feature_stats = []
    feature_parts: tuple[torch.Tensor, ...] = ()
    core_parts: list[torch.Tensor] = []
    scatter_max_abs_diff = 0.0
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        feature_parts = _qwen_per_image_features(
            model, pixel_values, inputs["image_grid_thw"]
        )
        feature_parts = tuple(
            part.to(device=embeds.device, dtype=embeds.dtype) for part in feature_parts
        )
        features = torch.cat(feature_parts, dim=0)
        mask = input_ids == int(model.config.image_token_id)
        if int(mask.sum()) != int(features.shape[0]):
            raise RuntimeError(
                f"Qwen image token/feature mismatch: {int(mask.sum())} != {features.shape[0]}"
            )
        embeds = embeds.masked_scatter(
            mask.unsqueeze(-1).expand_as(embeds),
            features,
        )
        scatter_max_abs_diff = float((embeds[mask] - features).abs().max())
        feature_stats = [_tensor_runtime_stats(part) for part in feature_parts]
        flat_core = embeds[mask]
        offset = 0
        for part in feature_parts:
            length = int(part.shape[0])
            core_parts.append(flat_core[offset : offset + length])
            offset += length
        if offset != int(flat_core.shape[0]):
            raise RuntimeError(
                "Qwen visual feature split did not consume the image core"
            )
    position_ids, rope_deltas = _qwen_position_ids(model, inputs)
    raw_tensors = {
        "inputs_embeds": embeds,
        "position_ids": position_ids,
        "rope_deltas": rope_deltas,
    }
    for idx, (feature, core) in enumerate(zip(feature_parts, core_parts)):
        raw_tensors[f"visual_feature_{idx}"] = feature
        raw_tensors[f"visual_core_{idx}"] = core
    return {
        "inputs_embeds": embeds,
        "position_ids": position_ids,
        "raw_tensors": raw_tensors,
        "public_meta": {
            "position_ids": position_ids,
            "rope_deltas": rope_deltas,
            "inputs_embeds_sha256": _tensor_sha256(embeds),
            "visual_features": feature_stats,
            "visual_scatter_max_abs_diff": scatter_max_abs_diff,
            "vision_encode_mode": "per_image_batch_one",
        },
    }


def _qwen_position_ids(
    model: Any, inputs: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    position_ids, rope_deltas = model.model.get_rope_index(
        inputs.get("input_ids"),
        inputs.get("image_grid_thw"),
        inputs.get("video_grid_thw"),
        second_per_grid_ts=inputs.get("second_per_grid_ts"),
        attention_mask=inputs.get("attention_mask"),
    )
    return position_ids, rope_deltas


def _run_qwen_allowed(
    model: Any,
    inputs: dict[str, Any],
    allowed: torch.Tensor,
    state: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    state = state or _prepare_qwen_state(model, inputs)
    embeds = state["inputs_embeds"]
    position_ids = state["position_ids"]
    logits = []
    for mask in allowed:
        attention = _additive_mask(mask.unsqueeze(0), embeds.dtype, embeds.device)
        with torch.inference_mode():
            outputs = model.model.language_model(
                input_ids=None,
                inputs_embeds=embeds,
                position_ids=position_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
            logits.append(model.lm_head(outputs.last_hidden_state[:, -1, :])[0])
    return torch.stack(logits), state["public_meta"], state


def _run_qwen_standard(model: Any, inputs: dict[str, Any]) -> torch.Tensor:
    model.model.rope_deltas = None
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False, return_dict=True)
    return outputs.logits[0, -1, :]


def _run_qwen_embedded_standard(
    model: Any, inputs: dict[str, Any], state: dict[str, Any]
) -> torch.Tensor:
    with torch.inference_mode():
        outputs = model.model.language_model(
            input_ids=None,
            inputs_embeds=state["inputs_embeds"],
            position_ids=state["position_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
    return logits[0]


def _has_nonempty_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    if isinstance(value, (list, tuple)):
        return any(_has_nonempty_tensor(item) for item in value)
    return False


def _minicpm_per_image_vision_states(
    model: Any, inputs: dict[str, Any]
) -> list[torch.Tensor]:
    pixels = inputs["pixel_values"][0]
    target_sizes = inputs["tgt_sizes"][0]
    image_count = len(inputs["image_bound"][0])
    if len(pixels) != image_count or len(target_sizes) != image_count:
        raise RuntimeError("MiniCPM image tensors and bounds are misaligned")
    parts = []
    for image_idx in range(image_count):
        image_inputs = {
            "pixel_values": [pixels[image_idx : image_idx + 1]],
            "tgt_sizes": [target_sizes[image_idx : image_idx + 1]],
        }
        image_state = model.get_vision_embedding(image_inputs)[0]
        if image_state.shape[:2] != (1, int(model.config.query_num)):
            raise RuntimeError(
                f"Unexpected MiniCPM per-image state shape: {image_state.shape}"
            )
        parts.append(image_state)
    vision_states = torch.cat(parts, dim=0)
    if int(vision_states.shape[0]) != image_count:
        raise RuntimeError("MiniCPM per-image states changed image order/count")
    return [vision_states]


def _minicpm_v_per_image_vision_states(
    model: Any, inputs: dict[str, Any]
) -> list[torch.Tensor]:
    pixels = inputs["pixel_values"][0]
    target_sizes = inputs["tgt_sizes"][0]
    temporal_ids = inputs["temporal_ids"][0]
    image_count = len(inputs["image_bound"][0])
    if not (
        len(pixels) == len(target_sizes) == len(temporal_ids) == image_count
    ):
        raise RuntimeError("MiniCPM-V image tensors, temporal IDs, and bounds misalign")
    parts = []
    query_num = int(model.config.query_num)
    for image_idx in range(image_count):
        image_inputs = {
            "input_ids": torch.zeros(
                (1, query_num), dtype=torch.long, device=model.device
            ),
            "image_bound": [
                torch.tensor([[0, query_num]], dtype=torch.long, device=model.device)
            ],
            "pixel_values": [pixels[image_idx : image_idx + 1]],
            "tgt_sizes": [target_sizes[image_idx : image_idx + 1]],
            "temporal_ids": [temporal_ids[image_idx : image_idx + 1]],
        }
        _, vision_states = model.get_vllm_embedding(image_inputs)
        image_state = vision_states[0]
        if image_state.shape[:2] != (1, query_num):
            raise RuntimeError(
                f"Unexpected MiniCPM-V per-image state shape: {image_state.shape}"
            )
        parts.append(image_state)
    vision_states = torch.cat(parts, dim=0)
    if int(vision_states.shape[0]) != image_count:
        raise RuntimeError("MiniCPM-V per-image states changed image order/count")
    return [vision_states]


def _prepare_minicpm_state(
    model: Any, inputs: dict[str, Any], *, apply_omni: bool
) -> dict[str, Any]:
    attention_backend = getattr(model.llm.config, "_attn_implementation", None)
    if attention_backend not in {"sdpa", "eager"}:
        raise RuntimeError(
            "MiniCPM carrier masks require sdpa/eager, got " f"{attention_backend!r}"
        )
    embedding_inputs = dict(inputs)
    embedding_inputs["vision_hidden_states"] = (
        _minicpm_per_image_vision_states(model, inputs)
        if apply_omni
        else _minicpm_v_per_image_vision_states(model, inputs)
    )
    embeds, vision_states = model.get_vllm_embedding(embedding_inputs)
    audio_features = inputs.get("audio_features")
    if _has_nonempty_tensor(audio_features):
        raise RuntimeError(
            "Readout carrier probe does not support MiniCPM audio inputs"
        )
    scatter_max_abs_diff = 0.0
    visual_stats = []
    visual_features: list[torch.Tensor] = []
    if vision_states and isinstance(vision_states[0], torch.Tensor):
        sample_states = vision_states[0]
        bounds = inputs["image_bound"][0]
        if len(bounds) != int(sample_states.shape[0]):
            raise RuntimeError(
                f"MiniCPM vision-state/bound mismatch: {sample_states.shape} vs {bounds}"
            )
        for image_idx, bound in enumerate(bounds):
            start, end = (int(bound[0]), int(bound[1]))
            expected = sample_states[image_idx].to(embeds.dtype)
            actual = embeds[0, start:end]
            if actual.shape != expected.shape:
                raise RuntimeError(
                    f"MiniCPM scatter shape mismatch: {actual.shape} != {expected.shape}"
                )
            scatter_max_abs_diff = max(
                scatter_max_abs_diff, float((actual - expected).abs().max())
            )
            visual_stats.append(_tensor_runtime_stats(expected))
            visual_features.append(expected)
    pre_omni_embeds = embeds.clone()
    if apply_omni:
        embeds = model.get_omni_embedding(
            inputs,
            input_embeddings=embeds,
            chunk_length=model.config.audio_chunk_length,
        )
        embedding_postprocess = "applied_no_audio_identity"
    else:
        embedding_postprocess = "not_applicable"
    positions = inputs["position_ids"].long()
    if not _has_nonempty_tensor(audio_features):
        omni_diff = float((embeds - pre_omni_embeds).abs().max())
        if omni_diff > 1e-5:
            raise RuntimeError(
                f"MiniCPM no-audio omni embedding changed the sequence: {omni_diff}"
            )
    else:
        omni_diff = math.nan
    cache_position = torch.arange(
        embeds.shape[1], dtype=torch.long, device=embeds.device
    )
    raw_tensors = {
        "inputs_embeds": embeds,
        "position_ids": positions,
        "image_bound": inputs["image_bound"][0],
        "cache_position": cache_position,
        "pre_omni_embeddings": pre_omni_embeds,
        "post_omni_embeddings": embeds,
    }
    bounds = inputs["image_bound"][0]
    for idx, feature in enumerate(visual_features):
        start, end = (int(bounds[idx][0]), int(bounds[idx][1]))
        raw_tensors[f"visual_feature_{idx}"] = feature
        raw_tensors[f"visual_core_{idx}"] = embeds[0, start:end]
    return {
        "inputs_embeds": embeds,
        "position_ids": positions,
        "cache_position": cache_position,
        "raw_tensors": raw_tensors,
        "public_meta": {
            "position_ids": positions,
            "inputs_embeds_sha256": _tensor_sha256(embeds),
            "visual_features": visual_stats,
            "visual_scatter_max_abs_diff": scatter_max_abs_diff,
            "attention_backend": attention_backend,
            "no_audio_omni_max_abs_diff": omni_diff,
            "embedding_postprocess": embedding_postprocess,
            "vision_encode_mode": "per_image_batch_one",
        },
    }


def _run_minicpm_allowed(
    model: Any,
    inputs: dict[str, Any],
    allowed: torch.Tensor,
    state: dict[str, Any] | None = None,
    *,
    apply_omni: bool,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    state = state or _prepare_minicpm_state(model, inputs, apply_omni=apply_omni)
    embeds = state["inputs_embeds"]
    positions = state["position_ids"]
    logits = []
    for mask in allowed:
        attention = _additive_mask(mask.unsqueeze(0), embeds.dtype, embeds.device)
        with torch.inference_mode():
            outputs = model.llm(
                input_ids=None,
                inputs_embeds=embeds,
                position_ids=positions,
                attention_mask=attention,
                cache_position=state["cache_position"],
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
        logits.append(outputs.logits[0, -1, :])
    return torch.stack(logits), state["public_meta"], state


def _run_minicpm_standard(
    model: Any,
    inputs: dict[str, Any],
    state: dict[str, Any] | None = None,
    *,
    apply_omni: bool,
) -> torch.Tensor:
    state = state or _prepare_minicpm_state(model, inputs, apply_omni=apply_omni)
    embeds = state["inputs_embeds"]
    with torch.inference_mode():
        outputs = model.llm(
            input_ids=None,
            inputs_embeds=embeds,
            position_ids=state["position_ids"],
            attention_mask=inputs["attention_mask"],
            cache_position=state["cache_position"],
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    return outputs.logits[0, -1, :]


def _llm_attention_config(wrapper: Any, family: str) -> Any:
    if family == "qwen25vl":
        return wrapper.model.model.language_model.config
    return wrapper.model.llm.config


@contextlib.contextmanager
def _attention_backend(config: Any, backend: str):
    previous = config._attn_implementation
    config._attn_implementation = backend
    try:
        yield
    finally:
        config._attn_implementation = previous


def _tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().cpu().contiguous()
    return hashlib.sha256(
        cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _tensor_dump_numpy(value: torch.Tensor) -> np.ndarray:
    cpu = value.detach().cpu().contiguous()
    if cpu.is_floating_point():
        cpu = cpu.float()
    return cpu.numpy()


def _tensor_runtime_stats(value: torch.Tensor) -> dict[str, Any]:
    floating = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": _tensor_sha256(value),
        "float32_sha256": _tensor_sha256(floating),
        "l2_norm": float(torch.linalg.vector_norm(floating)),
        "max_abs": float(floating.abs().max()) if floating.numel() else 0.0,
    }


def _nested_tensor_summary_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_nested_tensor_summary_value(item) for item in value]
    return value


def _nested_input_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
    if isinstance(value, (list, tuple)):
        return [_nested_input_summary(item) for item in value]
    return value


def _input_summaries(inputs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            out[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
        elif isinstance(value, (list, tuple)):
            out[key] = {
                "type": type(value).__name__,
                "value": _nested_input_summary(value),
            }
        else:
            out[key] = {"type": type(value).__name__, "value": value}
    return out


def _candidate_values(logits: torch.Tensor, plan: dict[str, Any]) -> dict[str, float]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return {
        label: float(log_probs[int(token_id)].detach().cpu())
        for label, token_id in plan["candidate_token_ids"].items()
    }


def _candidate_parity(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    plan: dict[str, Any],
    *,
    atol: float = LOGIT_PARITY_ATOL,
    rtol: float = LOGIT_PARITY_RTOL,
) -> dict[str, Any]:
    reference = reference.detach().float()
    candidate = candidate.detach().float()
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise RuntimeError(
            f"Parity logits must be equal one-dimensional vocab vectors: "
            f"{tuple(reference.shape)} vs {tuple(candidate.shape)}"
        )
    full_diff = (reference - candidate).abs()
    labels = list(plan["candidate_token_ids"])
    token_ids = [int(plan["candidate_token_ids"][label]) for label in labels]
    reference_values = reference[token_ids].detach().float()
    candidate_values = candidate[token_ids].detach().float()
    diff = (reference_values - candidate_values).abs()
    reference_prediction = labels[int(reference_values.argmax())]
    candidate_prediction = labels[int(candidate_values.argmax())]
    max_abs_diff = float(diff.max())
    return {
        "reference_candidate_logits": {
            label: float(value)
            for label, value in zip(labels, reference_values.detach().cpu().tolist())
        },
        "candidate_candidate_logits": {
            label: float(value)
            for label, value in zip(labels, candidate_values.detach().cpu().tolist())
        },
        "candidate_logit_max_abs_diff": max_abs_diff,
        "candidate_within_strict_atol": max_abs_diff <= CANDIDATE_PARITY_ATOL,
        "candidate_atol": CANDIDATE_PARITY_ATOL,
        "full_vocab_max_abs_diff": float(full_diff.max()),
        "full_vocab_mean_abs_diff": float(full_diff.mean()),
        "full_vocab_p99_abs_diff": float(torch.quantile(full_diff, 0.99)),
        "full_vocab_allclose": bool(
            torch.allclose(reference, candidate, atol=float(atol), rtol=float(rtol))
        ),
        "reference_prediction": reference_prediction,
        "candidate_prediction": candidate_prediction,
        "argmax_equal": reference_prediction == candidate_prediction,
        "atol": float(atol),
        "rtol": float(rtol),
        "within_atol": max_abs_diff <= float(atol),
        "passed": (
            reference_prediction == candidate_prediction
            and max_abs_diff <= CANDIDATE_PARITY_ATOL
            and torch.allclose(reference, candidate, atol=float(atol), rtol=float(rtol))
        ),
    }


def _score_values(values: dict[str, float], answer_key: str) -> dict[str, Any]:
    predicted = max(values, key=values.get)
    wrong = [value for label, value in values.items() if label != answer_key]
    correct_margin = float(values[answer_key] - max(wrong)) if wrong else math.nan
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
    top_margin = float(ordered[0][1] - ordered[1][1]) if len(ordered) > 1 else math.nan
    return {
        "candidate_logprobs": values,
        "predicted_key": predicted,
        "answer_key": answer_key,
        "hit": predicted == answer_key,
        "correct_margin": correct_margin,
        "top_margin": top_margin,
    }


def _extend_prepared(
    sequence: PreparedSequence, prefix_ids: list[int], family: str
) -> PreparedSequence:
    out = copy.copy(sequence)
    out.inputs = _append_ids(
        sequence.inputs,
        prefix_ids,
        recompute_position_ids=_is_minicpm_family(family),
    )
    out.token_roles = list(sequence.token_roles) + ["answer_prefix"] * len(prefix_ids)
    out.metadata = copy.deepcopy(sequence.metadata)
    out.artifact_images = dict(sequence.artifact_images)
    return out


def _prepare_literal_blind(
    wrapper: Any,
    family: str,
    tokenizer: Any,
    prefix_ids: list[int],
) -> PreparedSequence:
    device = next(wrapper.model.parameters()).device
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    if _is_minicpm_family(family):
        inputs["position_ids"] = torch.arange(
            input_ids.shape[-1], dtype=torch.long, device=device
        ).unsqueeze(0)
    text = tokenizer.decode(
        prefix_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return PreparedSequence(
        inputs=inputs,
        prefill_len=0,
        readout_indices=[],
        prompt_text="",
        generation_text=text,
        token_roles=["answer_prefix"] * len(prefix_ids),
        metadata={
            "carrier": "blind",
            "carrier_kind": "literal_context_free_answer_prefix",
        },
    )


def _corrupt_prefix_state(
    state: dict[str, Any], sequence: PreparedSequence
) -> tuple[dict[str, Any], dict[str, Any]]:
    first_readout = min(sequence.readout_indices)
    positions = list(range(first_readout))
    if not positions:
        raise RuntimeError(
            "Could not locate pre-readout embeddings for corruption smoke"
        )
    out = dict(state)
    embeds = state["inputs_embeds"].clone()
    before = embeds[:, positions, :].clone()
    embeds[:, positions, :] = -before + 0.03125
    out["inputs_embeds"] = embeds
    out["public_meta"] = copy.deepcopy(state["public_meta"])
    out["public_meta"]["inputs_embeds_sha256"] = _tensor_sha256(embeds)
    return out, {
        "positions": positions,
        "before_sha256": _tensor_sha256(before),
        "after_sha256": _tensor_sha256(embeds[:, positions, :]),
        "position_count": len(positions),
    }


def _run_allowed(
    family: str,
    wrapper: Any,
    inputs: dict[str, Any],
    masks: torch.Tensor,
    state: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    if family == "qwen25vl":
        return _run_qwen_allowed(wrapper.model, inputs, masks, state=state)
    if _is_minicpm_family(family):
        return _run_minicpm_allowed(
            wrapper.model,
            inputs,
            masks,
            state=state,
            apply_omni=family == "minicpmo45",
        )
    raise AssertionError(family)


def _run_standard(
    family: str,
    wrapper: Any,
    inputs: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> torch.Tensor:
    if family == "qwen25vl":
        return _run_qwen_standard(wrapper.model, inputs)
    if _is_minicpm_family(family):
        return _run_minicpm_standard(
            wrapper.model,
            inputs,
            state=state,
            apply_omni=family == "minicpmo45",
        )
    raise AssertionError(family)


def _run_embedded_standard(
    family: str,
    wrapper: Any,
    inputs: dict[str, Any],
    state: dict[str, Any],
) -> torch.Tensor:
    if family == "qwen25vl":
        return _run_qwen_embedded_standard(wrapper.model, inputs, state)
    if _is_minicpm_family(family):
        return _run_minicpm_standard(
            wrapper.model,
            inputs,
            state=state,
            apply_omni=family == "minicpmo45",
        )
    raise AssertionError(family)


def _run_literal_blind(
    family: str, wrapper: Any, sequence: PreparedSequence
) -> torch.Tensor:
    if family == "qwen25vl":
        return _run_qwen_standard(wrapper.model, sequence.inputs)
    if _is_minicpm_family(family):
        inputs = sequence.inputs
        with torch.inference_mode():
            outputs = wrapper.model.llm(
                input_ids=inputs["input_ids"],
                position_ids=inputs["position_ids"],
                attention_mask=inputs["attention_mask"],
                cache_position=torch.arange(
                    inputs["input_ids"].shape[-1],
                    device=inputs["input_ids"].device,
                ),
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
        return outputs.logits[0, -1, :]
    raise AssertionError(family)


def _prepare_sequences(
    family: str, wrapper: Any, dataset: Any, dataset_name: str, row: Any
) -> dict[str, PreparedSequence]:
    if family == "qwen25vl":
        return _prepare_qwen_sequences(wrapper, dataset, dataset_name, row)
    if _is_minicpm_family(family):
        return _prepare_minicpm_sequences(
            wrapper, dataset, dataset_name, row, family=family
        )
    raise AssertionError(family)


def _dump_token_table(
    sequence: PreparedSequence, tokenizer: Any
) -> list[dict[str, Any]]:
    ids = sequence.inputs["input_ids"][0].detach().cpu().tolist()
    if len(ids) != len(sequence.token_roles):
        raise RuntimeError(
            f"Token role length mismatch: {len(ids)} != {len(sequence.token_roles)}"
        )
    return [
        {
            "position": idx,
            "token_id": int(token_id),
            "token": tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "role": sequence.token_roles[idx],
            "is_readout": idx in set(sequence.readout_indices),
        }
        for idx, token_id in enumerate(ids)
    ]


def _runtime_identity(wrapper: Any, family: str) -> dict[str, Any]:
    import transformers

    llm_config = _llm_attention_config(wrapper, family)
    if family == "qwen25vl":
        attention_backend = getattr(wrapper.model.config, "_attn_implementation", None)
    else:
        attention_backend = getattr(
            wrapper.model.llm.config, "_attn_implementation", None
        )
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "attention_backend": attention_backend,
        "model_class": wrapper.model.__class__.__name__,
        "processor_class": wrapper.processor.__class__.__name__,
        "tokenizer_class": (
            wrapper.processor.tokenizer.__class__.__name__
            if family == "qwen25vl"
            else wrapper.tokenizer.__class__.__name__
        ),
        "hidden_size": int(llm_config.hidden_size),
        "num_hidden_layers": int(llm_config.num_hidden_layers),
        "hf_device_map": {
            str(key): str(value)
            for key, value in getattr(wrapper.model, "hf_device_map", {}).items()
        },
    }


def score_record(
    wrapper: Any,
    family: str,
    dataset: Any,
    dataset_name: str,
    row: Any,
    manifest_record: dict[str, Any],
    blind_cache: dict[tuple[str, ...], dict[str, Any]],
    runtime_identity: dict[str, Any],
    provenance: dict[str, Any],
    *,
    dump_dir: Path | None,
    diagnostics: bool,
) -> dict[str, Any]:
    sequences = _prepare_sequences(
        wrapper=wrapper,
        family=family,
        dataset=dataset,
        dataset_name=dataset_name,
        row=row,
    )
    tokenizer = (
        wrapper.processor.tokenizer if family == "qwen25vl" else wrapper.tokenizer
    )
    labels = list(manifest_record["choice_labels"])
    plan = candidate_token_plan(tokenizer, labels)
    prefix_ids = list(plan["forced_prefix_ids"])
    answer_key = str(manifest_record["answer_key"])
    target_n = len(sequences["blank_image"].readout_indices)

    cache_key = tuple(labels)
    if cache_key not in blind_cache:
        blind = _prepare_literal_blind(wrapper, family, tokenizer, prefix_ids)
        blind_logits = _run_literal_blind(family, wrapper, blind)
        blind_cache[cache_key] = {
            "logits": blind_logits.detach().float().cpu(),
            "sequence": blind,
        }
    blind_values = _candidate_values(blind_cache[cache_key]["logits"], plan)
    blind_score = _score_values(blind_values, answer_key)

    full = _extend_prepared(sequences["full"], prefix_ids, family)
    full_dump_state = None
    if dump_dir is not None:
        if family == "qwen25vl":
            full_dump_state = _prepare_qwen_state(wrapper.model, full.inputs)
        elif _is_minicpm_family(family):
            full_dump_state = _prepare_minicpm_state(
                wrapper.model,
                full.inputs,
                apply_omni=family == "minicpmo45",
            )
    full_logits = _run_standard(family, wrapper, full.inputs, state=full_dump_state)
    full_score = _score_values(_candidate_values(full_logits, plan), answer_key)

    carriers: dict[str, Any] = {}
    raw_logits: dict[str, torch.Tensor] = {
        "blind": blind_cache[cache_key]["logits"],
        "full": full_logits.detach().float().cpu(),
    }
    raw_masks: dict[str, torch.Tensor] = {}
    runtime_meta: dict[str, Any] = {}
    raw_runtime_tensors: dict[str, np.ndarray] = {}
    if full_dump_state is not None:
        runtime_meta["full"] = {
            key: (
                value.detach().cpu().tolist()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in full_dump_state["public_meta"].items()
        }
        for tensor_name in ("visual_feature_0", "visual_core_0"):
            raw_runtime_tensors[f"full__{tensor_name}"] = _tensor_dump_numpy(
                full_dump_state["raw_tensors"][tensor_name]
            )
    diagnostic_logits: dict[str, np.ndarray] = {}
    for carrier in CARRIERS:
        sequence = _extend_prepared(sequences[carrier], prefix_ids, family)
        seq_len = int(sequence.inputs["input_ids"].shape[-1])
        masks, checks = _carrier_masks(
            seq_len, sequence.prefill_len, sequence.readout_indices
        )
        logits, model_meta, state = _run_allowed(
            family, wrapper, sequence.inputs, masks
        )
        carrier_scores = {
            condition: _score_values(_candidate_values(logits[idx], plan), answer_key)
            for idx, condition in enumerate(MASK_CONDITIONS)
        }
        carriers[carrier] = {
            "token_count": len(sequence.readout_indices),
            "readout_indices": sequence.readout_indices,
            "prefill_len": sequence.prefill_len,
            "sequence_len": seq_len,
            "mask_checks": checks,
            "scores": carrier_scores,
            "metadata": sequence.metadata,
        }
        raw_logits[carrier] = logits.detach().float().cpu()
        raw_masks[carrier] = masks.cpu()
        runtime_meta[carrier] = {
            key: (
                value.detach().cpu().tolist()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in model_meta.items()
        }
        for tensor_name, tensor_value in state["raw_tensors"].items():
            if (
                tensor_name
                in {
                    "inputs_embeds",
                    "pre_omni_embeddings",
                    "post_omni_embeddings",
                }
                and not diagnostics
            ):
                continue
            raw_runtime_tensors[f"{carrier}__{tensor_name}"] = _tensor_dump_numpy(
                tensor_value
            )

        if diagnostics and carrier in TEXT_CARRIERS:
            embedding_layer = (
                wrapper.model.get_input_embeddings()
                if family == "qwen25vl"
                else wrapper.model.llm.get_input_embeddings()
            )
            readout_ids = sequence.inputs["input_ids"][:, sequence.readout_indices]
            with torch.no_grad():
                expected_text_core = embedding_layer(readout_ids)
                if _is_minicpm_family(family) and hasattr(
                    wrapper.model.llm.config, "scale_emb"
                ):
                    expected_text_core = (
                        expected_text_core * wrapper.model.llm.config.scale_emb
                    )
            raw_runtime_tensors[f"{carrier}__expected_text_core"] = _tensor_dump_numpy(
                expected_text_core
            )

        if diagnostics:
            corrupted_state, corruption = _corrupt_prefix_state(state, sequence)
            corrupted_logits, _, _ = _run_allowed(
                family,
                wrapper,
                sequence.inputs,
                masks,
                state=corrupted_state,
            )
            no_write_diff = (logits[1].float() - corrupted_logits[1].float()).abs()
            null_diff = (logits[2].float() - corrupted_logits[2].float()).abs()
            aware_diff = (logits[0].float() - corrupted_logits[0].float()).abs()

            causal = torch.tril(
                torch.ones((seq_len, seq_len), dtype=torch.bool)
            ).unsqueeze(0)
            with _attention_backend(_llm_attention_config(wrapper, family), "eager"):
                manual_causal_logits, _, _ = _run_allowed(
                    family, wrapper, sequence.inputs, causal, state=state
                )
                standard_causal_logits = _run_embedded_standard(
                    family, wrapper, sequence.inputs, state
                )
            causal_parity = _candidate_parity(
                standard_causal_logits,
                manual_causal_logits[0],
                plan,
            )
            batch_single_parity = {}
            single_logits_by_condition = {}
            for condition_idx, condition in enumerate(MASK_CONDITIONS):
                single_logits, _, _ = _run_allowed(
                    family,
                    wrapper,
                    sequence.inputs,
                    masks[condition_idx : condition_idx + 1],
                    state=state,
                )
                batch_single_parity[condition] = _candidate_parity(
                    logits[condition_idx], single_logits[0], plan
                )
                single_logits_by_condition[condition] = single_logits[0]
            diagnostic_logits[f"{carrier}__original_batch"] = _tensor_dump_numpy(logits)
            diagnostic_logits[f"{carrier}__corrupted_batch"] = _tensor_dump_numpy(
                corrupted_logits
            )
            diagnostic_logits[f"{carrier}__manual_causal"] = _tensor_dump_numpy(
                manual_causal_logits[0]
            )
            diagnostic_logits[f"{carrier}__standard_causal"] = _tensor_dump_numpy(
                standard_causal_logits
            )
            for condition, single_logits in single_logits_by_condition.items():
                diagnostic_logits[f"{carrier}__single_{condition}"] = (
                    _tensor_dump_numpy(single_logits)
                )
            carriers[carrier]["diagnostics"] = {
                "prefix_embedding_corruption": corruption,
                "aware_full_vocab_max_abs_diff": float(aware_diff.max()),
                "no_write_full_vocab_max_abs_diff": float(no_write_diff.max()),
                "position_null_full_vocab_max_abs_diff": float(null_diff.max()),
                "aware_corruption_effect_observed": bool(
                    float(aware_diff.max()) > CORRUPTION_POSITIVE_MIN
                ),
                "no_write_invariant_atol_1e_5": bool(
                    float(no_write_diff.max()) <= BLOCKED_PREFIX_ATOL
                ),
                "position_null_invariant_atol_1e_5": bool(
                    float(null_diff.max()) <= BLOCKED_PREFIX_ATOL
                ),
                "same_state_standard_2d_vs_manual_causal_4d": causal_parity,
                "batch_vs_single_4d": batch_single_parity,
            }

    record = {
        "schema": RECORD_SCHEMA,
        "dataset": dataset_name,
        "row_position": int(manifest_record["row_position"]),
        "sample_index": str(manifest_record["sample_index"]),
        "shard": int(manifest_record["shard"]),
        "answer_key": answer_key,
        "choice_labels": labels,
        "candidate_plan": plan,
        "matched_readout_token_count": target_n,
        "runtime_identity": runtime_identity,
        "cuda_memory": {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "entry_environment": {key: os.environ.get(key) for key in ENTRY_ENV_KEYS},
        "provenance": provenance,
        "blind": blind_score,
        "full": full_score,
        "carriers": carriers,
    }

    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        artifact_sequences = {}
        dump_sequences = {
            "blind": blind_cache[cache_key]["sequence"],
            "full": full,
            **{
                name: _extend_prepared(sequences[name], prefix_ids, family)
                for name in CARRIERS
            },
        }
        for name, sequence in dump_sequences.items():
            artifact_sequences[name] = {
                "prefill_len": sequence.prefill_len,
                "readout_indices": sequence.readout_indices,
                "prompt_text": sequence.prompt_text,
                "generation_text": sequence.generation_text,
                "metadata": sequence.metadata,
                "token_table": _dump_token_table(sequence, tokenizer),
                "inputs": _input_summaries(sequence.inputs),
            }
            np.save(
                dump_dir / f"{name}_input_ids.npy",
                sequence.inputs["input_ids"].detach().cpu().numpy(),
            )
            if name in CARRIERS:
                np.save(dump_dir / f"{name}_allowed_masks.npy", raw_masks[name].numpy())
            for image_name, image in sequence.artifact_images.items():
                image.save(dump_dir / f"{image_name}.png")
        np.savez_compressed(
            dump_dir / "last_token_logits.npz",
            **{name: value.numpy() for name, value in raw_logits.items()},
        )
        np.savez_compressed(
            dump_dir / "runtime_tensors.npz",
            **raw_runtime_tensors,
        )
        if diagnostic_logits:
            np.savez_compressed(
                dump_dir / "diagnostic_logits.npz",
                **diagnostic_logits,
            )
        artifact = {
            **record,
            "model_family": family,
            "sequences": artifact_sequences,
            "runtime_meta": runtime_meta,
            "raw_files": {
                "logits": "last_token_logits.npz",
                "runtime_tensors": "runtime_tensors.npz",
                "diagnostic_logits": (
                    "diagnostic_logits.npz" if diagnostic_logits else None
                ),
                "masks": {
                    carrier: f"{carrier}_allowed_masks.npy" for carrier in CARRIERS
                },
                "input_ids": {
                    name: f"{name}_input_ids.npy"
                    for name in ("blind", "full", *CARRIERS)
                },
                "images": {carrier: f"{carrier}.png" for carrier in VISUAL_CARRIERS}
                | {
                    "source_readout_image": "source_readout_image.png",
                },
            },
        }
        write_json(dump_dir / "artifact.json", artifact)
    return record


def _manifest_provenance(
    manifest: dict[str, Any], manifest_path: str | Path
) -> dict[str, Any]:
    return {
        "manifest_sha256": sha256_file(Path(manifest_path).resolve()),
        "manifest_schema": manifest["schema"],
        "manifest_records_sha256": manifest["records_sha256"],
        "implementation_sha256": manifest["implementation_sha256"],
        "repo_commit": manifest["repo_commit"],
        "model_key": manifest["model_key"],
        "model_family": manifest["model_family"],
        "model_identity_sha256": manifest["model_identity_sha256"],
        "source_data_sha256": manifest["source_data_sha256"],
        "matrix_config_sha256": manifest["matrix_config"]["sha256"],
        "models_config_sha256": manifest["models_config"]["sha256"],
    }


def _validated_dataset_row(
    dataset_name: str,
    dataset: Any,
    manifest_record: dict[str, Any],
) -> Any:
    key = _record_key(manifest_record)
    row = dataset.data.iloc[int(manifest_record["row_position"])]
    if str(row["index"]) != str(manifest_record["sample_index"]):
        raise RuntimeError(
            f"Frozen row identity mismatch for {key}: source index={row['index']}"
        )
    expected_labels = [str(item) for item in manifest_record["choice_labels"]]
    observed_labels = (
        row_choice_labels(dataset_name, row)
        if dataset_name == "MMBench_DEV_EN_V11"
        else all_single_choice_labels(dataset_name, row)
    )
    if observed_labels != expected_labels:
        raise RuntimeError(
            f"Frozen choice labels changed for {key}: "
            f"{observed_labels} != {expected_labels}"
        )
    observed_answer = str(row.get("answer", "")).strip()
    if observed_answer != str(manifest_record["answer_key"]):
        raise RuntimeError(f"Frozen answer changed for {key}: {observed_answer!r}")
    observed_option_hash = sha256_json(
        row_option_texts(dataset_name, row, expected_labels)
    )
    if observed_option_hash != manifest_record["option_text_sha256"]:
        raise RuntimeError(
            f"Frozen option text changed for {key}: {observed_option_hash}"
        )
    return row


def run_probe(args: argparse.Namespace) -> int:
    from vlmeval.dataset import build_dataset

    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError(
            f"Unexpected carrier manifest schema: {manifest.get('schema')}"
        )
    _verify_repo_contract(repo_root, manifest)
    if sha256_file(Path(__file__).resolve()) != manifest["implementation_sha256"]:
        raise RuntimeError("Carrier implementation changed after manifest creation")
    _validate_run_contract_attestation(
        args.run_contract_attestation, manifest_path, manifest, args.model_path
    )
    verify_checkpoint_identity_quick(args.model_path, manifest["model_identity"])
    family = _model_family(args.model_key)
    if manifest["model_key"] != args.model_key or manifest["model_family"] != family:
        raise RuntimeError("Manifest/model mismatch")
    datasets_requested = set(_parse_csv(args.datasets))
    if (
        _file_identity(args.matrix_config)["sha256"]
        != manifest["matrix_config"]["sha256"]
    ):
        raise RuntimeError("Runtime matrix config changed after manifest creation")
    if (
        _file_identity(repo_root / "configs" / "models.yaml")["sha256"]
        != manifest["models_config"]["sha256"]
    ):
        raise RuntimeError("Runtime models config changed after manifest creation")
    for dataset_name in datasets_requested:
        source = Path(args.lmu_data).resolve() / f"{dataset_name}.tsv"
        expected_source = manifest["source_data"][dataset_name]
        observed_source = _file_identity(source)
        if (
            int(observed_source["size"]) != int(expected_source["size"])
            or observed_source["sha256"] != expected_source["sha256"]
        ):
            raise RuntimeError(
                f"Runtime dataset source changed for {dataset_name}: {source}"
            )
    records = [
        record
        for record in manifest["records"]
        if record["dataset"] in datasets_requested
        and (args.shard_rank is None or int(record["shard"]) == int(args.shard_rank))
    ]
    if args.one_per_dataset:
        first_by_dataset = {}
        for record in records:
            first_by_dataset.setdefault(str(record["dataset"]), record)
        records = [
            first_by_dataset[name]
            for name in _parse_csv(args.datasets)
            if name in first_by_dataset
        ]
    if args.limit is not None:
        records = records[: int(args.limit)]
    if not records:
        raise RuntimeError("No frozen records matched this run")

    first_dataset = str(records[0]["dataset"])
    env, runner, _ = build_runtime(
        repo_root,
        Path(args.runtime_root).resolve(),
        first_dataset,
        args.model_key,
        args.gpu_id,
        args.model_path,
        args.lmu_data,
        str(Path(args.matrix_config).resolve()),
    )
    registry_name = runner.models[args.model_key].registry_name
    wrapper = _load_probe_model(env, registry_name, family)
    runtime_identity = _runtime_identity(wrapper, family)
    if args.expected_runtime_validation:
        expected_runtime = _load_json(args.expected_runtime_validation).get(
            "runtime_identity"
        )
        if runtime_identity != expected_runtime:
            raise RuntimeError(
                "Full-run runtime identity differs from the accepted smoke: "
                f"{runtime_identity} != {expected_runtime}"
            )
    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = _manifest_provenance(manifest, manifest_path)
    expected_keys = {_record_key(record) for record in records}
    completed: set[tuple[str, int, str]] = set()
    if args.resume and output_path.is_file():
        raw = output_path.read_bytes()
        lines = raw.splitlines(keepends=True)
        parsed_lines: list[dict[str, Any]] = []
        for line_idx, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                parsed_lines.append(json.loads(line))
            except json.JSONDecodeError as exc:
                is_torn_tail = line_idx == len(lines) - 1 and not raw.endswith(b"\n")
                if not (is_torn_tail and args.repair_torn_jsonl):
                    raise RuntimeError(
                        f"Invalid resume JSONL at line {line_idx + 1}; "
                        "use --repair-torn-jsonl only for a truncated final line"
                    ) from exc
                backup = output_path.with_suffix(output_path.suffix + ".torn.bak")
                backup.write_bytes(raw)
                output_path.write_bytes(b"".join(lines[:line_idx]))
        for old_record in parsed_lines:
            key = _record_key(old_record)
            if key not in expected_keys:
                raise RuntimeError(
                    f"Resume record is outside this worker contract: {key}"
                )
            if old_record.get("schema") != RECORD_SCHEMA:
                raise RuntimeError(f"Resume schema mismatch: {key}")
            if old_record.get("provenance") != provenance:
                raise RuntimeError(f"Resume provenance mismatch: {key}")
            if old_record.get("runtime_identity") != runtime_identity:
                raise RuntimeError(f"Resume runtime identity mismatch: {key}")
            if args.shard_rank is not None and int(old_record.get("shard", -1)) != int(
                args.shard_rank
            ):
                raise RuntimeError(f"Resume shard mismatch: {key}")
            if key in completed:
                raise RuntimeError(f"Duplicate resume record: {key}")
            completed.add(key)
    elif output_path.exists():
        output_path.unlink()

    datasets: dict[str, Any] = {}
    blind_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    for ordinal, manifest_record in enumerate(records):
        key = _record_key(manifest_record)
        if key in completed:
            continue
        dataset_name = str(manifest_record["dataset"])
        env, _, _ = build_runtime(
            repo_root,
            Path(args.runtime_root).resolve(),
            dataset_name,
            args.model_key,
            args.gpu_id,
            args.model_path,
            args.lmu_data,
            str(Path(args.matrix_config).resolve()),
        )
        with patched_environ(env):
            if family == "qwen25vl":
                refresh_replay_runtime(wrapper, env)
            if dataset_name not in datasets:
                datasets[dataset_name] = build_dataset(dataset_name)
            dataset = datasets[dataset_name]
            row = _validated_dataset_row(dataset_name, dataset, manifest_record)
            dump_dir = None
            if args.dump_raw_root:
                dump_dir = (
                    Path(args.dump_raw_root).resolve()
                    / dataset_name
                    / f"row{int(manifest_record['row_position'])}_idx{manifest_record['sample_index']}"
                )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            record = score_record(
                wrapper,
                family,
                dataset,
                dataset_name,
                row,
                manifest_record,
                blind_cache,
                runtime_identity,
                provenance,
                dump_dir=dump_dir,
                diagnostics=(
                    bool(args.diagnostics)
                    and (
                        args.diagnostics_limit is None
                        or ordinal < int(args.diagnostics_limit)
                    )
                ),
            )
            torch.cuda.synchronize()
        record["timing_seconds"] = time.perf_counter() - started
        record["shard"] = int(manifest_record["shard"])
        append_jsonl(output_path, record)
        print(
            json.dumps(
                {
                    "event": "carrier_record_complete",
                    "ordinal": ordinal,
                    "total": len(records),
                    "key": key,
                    "seconds": record["timing_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


def _iter_jsonl(root: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _flatten_condition_scores(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {"blind": record["blind"], "full": record["full"]}
    for carrier in CARRIERS:
        for condition in MASK_CONDITIONS:
            out[f"{carrier}_{condition}"] = record["carriers"][carrier]["scores"][
                condition
            ]
    return out


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_json(manifest_path)
    expected_records = [
        record
        for record in manifest["records"]
        if record["dataset"] in set(_parse_csv(args.datasets))
    ]
    expected = {_record_key(record): record for record in expected_records}
    observed: dict[tuple[str, int, str], dict[str, Any]] = {}
    runtime_identity: dict[str, Any] | None = None
    for record in _iter_jsonl(Path(args.input_root).resolve()):
        key = _record_key(record)
        if record.get("schema") != RECORD_SCHEMA:
            raise RuntimeError(f"Prediction schema mismatch: {key}")
        if key in observed:
            raise RuntimeError(f"Duplicate prediction record: {key}")
        if key not in expected:
            raise RuntimeError(f"Unexpected prediction record: {key}")
        if record.get("provenance") != _manifest_provenance(manifest, manifest_path):
            raise RuntimeError(f"Prediction provenance mismatch: {key}")
        if int(record["matched_readout_token_count"]) <= 0:
            raise RuntimeError(f"Invalid readout count: {key}")
        if (
            len({record["carriers"][carrier]["token_count"] for carrier in CARRIERS})
            != 1
        ):
            raise RuntimeError(f"Unmatched carrier counts in prediction: {key}")
        if runtime_identity is None:
            runtime_identity = record.get("runtime_identity")
        elif record.get("runtime_identity") != runtime_identity:
            raise RuntimeError(f"Mixed runtime identities in prediction set: {key}")
        observed[key] = record
    missing = sorted(set(expected) - set(observed))
    if args.require_complete and missing:
        raise RuntimeError(f"Missing {len(missing)} predictions; first={missing[:5]}")

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in observed.values():
        by_dataset[record["dataset"]].append(record)
    rows = []
    summary_datasets = {}
    condition_names = ["blind", "full"] + [
        f"{carrier}_{condition}"
        for carrier in CARRIERS
        for condition in MASK_CONDITIONS
    ]
    for dataset in _parse_csv(args.datasets):
        records = by_dataset.get(dataset, [])
        if not records:
            continue
        scores = {name: [] for name in condition_names}
        margins = {name: [] for name in condition_names}
        token_counts = []
        for record in records:
            flat = _flatten_condition_scores(record)
            for name, score in flat.items():
                scores[name].append(int(bool(score["hit"])))
                margins[name].append(float(score["correct_margin"]))
            token_counts.append(int(record["matched_readout_token_count"]))
        accuracy = {name: float(np.mean(values)) for name, values in scores.items()}
        mean_margin = {name: float(np.mean(values)) for name, values in margins.items()}
        blind = accuracy["blind"]
        full = accuracy["full"]
        gap = full - blind
        row = {"dataset": dataset, "n": len(records), "blind": 100 * blind}
        for carrier in CARRIERS:
            aware = accuracy[f"{carrier}_aware"]
            no_write = accuracy[f"{carrier}_no_write"]
            null = accuracy[f"{carrier}_position_null"]
            row[carrier] = 100 * aware
            row[f"{carrier}_no_write"] = 100 * no_write
            row[f"{carrier}_write_delta"] = 100 * (aware - no_write)
            row[f"{carrier}_position_null"] = 100 * null
            row[f"{carrier}_recovery"] = (aware - blind) / gap if gap > 0 else math.nan
        row["full"] = 100 * full
        rows.append(row)
        summary_datasets[dataset] = {
            "n": len(records),
            "accuracy": accuracy,
            "mean_correct_margin": mean_margin,
            "readout_token_count_histogram": dict(
                sorted(Counter(token_counts).items())
            ),
        }

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "topic-image-replay/readout-random-carrier-summary/v1",
        "manifest_records_sha256": manifest["records_sha256"],
        "expected_records": len(expected),
        "observed_records": len(observed),
        "missing_records": len(missing),
        "datasets": summary_datasets,
    }
    write_json(output_root / "summary.json", summary)
    if rows:
        fieldnames = ["dataset", "n", "blind"]
        for carrier in CARRIERS:
            fieldnames.extend(
                [
                    carrier,
                    f"{carrier}_no_write",
                    f"{carrier}_write_delta",
                    f"{carrier}_position_null",
                    f"{carrier}_recovery",
                ]
            )
        fieldnames.append("full")
        with (output_root / "accuracy.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return summary


def _independent_expected_masks(
    seq_len: int, prefill_len: int, readout_indices: list[int]
) -> np.ndarray:
    causal = np.tri(seq_len, seq_len, dtype=bool)
    readout = sorted(set(int(item) for item in readout_indices))
    aware = causal.copy()
    aware[prefill_len:, :prefill_len] = False
    aware[prefill_len:, readout] = True
    no_write = aware.copy()
    for query in readout:
        no_write[query, :prefill_len] = False
        no_write[query, [key for key in readout if key <= query]] = True
    position_null = causal.copy()
    position_null[prefill_len:, :prefill_len] = False
    return np.stack([aware, no_write, position_null], axis=0)


def _validate_raw_score(
    saved: dict[str, Any],
    logits: np.ndarray,
    plan: dict[str, Any],
    *,
    context: str,
) -> None:
    logits64 = np.asarray(logits, dtype=np.float64)
    if logits64.ndim != 1 or not np.isfinite(logits64).all():
        raise RuntimeError(f"Invalid raw logits for {context}: {logits64.shape}")
    normalizer = float(logits64.max() + np.log(np.exp(logits64 - logits64.max()).sum()))
    values = {
        label: float(logits64[int(token_id)] - normalizer)
        for label, token_id in plan["candidate_token_ids"].items()
    }
    for label, value in values.items():
        observed = float(saved["candidate_logprobs"][label])
        if abs(value - observed) > 5e-5:
            raise RuntimeError(
                f"Raw-logit score reconstruction failed for {context} {label}: "
                f"{value} != {observed}"
            )
    if max(values, key=values.get) != saved["predicted_key"]:
        raise RuntimeError(f"Raw-logit argmax reconstruction failed for {context}")


def _numpy_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _validate_raw_parity(
    reference: np.ndarray,
    candidate: np.ndarray,
    plan: dict[str, Any],
    *,
    context: str,
    atol: float = LOGIT_PARITY_ATOL,
    rtol: float = LOGIT_PARITY_RTOL,
) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 1 or reference.shape != candidate.shape:
        raise RuntimeError(
            f"Raw parity shape mismatch for {context}: "
            f"{reference.shape} vs {candidate.shape}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise RuntimeError(f"Non-finite raw parity logits for {context}")
    diff = np.abs(reference - candidate)
    threshold = float(atol) + float(rtol) * np.abs(candidate)
    if np.any(diff > threshold):
        raise RuntimeError(
            f"Full-vocab parity failed for {context}: "
            f"max={float(diff.max())} mean={float(diff.mean())}"
        )
    labels = list(plan["candidate_token_ids"])
    candidate_ids = np.asarray(
        [int(plan["candidate_token_ids"][label]) for label in labels], dtype=np.int64
    )
    reference_label = labels[int(np.argmax(reference[candidate_ids]))]
    candidate_label = labels[int(np.argmax(candidate[candidate_ids]))]
    if reference_label != candidate_label:
        raise RuntimeError(
            f"Candidate argmax parity failed for {context}: "
            f"{reference_label} != {candidate_label}"
        )
    candidate_max = float(
        np.abs(reference[candidate_ids] - candidate[candidate_ids]).max()
    )
    if candidate_max > CANDIDATE_PARITY_ATOL:
        raise RuntimeError(
            f"Candidate-logit parity failed for {context}: {candidate_max}"
        )
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def _validate_corruption_control(
    original: np.ndarray,
    corrupted: np.ndarray,
    *,
    context: str,
) -> dict[str, float]:
    original = np.asarray(original, dtype=np.float64)
    corrupted = np.asarray(corrupted, dtype=np.float64)
    if (
        original.shape != corrupted.shape
        or original.ndim != 2
        or original.shape[0] != 3
    ):
        raise RuntimeError(
            f"Corruption-control shape mismatch for {context}: "
            f"{original.shape} vs {corrupted.shape}"
        )
    maxima = {
        condition: float(np.abs(original[idx] - corrupted[idx]).max())
        for idx, condition in enumerate(MASK_CONDITIONS)
    }
    if maxima["aware"] <= CORRUPTION_POSITIVE_MIN:
        raise RuntimeError(
            f"Aware corruption positive control was ineffective for "
            f"{context}: {maxima['aware']}"
        )
    for condition in ("no_write", "position_null"):
        if maxima[condition] > BLOCKED_PREFIX_ATOL:
            raise RuntimeError(
                f"Raw blocked-prefix invariance failed for "
                f"{context} {condition}: {maxima[condition]}"
            )
    return maxima


def _contiguous_value_spans(ids: np.ndarray, value: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, token_id in enumerate(np.asarray(ids).reshape(-1).tolist()):
        if int(token_id) == int(value) and start is None:
            start = idx
        elif int(token_id) != int(value) and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, int(np.asarray(ids).size)))
    return spans


def _independent_qwen_mrope(
    ids: np.ndarray,
    image_grid_thw: list[list[int]],
    *,
    image_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = [int(item) for item in np.asarray(ids).reshape(-1).tolist()]
    image_starts = [
        idx
        for idx, token in enumerate(tokens[:-1])
        if token == int(vision_start_token_id)
        and tokens[idx + 1] == int(image_token_id)
    ]
    if len(image_starts) != len(image_grid_thw):
        raise RuntimeError(
            f"Independent mRoPE image count mismatch: "
            f"{len(image_starts)} != {len(image_grid_thw)}"
        )
    chunks: list[np.ndarray] = []
    cursor = 0
    for grid in image_grid_thw:
        try:
            image_start = tokens.index(int(image_token_id), cursor)
        except ValueError as exc:
            raise RuntimeError("Independent mRoPE could not locate image core") from exc
        text_len = image_start - cursor
        start_position = int(chunks[-1].max()) + 1 if chunks else 0
        if text_len:
            text = np.arange(text_len, dtype=np.int64)[None, :]
            chunks.append(np.repeat(text, 3, axis=0) + start_position)
        t, h, w = (int(value) for value in grid)
        if h % spatial_merge_size or w % spatial_merge_size:
            raise RuntimeError(f"Qwen grid is not divisible by merge size: {grid}")
        merged_h = h // spatial_merge_size
        merged_w = w // spatial_merge_size
        temporal = np.zeros((t, merged_h, merged_w), dtype=np.int64).reshape(-1)
        height = np.broadcast_to(
            np.arange(merged_h, dtype=np.int64)[None, :, None],
            (t, merged_h, merged_w),
        ).reshape(-1)
        width = np.broadcast_to(
            np.arange(merged_w, dtype=np.int64)[None, None, :],
            (t, merged_h, merged_w),
        ).reshape(-1)
        visual = np.stack([temporal, height, width], axis=0)
        visual += start_position + text_len
        chunks.append(visual)
        cursor = image_start + int(visual.shape[-1])
    if cursor < len(tokens):
        start_position = int(chunks[-1].max()) + 1 if chunks else 0
        text_len = len(tokens) - cursor
        text = np.arange(text_len, dtype=np.int64)[None, :]
        chunks.append(np.repeat(text, 3, axis=0) + start_position)
    positions = np.concatenate(chunks, axis=1)
    if positions.shape != (3, len(tokens)):
        raise RuntimeError(
            f"Independent mRoPE length mismatch: {positions.shape} vs {len(tokens)}"
        )
    rope_delta = np.asarray([[int(positions.max()) + 1 - len(tokens)]], dtype=np.int64)
    return positions[:, None, :], rope_delta


def validate_smoke(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_json(manifest_path)
    expected_provenance = _manifest_provenance(manifest, manifest_path)
    raw_root = Path(args.raw_root).resolve()
    artifacts = sorted(raw_root.glob("**/artifact.json"))
    if not artifacts:
        raise RuntimeError(f"No raw smoke artifacts found under {raw_root}")
    if args.expected_artifacts is not None and len(artifacts) != int(
        args.expected_artifacts
    ):
        raise RuntimeError(
            f"Smoke artifact count mismatch: {len(artifacts)} != {args.expected_artifacts}"
        )
    validations = []
    diagnostic_artifacts = 0
    runtime_identities: list[dict[str, Any]] = []
    for path in artifacts:
        artifact = _load_json(path)
        if artifact.get("schema") != RECORD_SCHEMA:
            raise RuntimeError(f"Unexpected smoke record schema: {path}")
        if artifact.get("provenance") != expected_provenance:
            raise RuntimeError(f"Smoke artifact provenance mismatch: {path}")
        runtime_identity = artifact.get("runtime_identity")
        required_runtime_fields = {
            "python",
            "torch",
            "transformers",
            "cuda_runtime",
            "cuda_device_name",
            "attention_backend",
            "model_class",
            "processor_class",
            "tokenizer_class",
            "hidden_size",
            "num_hidden_layers",
            "hf_device_map",
        }
        if not isinstance(
            runtime_identity, dict
        ) or not required_runtime_fields.issubset(runtime_identity):
            raise RuntimeError(f"Incomplete runtime identity in {path}")
        expected_transformers = (
            "4.53.3" if artifact.get("model_family") == "qwen25vl" else "4.51.0"
        )
        if runtime_identity.get("transformers") != expected_transformers:
            raise RuntimeError(
                f"Unexpected transformers runtime in {path}: "
                f"{runtime_identity.get('transformers')} != {expected_transformers}"
            )
        runtime_identities.append(runtime_identity)
        cuda_memory = artifact.get("cuda_memory", {})
        if not all(
            int(cuda_memory.get(key, 0)) > 0
            for key in (
                "allocated_bytes",
                "reserved_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            )
        ):
            raise RuntimeError(f"Invalid CUDA memory counters in {path}")
        if manifest["model_key"] == "qwen25vl_32b":
            device_targets = set(runtime_identity["hf_device_map"].values())
            if not device_targets or any(
                target in {"cpu", "disk"} for target in device_targets
            ):
                raise RuntimeError(f"Qwen32 checkpoint was offloaded in {path}")
            if (
                int(runtime_identity["hidden_size"]) != 5120
                or int(runtime_identity["num_hidden_layers"]) != 64
            ):
                raise RuntimeError(f"Unexpected Qwen32 architecture in {path}")
        if artifact.get("model_family") == "minicpmv45":
            if runtime_identity["model_class"] != "MiniCPMV":
                raise RuntimeError(f"Unexpected MiniCPM-V model class in {path}")
            if runtime_identity["processor_class"] != "MiniCPMVProcessor":
                raise RuntimeError(f"Unexpected MiniCPM-V processor class in {path}")
            for sequence_name in ("full", *CARRIERS):
                sequence = artifact["sequences"][sequence_name]
                inputs = sequence["inputs"]
                if "temporal_ids" not in inputs:
                    raise RuntimeError(
                        f"MiniCPM-V temporal IDs missing: {path} {sequence_name}"
                    )
                if "audio_features" in inputs or "audio_bounds" in inputs:
                    raise RuntimeError(
                        f"MiniCPM-V received audio inputs: {path} {sequence_name}"
                    )
                temporal_ids = _flatten_ints(sequence["metadata"]["temporal_ids"])
                expected_images = 2 if sequence_name in VISUAL_CARRIERS else 1
                if temporal_ids != [-1] * expected_images:
                    raise RuntimeError(
                        f"MiniCPM-V static-image temporal IDs mismatch: "
                        f"{path} {sequence_name} {temporal_ids}"
                    )
        entry_environment = artifact.get("entry_environment", {})
        if entry_environment.get("REPLAY_MODE") != "image_text_image":
            raise RuntimeError(f"Smoke replay mode is not IQI: {path}")
        if entry_environment.get("REPLAY_PROMPT_TEMPLATE_NAME") != "directly_answer":
            raise RuntimeError(
                f"Smoke prompt policy is not the standard direct template: {path} "
                f"{entry_environment}"
            )
        plan = artifact["candidate_plan"]
        with np.load(
            path.parent / artifact["raw_files"]["logits"], allow_pickle=False
        ) as raw_logits:
            logits = {name: raw_logits[name] for name in raw_logits.files}
        with np.load(
            path.parent / artifact["raw_files"]["runtime_tensors"],
            allow_pickle=False,
        ) as raw_runtime:
            runtime_tensors = {name: raw_runtime[name] for name in raw_runtime.files}
        for name in ("blind", "full", *CARRIERS):
            if name not in logits:
                raise RuntimeError(f"Raw smoke logits missing {name}: {path}")
        _validate_raw_score(
            artifact["blind"], logits["blind"], plan, context=f"{path}:blind"
        )
        _validate_raw_score(
            artifact["full"], logits["full"], plan, context=f"{path}:full"
        )

        blind_ids = np.load(path.parent / artifact["raw_files"]["input_ids"]["blind"])
        forced_prefix = np.asarray([plan["forced_prefix_ids"]], dtype=blind_ids.dtype)
        if not np.array_equal(blind_ids, forced_prefix):
            raise RuntimeError(f"Blind path contains chat/context tokens: {path}")
        blind_roles = [
            item["role"] for item in artifact["sequences"]["blind"]["token_table"]
        ]
        if blind_roles != ["answer_prefix"] * blind_ids.shape[-1]:
            raise RuntimeError(
                f"Blind token roles are not literal Answer prefix: {path}"
            )

        counts = {
            carrier: int(artifact["carriers"][carrier]["token_count"])
            for carrier in CARRIERS
        }
        if len(set(counts.values())) != 1 or next(iter(counts.values())) <= 0:
            raise RuntimeError(f"Smoke carrier counts differ in {path}: {counts}")
        readout_spans = {
            carrier: tuple(
                int(item) for item in artifact["carriers"][carrier]["readout_indices"]
            )
            for carrier in CARRIERS
        }
        if len(set(readout_spans.values())) != 1:
            raise RuntimeError(
                f"Smoke carrier readout positions differ in {path}: {readout_spans}"
            )
        shared_readout = next(iter(readout_spans.values()))
        if shared_readout != tuple(range(shared_readout[0], shared_readout[-1] + 1)):
            raise RuntimeError(
                f"Smoke carrier readout is non-contiguous in {path}: {shared_readout}"
            )
        carrier_lengths = {
            carrier: int(artifact["carriers"][carrier]["sequence_len"])
            for carrier in CARRIERS
        }
        carrier_prefill_lengths = {
            carrier: int(artifact["carriers"][carrier]["prefill_len"])
            for carrier in CARRIERS
        }
        carrier_readout_to_boundary = {
            carrier: carrier_prefill_lengths[carrier]
            - int(artifact["carriers"][carrier]["readout_indices"][-1])
            - 1
            for carrier in CARRIERS
        }
        for invariant_name, invariant in (
            ("sequence length", carrier_lengths),
            ("prefill length", carrier_prefill_lengths),
            ("readout-to-boundary distance", carrier_readout_to_boundary),
        ):
            if len(set(invariant.values())) != 1:
                raise RuntimeError(
                    f"Smoke carrier {invariant_name} mismatch in {path}: {invariant}"
                )
        full_ids = np.load(path.parent / artifact["raw_files"]["input_ids"]["full"])
        prompt_hashes = {
            name: artifact["sequences"][name]["metadata"].get("standard_prompt_sha256")
            for name in ("full", *CARRIERS)
        }
        if None in prompt_hashes.values() or len(set(prompt_hashes.values())) != 1:
            raise RuntimeError(
                f"Carrier paths did not share one standard-entry prompt: {path} "
                f"{prompt_hashes}"
            )
        artifact_has_diagnostics = True
        text_readout_ids: dict[str, np.ndarray] = {}
        readout_embedding_stats: dict[str, dict[str, float]] = {}
        for carrier in CARRIERS:
            carrier_record = artifact["carriers"][carrier]
            sequence = artifact["sequences"][carrier]
            seq_len = int(carrier_record["sequence_len"])
            prefill_len = int(carrier_record["prefill_len"])
            readout = [int(item) for item in carrier_record["readout_indices"]]
            input_ids = np.load(
                path.parent / artifact["raw_files"]["input_ids"][carrier]
            )
            if input_ids.shape != (1, seq_len):
                raise RuntimeError(
                    f"Raw input shape mismatch for {path} {carrier}: {input_ids.shape}"
                )
            edit_at = int(carrier_record["metadata"]["edit_position"])
            if not np.array_equal(input_ids[0, :edit_at], full_ids[0, :edit_at]):
                raise RuntimeError(
                    f"Pre-carrier prefix diverges from standard IQ input: {path} {carrier}"
                )
            raw_mask = np.load(
                path.parent / artifact["raw_files"]["masks"][carrier]
            ).astype(bool)
            expected_mask = _independent_expected_masks(seq_len, prefill_len, readout)
            if not np.array_equal(raw_mask, expected_mask):
                mismatch = int(np.count_nonzero(raw_mask != expected_mask))
                raise RuntimeError(
                    f"Raw mask oracle mismatch for {path} {carrier}: {mismatch} cells"
                )
            token_table = sequence["token_table"]
            table_readout = [
                int(item["position"]) for item in token_table if item["is_readout"]
            ]
            if table_readout != readout:
                raise RuntimeError(f"Token-table readout mismatch for {path} {carrier}")
            if carrier in TEXT_CARRIERS:
                observed_ids = input_ids[0, readout].astype(np.int64)
                text_readout_ids[carrier] = observed_ids
                if (
                    _token_ids_sha256(observed_ids.tolist())
                    != carrier_record["metadata"]["carrier_token_ids_sha256"]
                ):
                    raise RuntimeError(
                        f"Text carrier token hash mismatch for {path} {carrier}"
                    )
            if carrier in {"dot_text", "space_text"}:
                literal_id = int(carrier_record["metadata"]["literal_token_id"])
                if np.any(text_readout_ids[carrier] != literal_id):
                    raise RuntimeError(
                        f"Literal carrier token mismatch for {path} {carrier}"
                    )

            carrier_logits = logits[carrier]
            if carrier_logits.ndim != 2 or carrier_logits.shape[0] != len(
                MASK_CONDITIONS
            ):
                raise RuntimeError(
                    f"Carrier raw-logit shape mismatch for {path} {carrier}: "
                    f"{carrier_logits.shape}"
                )
            for condition_idx, condition in enumerate(MASK_CONDITIONS):
                _validate_raw_score(
                    carrier_record["scores"][condition],
                    carrier_logits[condition_idx],
                    plan,
                    context=f"{path}:{carrier}:{condition}",
                )

            model_meta = artifact["runtime_meta"][carrier]
            if float(model_meta["visual_scatter_max_abs_diff"]) > 1e-5:
                raise RuntimeError(f"Visual scatter mismatch for {path} {carrier}")
            positions_key = f"{carrier}__position_ids"
            if positions_key not in runtime_tensors:
                raise RuntimeError(f"Raw position IDs missing for {path} {carrier}")
            positions = np.asarray(runtime_tensors[positions_key])
            if positions.shape[-1] != seq_len:
                raise RuntimeError(f"Position-id length mismatch for {path} {carrier}")
            if artifact["model_family"] == "qwen25vl":
                rope_key = f"{carrier}__rope_deltas"
                if (
                    rope_key not in runtime_tensors
                    or np.asarray(runtime_tensors[rope_key]).size != 1
                ):
                    raise RuntimeError(
                        f"Qwen raw rope delta missing or malformed: {path} {carrier}"
                    )
                expected_positions, expected_rope_delta = _independent_qwen_mrope(
                    input_ids[0],
                    sequence["metadata"]["image_grid_thw"],
                    image_token_id=int(sequence["metadata"]["image_token_id"]),
                    vision_start_token_id=int(
                        sequence["metadata"]["vision_start_token_id"]
                    ),
                    spatial_merge_size=int(sequence["metadata"]["spatial_merge_size"]),
                )
                if not np.array_equal(positions, expected_positions):
                    mismatch = int(np.count_nonzero(positions != expected_positions))
                    raise RuntimeError(
                        f"Qwen mRoPE exact oracle mismatch for {path} {carrier}: "
                        f"{mismatch} cells"
                    )
                if not np.array_equal(
                    np.asarray(runtime_tensors[rope_key]), expected_rope_delta
                ):
                    raise RuntimeError(
                        f"Qwen rope delta exact oracle mismatch: {path} {carrier}"
                    )
                if positions.shape != (3, 1, seq_len):
                    raise RuntimeError(
                        f"Qwen mRoPE position shape mismatch for {path} {carrier}: "
                        f"{positions.shape}"
                    )
                image_token_id = int(sequence["metadata"]["image_token_id"])
                image_spans = _contiguous_value_spans(input_ids[0], image_token_id)
                expected_images = 2 if carrier in VISUAL_CARRIERS else 1
                if len(image_spans) != expected_images:
                    raise RuntimeError(
                        f"Qwen raw image-core count mismatch for {path} {carrier}: "
                        f"{image_spans}"
                    )
                if carrier in VISUAL_CARRIERS:
                    if tuple(readout) != tuple(range(*image_spans[1])):
                        raise RuntimeError(
                            f"Qwen readout is not exactly the second raw visual core: "
                            f"{path} {carrier}"
                        )
                text_mask = input_ids[0] != image_token_id
                if not (
                    np.array_equal(
                        positions[0, 0, text_mask], positions[1, 0, text_mask]
                    )
                    and np.array_equal(
                        positions[0, 0, text_mask], positions[2, 0, text_mask]
                    )
                ):
                    raise RuntimeError(
                        f"Qwen text mRoPE axes diverge in {path} {carrier}"
                    )
                for span_start, span_end in image_spans:
                    spatial = positions[1:, 0, span_start:span_end]
                    if spatial.shape[-1] > 1 and all(
                        np.unique(axis).size == 1 for axis in spatial
                    ):
                        raise RuntimeError(
                            f"Qwen visual mRoPE has no spatial variation: "
                            f"{path} {carrier} {(span_start, span_end)}"
                        )
            else:
                if (
                    sequence["metadata"].get("vision_encode_mode")
                    != "per_image_batch_one"
                    or model_meta.get("vision_encode_mode") != "per_image_batch_one"
                ):
                    raise RuntimeError(
                        f"MiniCPM images were not encoded independently: {path} {carrier}"
                    )
                if positions.shape != (1, seq_len) or not np.array_equal(
                    positions[0], np.arange(seq_len, dtype=positions.dtype)
                ):
                    raise RuntimeError(
                        f"MiniCPM raw position IDs are not exact arange: {path} {carrier}"
                    )
                bounds_key = f"{carrier}__image_bound"
                cache_key = f"{carrier}__cache_position"
                if (
                    bounds_key not in runtime_tensors
                    or cache_key not in runtime_tensors
                ):
                    raise RuntimeError(
                        f"MiniCPM raw bounds/cache positions missing: {path} {carrier}"
                    )
                bounds = [
                    tuple(int(value) for value in bound)
                    for bound in np.asarray(runtime_tensors[bounds_key]).tolist()
                ]
                if bounds != [
                    tuple(int(value) for value in bound)
                    for bound in sequence["metadata"]["image_bound"]
                ]:
                    raise RuntimeError(
                        f"MiniCPM raw/metadata image bounds differ: {path} {carrier}"
                    )
                if not np.array_equal(
                    np.asarray(runtime_tensors[cache_key]),
                    np.arange(
                        seq_len, dtype=np.asarray(runtime_tensors[cache_key]).dtype
                    ),
                ):
                    raise RuntimeError(
                        f"MiniCPM raw cache position mismatch: {path} {carrier}"
                    )
                expected_images = 2 if carrier in VISUAL_CARRIERS else 1
                if len(bounds) != expected_images or any(
                    end - start != 64 for start, end in bounds
                ):
                    raise RuntimeError(
                        f"MiniCPM raw image bounds mismatch for {path} {carrier}: {bounds}"
                    )
                if carrier in VISUAL_CARRIERS and tuple(readout) != tuple(
                    range(*bounds[1])
                ):
                    raise RuntimeError(
                        f"MiniCPM readout is not exactly image_bound[1]: {path} {carrier}"
                    )
                if (
                    float(model_meta["no_audio_omni_max_abs_diff"])
                    > BLOCKED_PREFIX_ATOL
                ):
                    raise RuntimeError(
                        f"MiniCPM no-audio omni path changed embeddings: {path} {carrier}"
                    )
                expected_postprocess = (
                    "applied_no_audio_identity"
                    if artifact["model_family"] == "minicpmo45"
                    else "not_applicable"
                )
                if model_meta.get("embedding_postprocess") != expected_postprocess:
                    raise RuntimeError(
                        f"MiniCPM embedding postprocess mismatch: {path} {carrier}"
                    )
                image_spans = bounds

            for feature_idx, feature in enumerate(model_meta["visual_features"]):
                if float(feature["l2_norm"]) <= 0 or float(feature["max_abs"]) <= 0:
                    raise RuntimeError(
                        f"Degenerate visual feature for {path} {carrier}"
                    )
                feature_key = f"{carrier}__visual_feature_{feature_idx}"
                core_key = f"{carrier}__visual_core_{feature_idx}"
                if (
                    feature_key not in runtime_tensors
                    or core_key not in runtime_tensors
                ):
                    raise RuntimeError(
                        f"Raw visual tensors missing for {path} {carrier} image {feature_idx}"
                    )
                raw_feature = np.asarray(runtime_tensors[feature_key], dtype=np.float32)
                raw_core = np.asarray(runtime_tensors[core_key], dtype=np.float32)
                if _numpy_sha256(raw_feature) != feature["float32_sha256"]:
                    raise RuntimeError(
                        f"Raw visual feature hash mismatch for {path} {carrier} image {feature_idx}"
                    )
                if raw_feature.shape != raw_core.shape or not np.array_equal(
                    raw_feature, raw_core
                ):
                    raise RuntimeError(
                        f"Visual feature/core scatter mismatch for {path} {carrier} image {feature_idx}"
                    )
                if int(raw_core.shape[0]) != int(
                    image_spans[feature_idx][1] - image_spans[feature_idx][0]
                ):
                    raise RuntimeError(
                        f"Visual tensor/core-span length mismatch for {path} {carrier} image {feature_idx}"
                    )

            diagnostics = artifact["carriers"][carrier].get("diagnostics", {})
            if diagnostics:
                diagnostic_file = artifact["raw_files"].get("diagnostic_logits")
                if not diagnostic_file:
                    raise RuntimeError(
                        f"Raw diagnostic logits missing: {path} {carrier}"
                    )
                with np.load(
                    path.parent / diagnostic_file, allow_pickle=False
                ) as raw_diag:
                    original = raw_diag[f"{carrier}__original_batch"]
                    corrupted = raw_diag[f"{carrier}__corrupted_batch"]
                    corruption_meta = diagnostics["prefix_embedding_corruption"]
                    if (
                        corruption_meta["before_sha256"]
                        == corruption_meta["after_sha256"]
                    ):
                        raise RuntimeError(
                            f"Prefix corruption did not change raw embeddings: {path} {carrier}"
                        )
                    _validate_corruption_control(
                        original,
                        corrupted,
                        context=f"{path}:{carrier}",
                    )
                    if _is_minicpm_family(artifact["model_family"]):
                        pre_key = f"{carrier}__pre_omni_embeddings"
                        post_key = f"{carrier}__post_omni_embeddings"
                        if (
                            pre_key not in runtime_tensors
                            or post_key not in runtime_tensors
                        ):
                            raise RuntimeError(
                                f"MiniCPM diagnostic omni tensors missing: {path} {carrier}"
                            )
                        if not np.array_equal(
                            runtime_tensors[pre_key], runtime_tensors[post_key]
                        ):
                            omni_diff = float(
                                np.max(
                                    np.abs(
                                        runtime_tensors[pre_key]
                                        - runtime_tensors[post_key]
                                    )
                                )
                            )
                            raise RuntimeError(
                                f"MiniCPM raw pre/post-omni embeddings differ: "
                                f"{path} {carrier} {omni_diff}"
                            )
                    _validate_raw_parity(
                        raw_diag[f"{carrier}__standard_causal"],
                        raw_diag[f"{carrier}__manual_causal"],
                        plan,
                        context=f"{path}:{carrier}:standard-vs-manual",
                    )
                    for condition_idx, condition in enumerate(MASK_CONDITIONS):
                        _validate_raw_parity(
                            original[condition_idx],
                            raw_diag[f"{carrier}__single_{condition}"],
                            plan,
                            context=f"{path}:{carrier}:batch-vs-single:{condition}",
                        )
            else:
                artifact_has_diagnostics = False

        ordered_lorem = text_readout_ids["ordered_lorem"]
        shuffled_orders = set()
        for carrier, expected_seed in SHUFFLED_LOREM_SEEDS.items():
            shuffled = text_readout_ids[carrier]
            metadata = artifact["carriers"][carrier]["metadata"]
            if int(metadata["shuffle_seed"]) != expected_seed:
                raise RuntimeError(f"Lorem shuffle seed mismatch: {path} {carrier}")
            if not np.array_equal(np.sort(ordered_lorem), np.sort(shuffled)):
                raise RuntimeError(f"Lorem token multiset changed for {path} {carrier}")
            if np.array_equal(ordered_lorem, shuffled):
                raise RuntimeError(
                    f"Lorem shuffle kept the original order: {path} {carrier}"
                )
            shuffled_orders.add(shuffled.tobytes())
        if len(shuffled_orders) != len(SHUFFLED_LOREM_SEEDS):
            raise RuntimeError(f"Lorem shuffle seeds produced duplicate orders: {path}")

        if artifact_has_diagnostics:
            outside_readout = np.ones(carrier_lengths[CARRIERS[0]], dtype=bool)
            outside_readout[list(shared_readout)] = False
            for topology, names in (
                ("visual", VISUAL_CARRIERS),
                ("text", TEXT_CARRIERS),
            ):
                reference_key = f"{names[0]}__inputs_embeds"
                if reference_key not in runtime_tensors:
                    raise RuntimeError(
                        f"Raw diagnostic embeddings missing: {path} {reference_key}"
                    )
                reference = np.asarray(runtime_tensors[reference_key])
                for name in names[1:]:
                    key = f"{name}__inputs_embeds"
                    if key not in runtime_tensors:
                        raise RuntimeError(
                            f"Raw diagnostic embeddings missing: {path} {key}"
                        )
                    observed = np.asarray(runtime_tensors[key])
                    if reference.shape != observed.shape or not np.array_equal(
                        reference[:, outside_readout], observed[:, outside_readout]
                    ):
                        raise RuntimeError(
                            f"Non-readout embeddings changed within {topology} topology: "
                            f"{path} {names[0]} vs {name}"
                        )
            for name in TEXT_CARRIERS:
                actual = np.asarray(runtime_tensors[f"{name}__inputs_embeds"])[
                    :, list(shared_readout)
                ]
                expected_key = f"{name}__expected_text_core"
                if expected_key not in runtime_tensors or not np.array_equal(
                    actual, np.asarray(runtime_tensors[expected_key])
                ):
                    raise RuntimeError(
                        f"Text readout embeddings do not match token lookup: {path} {name}"
                    )
            for name in CARRIERS:
                core = np.asarray(
                    runtime_tensors[f"{name}__inputs_embeds"], dtype=np.float64
                )[0, list(shared_readout)]
                centered = core - core.mean(axis=0, keepdims=True)
                readout_embedding_stats[name] = {
                    "mean_token_l2_norm": float(np.linalg.norm(core, axis=1).mean()),
                    "mean_tokenwise_variance": float(np.mean(centered * centered)),
                }

        if artifact["model_family"] == "qwen25vl":
            source_features = {}
            for name in ("full", *CARRIERS):
                if (
                    artifact["runtime_meta"][name].get("vision_encode_mode")
                    != "per_image_batch_one"
                ):
                    raise RuntimeError(
                        f"Qwen images were not encoded independently: {path} {name}"
                    )
                feature_key = f"{name}__visual_feature_0"
                core_key = f"{name}__visual_core_0"
                if (
                    feature_key not in runtime_tensors
                    or core_key not in runtime_tensors
                ):
                    raise RuntimeError(
                        f"Source visual tensors missing for {path} {name}"
                    )
                feature = np.asarray(runtime_tensors[feature_key])
                core = np.asarray(runtime_tensors[core_key])
                if not np.array_equal(feature, core):
                    raise RuntimeError(
                        f"Source visual feature/core mismatch for {path} {name}"
                    )
                source_features[name] = feature
            reference_source = source_features["full"]
            for name in CARRIERS:
                if not np.array_equal(reference_source, source_features[name]):
                    max_diff = float(
                        np.max(np.abs(reference_source - source_features[name]))
                    )
                    raise RuntimeError(
                        f"Qwen source visual feature changed across carriers: "
                        f"{path} full vs {name} max={max_diff}"
                    )

        if _is_minicpm_family(artifact["model_family"]):
            if (
                artifact["sequences"]["full"]["metadata"].get("vision_encode_mode")
                != "per_image_batch_one"
            ):
                raise RuntimeError(f"MiniCPM full image was not encoded alone: {path}")
            source_features = {}
            for name in ("full", *CARRIERS):
                feature_key = f"{name}__visual_feature_0"
                core_key = f"{name}__visual_core_0"
                if (
                    feature_key not in runtime_tensors
                    or core_key not in runtime_tensors
                ):
                    raise RuntimeError(
                        f"Source visual tensors missing for {path} {name}"
                    )
                feature = np.asarray(runtime_tensors[feature_key])
                core = np.asarray(runtime_tensors[core_key])
                if not np.array_equal(feature, core):
                    raise RuntimeError(
                        f"Source visual feature/core mismatch for {path} {name}"
                    )
                source_features[name] = feature
            reference_source = source_features["full"]
            for name in CARRIERS:
                if not np.array_equal(reference_source, source_features[name]):
                    max_diff = float(
                        np.max(np.abs(reference_source - source_features[name]))
                    )
                    raise RuntimeError(
                        f"Source visual feature changed across carriers: "
                        f"{path} full vs {name} max={max_diff}"
                    )

        if artifact_has_diagnostics:
            diagnostic_artifacts += 1

        for carrier, rgb in COLOR_VALUES.items():
            image_path = path.parent / f"{carrier}.png"
            image = np.asarray(Image.open(image_path).convert("RGB"))
            unique = np.unique(image.reshape(-1, 3), axis=0)
            if unique.shape != (1, 3) or unique[0].tolist() != list(rgb):
                raise RuntimeError(
                    f"Raw color image mismatch: {image_path} {unique.tolist()}"
                )
        noise_hashes = set()
        for carrier, seed in NOISE_CARRIER_SEEDS.items():
            image_path = path.parent / f"{carrier}.png"
            image = np.asarray(Image.open(image_path).convert("RGB"))
            if not (
                np.array_equal(image[:, :, 0], image[:, :, 1])
                and np.array_equal(image[:, :, 0], image[:, :, 2])
                and float(image[:, :, 0].std()) > 20.0
            ):
                raise RuntimeError(
                    f"Noise image is not nonconstant grayscale: {image_path}"
                )
            metadata = artifact["sequences"][carrier]["metadata"]
            observed_hash = hashlib.sha256(image.tobytes()).hexdigest()
            if (
                int(metadata["noise_seed"]) != seed
                or metadata["raw_carrier_sha256"] != observed_hash
            ):
                raise RuntimeError(f"Noise image metadata mismatch: {image_path}")
            noise_hashes.add(observed_hash)
        if len(noise_hashes) != len(NOISE_CARRIER_SEEDS):
            raise RuntimeError(f"Noise image seeds produced duplicate images: {path}")
        source_image_path = path.parent / "source_readout_image.png"
        if not source_image_path.is_file():
            raise RuntimeError(
                f"Raw source readout image is missing: {source_image_path}"
            )

        layout_key = (
            "image_grid_thw"
            if artifact["model_family"] == "qwen25vl"
            else "image_bound"
        )
        layouts = {
            carrier: artifact["sequences"][carrier]["metadata"][layout_key]
            for carrier in VISUAL_CARRIERS
        }
        if len({json.dumps(value, sort_keys=True) for value in layouts.values()}) != 1:
            raise RuntimeError(f"Visual carrier layouts differ: {path} {layouts}")
        feature_hashes = set()
        for carrier in VISUAL_CARRIERS:
            features = artifact["runtime_meta"][carrier]["visual_features"]
            if len(features) != 2:
                raise RuntimeError(
                    f"Visual carrier feature count is not two: {path} {carrier}"
                )
            feature_hashes.add(features[1]["sha256"])
        if len(feature_hashes) != len(VISUAL_CARRIERS):
            raise RuntimeError(f"Visual readout embeddings are not distinct: {path}")

        validations.append(
            {
                "artifact": str(path),
                "dataset": artifact["dataset"],
                "model_family": artifact["model_family"],
                "matched_token_count": next(iter(counts.values())),
                "diagnostics": artifact_has_diagnostics,
                "readout_embedding_stats": readout_embedding_stats,
            }
        )
    if diagnostic_artifacts < int(args.require_diagnostics):
        raise RuntimeError(
            f"Only {diagnostic_artifacts} diagnostic artifacts; "
            f"required {args.require_diagnostics}"
        )
    runtime_hashes = {sha256_json(identity) for identity in runtime_identities}
    if len(runtime_hashes) != 1:
        raise RuntimeError(
            f"Smoke artifacts used inconsistent runtime identities: {runtime_hashes}"
        )
    payload = {
        "schema": "topic-image-replay/readout-random-carrier-smoke-validation/v1",
        "manifest_sha256": sha256_file(manifest_path),
        "provenance": expected_provenance,
        "model_key": manifest["model_key"],
        "model_family": manifest["model_family"],
        "model_identity": manifest["model_identity"],
        "runtime_identity": runtime_identities[0],
        "artifact_count": len(validations),
        "diagnostic_artifact_count": diagnostic_artifacts,
        "validations": validations,
        "passed": True,
    }
    write_json(Path(args.output), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Matched visual and text readout carriers"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--repo-root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--all-single-choice-manifest", required=True)
    manifest.add_argument("--fixed-choice-manifest", required=True)
    manifest.add_argument("--mmstar-ai2d-manifest", required=True)
    manifest.add_argument("--lmu-data", required=True)
    manifest.add_argument("--matrix-config", required=True)
    manifest.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    manifest.add_argument("--model-key", required=True, choices=sorted(MODEL_FAMILIES))
    manifest.add_argument("--model-path", required=True)
    manifest.add_argument("--num-shards", type=int, default=8)
    manifest.add_argument("--samples-per-dataset", type=int)
    manifest.add_argument("--selection-seed", type=int, default=20260804)

    verify = sub.add_parser("verify-run-contract")
    verify.add_argument("--repo-root", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--model-path", required=True)
    verify.add_argument("--lmu-data", required=True)
    verify.add_argument("--matrix-config", required=True)
    verify.add_argument("--output", required=True)

    run = sub.add_parser("run")
    run.add_argument("--repo-root", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--output-jsonl", required=True)
    run.add_argument("--runtime-root", required=True)
    run.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    run.add_argument("--model-key", required=True, choices=sorted(MODEL_FAMILIES))
    run.add_argument("--model-path", required=True)
    run.add_argument("--lmu-data", required=True)
    run.add_argument("--matrix-config", required=True)
    run.add_argument("--run-contract-attestation", required=True)
    run.add_argument("--expected-runtime-validation")
    run.add_argument("--gpu-id", default="0")
    run.add_argument("--shard-rank", type=int)
    run.add_argument("--limit", type=int)
    run.add_argument("--one-per-dataset", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--repair-torn-jsonl", action="store_true")
    run.add_argument("--dump-raw-root")
    run.add_argument("--diagnostics", action="store_true")
    run.add_argument("--diagnostics-limit", type=int)

    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--manifest", required=True)
    aggregate_parser.add_argument("--input-root", required=True)
    aggregate_parser.add_argument("--output-root", required=True)
    aggregate_parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    aggregate_parser.add_argument("--require-complete", action="store_true")

    smoke = sub.add_parser("validate-smoke")
    smoke.add_argument("--manifest", required=True)
    smoke.add_argument("--raw-root", required=True)
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--expected-artifacts", type=int)
    smoke.add_argument("--require-diagnostics", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "manifest":
        build_frozen_manifest(args)
        return 0
    if args.command == "verify-run-contract":
        verify_run_contract(args)
        return 0
    if args.command == "run":
        return run_probe(args)
    if args.command == "aggregate":
        aggregate(args)
        return 0
    if args.command == "validate-smoke":
        validate_smoke(args)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
