#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MIN_BATCH_SIZE = 3
MIN_PREFILL_CALLS = 2
POLICY_MODE_ROOT = Path("default/image_text_image_text")


MODEL_SPECS = {
    "qwen25vl_3b_token_roll": {
        "family": "qwen2_5_vl",
        "stage": "legacy_v0_post_projector_pre_llm",
        "engine_mode": "v0",
        "ranks": {0},
        "batch_size": 2,
        "max_num_seqs": 2,
        "tp_size": 1,
    },
    "qwen25vl_32b_token_roll": {
        "family": "qwen2_5_vl",
        "stage": "legacy_v0_post_projector_pre_llm",
        "engine_mode": "v0",
        "ranks": {0, 1},
        "batch_size": 2,
        "max_num_seqs": 2,
        "tp_size": 2,
    },
    "minicpm_o_45_token_roll": {
        "family": "minicpm_o_4_5",
        "stage": "post_gather_pre_llm_embed_input_ids",
        "engine_mode": "v1",
        "ranks": {0},
        "batch_size": 2,
        "max_num_seqs": 2,
        "tp_size": 1,
    },
}


def contiguous_partition_refines(
    refinement: list[int],
    coarser: list[int],
) -> bool:
    if (
        not refinement
        or not coarser
        or any(value <= 0 for value in refinement)
        or any(value <= 0 for value in coarser)
        or sum(refinement) != sum(coarser)
    ):
        return False
    boundaries = set()
    offset = 0
    for value in refinement:
        offset += value
        boundaries.add(offset)
    offset = 0
    for value in coarser:
        offset += value
        if offset not in boundaries:
            return False
    return True


def smoke_binding_components(
    provenance: dict[str, Any],
    *,
    model_key: str,
    model_family: str,
    mode: str,
    engine_mode: str,
) -> dict[str, Any]:
    runtime = provenance.get("runtime_identity") or {}
    runtime_sources = runtime.get("source_identity") or {}
    model = provenance.get("model_identity") or {}
    normalized_files = sorted(
        (
            {
                key: item.get(key)
                for key in ("name", "size", "digest_kind", "sha256")
            }
            for item in model.get("files", [])
        ),
        key=lambda item: str(item.get("name")),
    )
    model_content = {
        "configured_path": model.get("configured_path"),
        "resolved_path": model.get("resolved_path"),
        "files": normalized_files,
    }
    model_content_sha256 = hashlib.sha256(
        json.dumps(model_content, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "model_key": model_key,
        "model_family": model_family,
        "mode": mode,
        "vllm_engine_mode": engine_mode,
        "mechanism_implementation_sha256": provenance.get(
            "mechanism_implementation_sha256"
        ),
        "validation_harness_sha256": provenance.get(
            "validation_harness_sha256"
        ),
        "repository_python_source_sha256": (
            provenance.get("repository_source_identity") or {}
        ).get("python_source_sha256"),
        "runtime": {
            "python": runtime.get("python"),
            "python_version": runtime.get("python_version"),
            "packages": runtime.get("packages"),
            "vllm_python_source_sha256": (
                runtime_sources.get("vllm") or {}
            ).get("python_source_sha256"),
            "transformers_python_source_sha256": (
                runtime_sources.get("transformers") or {}
            ).get("python_source_sha256"),
        },
        "model": {
            "configured_path": model.get("configured_path"),
            "resolved_path": model.get("resolved_path"),
            "content_sha256": model_content_sha256,
        },
    }


def smoke_certificate_entry(
    task_dir: Path,
    *,
    model_key: str,
    model_family: str,
    mode: str,
    engine_mode: str,
) -> dict[str, Any]:
    manifest = load_json(task_dir / "predictions/manifest.json")
    binding = smoke_binding_components(
        manifest.get("inference_provenance") or {},
        model_key=model_key,
        model_family=model_family,
        mode=mode,
        engine_mode=engine_mode,
    )
    required_values = (
        binding.get("mechanism_implementation_sha256"),
        binding.get("validation_harness_sha256"),
        binding.get("repository_python_source_sha256"),
        binding.get("runtime", {}).get("python"),
        binding.get("runtime", {}).get("python_version"),
        binding.get("runtime", {}).get("packages"),
        binding.get("runtime", {}).get("vllm_python_source_sha256"),
        binding.get("runtime", {}).get("transformers_python_source_sha256"),
        binding.get("model", {}).get("configured_path"),
        binding.get("model", {}).get("resolved_path"),
        binding.get("model", {}).get("content_sha256"),
    )
    if any(value in (None, "", {}) for value in required_values):
        raise ValueError(
            f"incomplete smoke-certificate identity for {engine_mode}/{model_key}/{mode}"
        )
    fingerprint = hashlib.sha256(
        json.dumps(binding, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"binding_fingerprint": fingerprint, "binding": binding}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_records(task_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in (task_dir / "_visual_token_shift").glob(
        "visual_token_shift.vllm.pid*.jsonl"
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                record["jsonl_path"] = str(path)
                records.append(record)
    return records


def load_call_records(task_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in (task_dir / "_visual_token_shift").glob(
        "visual_token_shift_calls.vllm.pid*.jsonl"
    ):
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return records


def load_runtime_contracts(task_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in (task_dir / "_visual_token_shift").glob(
        "vllm_runtime_contract.pid*.jsonl"
    ):
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return records


def load_prediction(task_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = task_dir / "predictions/manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prediction manifest missing: {manifest_path}")
    manifest = load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"prediction manifest is not complete: {manifest_path}")
    prediction_path = Path(str(manifest.get("prediction_file", "")))
    if not prediction_path.is_file():
        raise FileNotFoundError(f"prediction file missing: {prediction_path}")
    if prediction_path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(prediction_path)
    else:
        frame = pd.read_csv(prediction_path, sep="\t")
    if len(frame) != MIN_BATCH_SIZE:
        raise ValueError(f"expected {MIN_BATCH_SIZE} smoke rows, got {len(frame)}")
    if "index" not in frame or frame["index"].astype(str).nunique() != MIN_BATCH_SIZE:
        raise ValueError("smoke predictions do not contain three distinct sample indices")
    return manifest, frame


def output_signature(frame: pd.DataFrame) -> list[tuple[str, str]]:
    output_column = next(
        (name for name in ("prediction", "detailed_prediction", "description") if name in frame),
        None,
    )
    if output_column is None:
        raise ValueError("prediction file has no output column")
    rows = []
    for _, row in frame.sort_values("index", key=lambda values: values.astype(str)).iterrows():
        value = row[output_column]
        normalized = "" if pd.isna(value) else str(value)
        rows.append((str(row["index"]), normalized))
    return rows


def validate_raw_record(
    record: dict[str, Any],
    *,
    expected_mode: str,
    expected_stage: str,
    expected_family: str,
    expected_fingerprint: str,
) -> list[str]:
    failures = []
    audit = record.get("audit", {})
    pair_count = int(audit.get("request_pair_count", 0))
    if pair_count < 1:
        failures.append("detailed record has no request pairs")
    for key in (
        "input_ids_unchanged_exact",
        "is_multimodal_unchanged_exact",
        "item_token_coverage_exact",
        "final_mm_scatter_exact",
        "item_span_grouping_exact",
        "output_shape_matches_input_ids",
    ):
        if record.get(key) is not True:
            failures.append(f"{key} is not true")
    if record.get("strict") is not True:
        failures.append("strict mode is not enabled")
    if record.get("full_tensor_validation") is not True or record.get("validation_level") != "full":
        failures.append("record is not full tensor validation")
    if record.get("real_request") is not True or record.get("recording_armed") is not True:
        failures.append("record is not an armed real request")
    if record.get("inference_fingerprint") != expected_fingerprint:
        failures.append("record inference fingerprint mismatch")
    if expected_family == "qwen2_5_vl" and record.get("item_span_lengths_exact") is not True:
        failures.append("Qwen item/span alignment is not exact")
    if expected_family == "minicpm_o_4_5" and record.get("item_span_lengths_required") is not False:
        failures.append("MiniCPM incorrectly requires one span per image item")
    span_groups = record.get("item_span_groups") or []
    item_counts = record.get("item_token_counts") or []
    if len(span_groups) != len(item_counts) or any(
        sum(int(value) for value in group) != int(item_count)
        for group, item_count in zip(span_groups, item_counts)
    ):
        failures.append("per-image-item slice span grouping is invalid")
    if expected_family == "minicpm_o_4_5" and not any(len(group) > 1 for group in span_groups):
        failures.append("MiniCPM smoke did not exercise a multi-slice image item")
    if expected_family == "qwen2_5_vl":
        if record.get("qwen_grid_token_count_exact") is not True:
            failures.append("Qwen image_grid_thw token-count check is not exact")
        grid_rows = record.get("qwen_image_grid_thw") or []
        merge_size = int(record.get("qwen_spatial_merge_size") or 0)
        expected_counts = record.get("qwen_expected_visual_token_counts") or []
        actual_counts = record.get("qwen_actual_visual_token_counts") or []
        independent_counts = [
            int(np.prod(row)) // (merge_size * merge_size)
            for row in grid_rows
        ] if merge_size > 0 else []
        if not (
            independent_counts
            and independent_counts == expected_counts == actual_counts == item_counts
        ):
            failures.append("Qwen grid-derived and actual visual-token counts differ")
    if expected_family == "minicpm_o_4_5":
        query_num = int((record.get("model_metadata") or {}).get("query_num") or 0)
        placeholder_slice_counts = (
            record.get("minicpm_placeholder_slice_counts") or []
        )
        if query_num != 64:
            failures.append(f"MiniCPM query_num is not 64: {query_num}")
        if record.get("minicpm_placeholder_token_contract_exact") is not True:
            failures.append("MiniCPM placeholder/token contract is not exact")
        if not (
            placeholder_slice_counts
            and placeholder_slice_counts == [len(group) for group in span_groups]
            and all(
                all(int(span_length) == query_num for span_length in group)
                for group in span_groups
            )
        ) or any(
            int(item_count) != query_num * int(slice_count)
            for item_count, slice_count in zip(item_counts, placeholder_slice_counts)
        ):
            failures.append(
                "MiniCPM placeholder spans and LLM visual-token counts differ"
            )
    for key in (
        "model_name",
        "input_ids_sha256",
        "is_multimodal_sha256",
        "final_mm_scatter_sha256",
    ):
        if not record.get(key):
            failures.append(f"{key} is missing")
    if record.get("mode") != expected_mode:
        failures.append(f"unexpected mode {record.get('mode')!r}")
    if record.get("stage") != expected_stage:
        failures.append(f"unexpected stage {record.get('stage')!r}")

    expected_shift = 0 if expected_mode == "noop_vllm" else 1
    pair_records = audit.get("pair_records", [])
    if len(pair_records) != pair_count:
        failures.append("pair record count differs from request_pair_count")
    for pair in pair_records:
        for key in (
            "i1_i2_equal_exact",
            "i1_unchanged_exact",
            "i2_source_unchanged_exact",
            "i2_roll_exact",
            "i2_out_of_place",
        ):
            if pair.get(key) is not True:
                failures.append(f"pair {pair.get('pair_index')} failed {key}")
        if pair.get("shift") != expected_shift or pair.get("max_abs_error") != 0.0:
            failures.append(f"pair {pair.get('pair_index')} shift/error mismatch")

    raw_path = Path(str(record.get("raw_npz_path", "")))
    if not raw_path.is_file():
        failures.append(f"raw NPZ missing: {raw_path}")
        return failures
    with np.load(raw_path) as raw:
        shifted_items = []
        for pair_index in range(pair_count):
            before = raw[f"pair{pair_index}_i2_before"]
            after = raw[f"pair{pair_index}_i2_after"]
            source = raw[f"pair{pair_index}_source_index_for_output"].astype(int)
            token_count = int(before.shape[0])
            expected_source = (
                np.arange(token_count, dtype=int)
                if expected_mode == "noop_vllm"
                else np.concatenate(
                    [np.asarray([token_count - 1], dtype=int), np.arange(token_count - 1)]
                )
            )
            if not np.array_equal(source, expected_source):
                failures.append(
                    f"raw pair {pair_index} source index is not the independent expected mapping"
                )
            if not np.array_equal(after, before[source]):
                failures.append(f"raw pair {pair_index} is not the expected roll")
            if not np.array_equal(
                raw[f"pair{pair_index}_i1_before"],
                raw[f"pair{pair_index}_i1_after"],
            ):
                failures.append(f"raw pair {pair_index} changed I1")
            shifted_items.extend(
                [
                    raw[f"pair{pair_index}_i1_after_bytes"].reshape(-1),
                    raw[f"pair{pair_index}_i2_after_bytes"].reshape(-1),
                ]
            )
        if not np.array_equal(
            raw["final_mm_scatter_bytes"].reshape(-1),
            np.concatenate(shifted_items),
        ):
            failures.append("final LLM scatter bytes differ from shifted [I1,I2,...]")
    return failures


def validate_condition(
    task_dir: Path,
    *,
    model_key: str,
    mode: str,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]], dict[str, Any], list[str]]:
    failures = []
    try:
        manifest, frame = load_prediction(task_dir)
    except Exception as exc:
        return [], [], {}, [str(exc)]
    runtime = manifest.get("runtime", {})
    provenance = manifest.get("inference_provenance") or {}
    inference_fingerprint = provenance.get("fingerprint")
    if not inference_fingerprint or provenance.get("schema_version") != 2:
        failures.append("prediction manifest lacks schema-v2 inference provenance")
    dataset_artifacts = provenance.get("dataset_identity", {}).get("artifacts", [])
    runtime_sources = provenance.get("runtime_identity", {}).get("source_identity", {})
    model_files = provenance.get("model_identity", {}).get("files", [])
    if not provenance.get("repository_source_identity", {}).get("python_source_sha256"):
        failures.append("prediction provenance lacks the full VLMEvalKit source-tree hash")
    if not provenance.get("mechanism_implementation_sha256"):
        failures.append("prediction provenance lacks the matrix-independent mechanism hash")
    if not provenance.get("validation_harness_sha256"):
        failures.append("prediction provenance lacks the real-smoke validation harness hash")
    if not dataset_artifacts or any(
        item.get("digest_kind") != "full_sha256" or not item.get("sha256")
        for item in dataset_artifacts
    ):
        failures.append("prediction provenance lacks full dataset-content hashes")
    if not all(
        runtime_sources.get(name, {}).get("python_source_sha256")
        for name in ("vllm", "transformers")
    ):
        failures.append("prediction provenance lacks runtime source-tree hashes")
    if not model_files or any(not item.get("sha256") for item in model_files):
        failures.append("prediction provenance lacks model file content digests")
    if runtime.get("infer_batch_size") != spec["batch_size"]:
        failures.append(f"infer_batch_size != {spec['batch_size']}")
    if runtime.get("tp_size") != spec["tp_size"]:
        failures.append(f"tp_size != {spec['tp_size']}")
    if runtime.get("max_num_seqs") != spec["max_num_seqs"]:
        failures.append(f"max_num_seqs != {spec['max_num_seqs']}")

    semantic_env = dict(provenance.get("semantic_env") or {})
    semantic_env.pop("REPLAY_VISUAL_TOKEN_SHIFT", None)
    task_dir_text = str(task_dir.resolve())
    normalized_semantic_env = {
        key: (
            value.replace(task_dir_text, "<TASK_DIR>")
            if isinstance(value, str)
            else value
        )
        for key, value in semantic_env.items()
    }

    run_id = manifest.get("runtime_contract_run_id")
    contracts = [
        record
        for record in load_runtime_contracts(task_dir)
        if record.get("run_id") == run_id
    ]
    contracts = sorted(contracts, key=lambda item: float(item.get("timestamp", 0.0)))
    if len(contracts) < MIN_PREFILL_CALLS:
        failures.append(
            f"runtime contract count {len(contracts)} < {MIN_PREFILL_CALLS}"
        )
        runtime_contract = {}
    else:
        request_hashes = [
            request_hash
            for contract in contracts
            for request_hash in contract.get("request_hashes", [])
        ]
        request_count = sum(int(contract.get("request_count", -1)) for contract in contracts)
        contract_request_counts = [
            int(contract.get("request_count", -1)) for contract in contracts
        ]
        expected_request_counts = []
        remaining_request_count = MIN_BATCH_SIZE
        while remaining_request_count > 0:
            call_request_count = min(spec["batch_size"], remaining_request_count)
            expected_request_counts.append(call_request_count)
            remaining_request_count -= call_request_count
        sampling_contracts = [contract.get("sampling_contract") for contract in contracts]
        request_metadata_by_call = [
            list(contract.get("request_metadata", [])) for contract in contracts
        ]
        request_metadata = [
            metadata for call_metadata in request_metadata_by_call
            for metadata in call_metadata
        ]
        runtime_configs = [
            contract.get("actual_vllm_runtime_config") for contract in contracts
        ]
        if any(
            contract.get("mode") != mode
            or contract.get("model_family") != spec["family"]
            or contract.get("request_identity_level") != "full"
            or int(contract.get("effective_infer_batch_size", -1)) != spec["batch_size"]
            or int(contract.get("effective_max_num_seqs", -1)) != spec["max_num_seqs"]
            or contract.get("actual_vllm_engine_mode") != spec["engine_mode"]
            or not contract.get("actual_vllm_engine_module")
            or not contract.get("actual_vllm_engine_qualname")
            or not contract.get("actual_vllm_runtime_config")
            or contract.get("inference_fingerprint") != inference_fingerprint
            for contract in contracts
        ) or (
            request_count != MIN_BATCH_SIZE
            or contract_request_counts != expected_request_counts
            or len(request_hashes) != MIN_BATCH_SIZE
            or len(set(request_hashes)) != MIN_BATCH_SIZE
            or len({json.dumps(item, sort_keys=True) for item in sampling_contracts}) != 1
            or len({json.dumps(item, sort_keys=True) for item in runtime_configs}) != 1
        ):
            failures.append("runtime request/sampling contract is invalid")
        if spec["family"] == "minicpm_o_4_5":
            if any(
                len(call_metadata) != call_request_count
                or [
                    int(metadata.get("request_index", -1))
                    for metadata in call_metadata
                ]
                != list(range(call_request_count))
                for call_metadata, call_request_count in zip(
                    request_metadata_by_call,
                    contract_request_counts,
                )
            ) or len(request_metadata) != request_count or any(
                metadata.get("image_count") != 2
                or len(metadata.get("image_sha256") or []) != 2
                or len(metadata.get("image_sizes") or []) != 2
                or len(metadata.get("minicpm_num_slices") or []) != 2
                or metadata["image_sha256"][0] != metadata["image_sha256"][1]
                or metadata["minicpm_num_slices"][0]
                != metadata["minicpm_num_slices"][1]
                or any(
                    int(value) <= 0
                    for value in metadata.get("minicpm_num_slices") or []
                )
                for metadata in request_metadata
            ):
                failures.append("MiniCPM driver request slice metadata is invalid")
            if len(
                {
                    tuple(metadata.get("image_sha256") or [])
                    for metadata in request_metadata
                }
            ) != request_count:
                failures.append("MiniCPM sentinel driver image hashes are not distinct")
            if any(
                len(
                    {
                        tuple(
                            int(value)
                            for value in metadata.get(
                                "minicpm_num_slices",
                                [],
                            )
                        )
                        for metadata in call_metadata
                    }
                )
                != len(call_metadata)
                for call_metadata in request_metadata_by_call
                if len(call_metadata) > 1
            ):
                failures.append(
                    "MiniCPM sentinel slices are not request-distinguishing "
                    "within each driver call"
                )
        elif request_metadata:
            failures.append("Qwen runtime unexpectedly contains MiniCPM request metadata")
        if any(
            not isinstance(config, dict)
            or int(config.get("max_num_seqs") or -1) != spec["max_num_seqs"]
            or int(config.get("tensor_parallel_size") or -1) != spec["tp_size"]
            or config.get("enable_chunked_prefill") is not False
            or config.get("disable_chunked_mm_input") is not True
            or config.get("enable_prefix_caching") is not False
            or int(config.get("max_model_len") or 0) <= 0
            or int(config.get("max_num_batched_tokens") or 0) <= 0
            or not isinstance(config.get("limit_mm_per_prompt"), dict)
            for config in runtime_configs
        ):
            failures.append("concrete vLLM runtime configuration is invalid")
        runtime_contract = {
            "request_count": request_count,
            "request_hashes": request_hashes,
            "sampling_contract": sampling_contracts[0],
            "contract_count": len(contracts),
            "actual_vllm_engine_mode": contracts[0].get(
                "actual_vllm_engine_mode"
            ),
            "actual_vllm_engine_module": contracts[0].get(
                "actual_vllm_engine_module"
            ),
            "actual_vllm_engine_qualname": contracts[0].get(
                "actual_vllm_engine_qualname"
            ),
            "actual_vllm_runtime_config": contracts[0].get(
                "actual_vllm_runtime_config"
            ),
            "request_metadata": request_metadata,
            "request_metadata_by_call": request_metadata_by_call,
            "request_counts_by_call": contract_request_counts,
            "normalized_semantic_env": normalized_semantic_env,
        }
    if not run_id:
        failures.append("runtime contract run id is missing from prediction manifest")

    handshakes = []
    for path in (task_dir / "_visual_token_shift").glob(
        f"worker_handshake.{spec['family']}.pid*.json"
    ):
        payload = load_json(path)
        if payload.get("run_id") == run_id:
            handshakes.append(payload)
    handshake_ranks = {int(item.get("rank", -1)) for item in handshakes}
    if handshake_ranks != spec["ranks"] or len(handshakes) != len(spec["ranks"]):
        failures.append(
            f"worker handshake ranks {sorted(handshake_ranks)} != {sorted(spec['ranks'])}"
        )
    if any(
        item.get("phase") != "worker_model_initialized"
        or item.get("backend") != "vllm"
        or item.get("mode") != mode
        or item.get("model_family") != spec["family"]
        or item.get("target_family") != spec["family"]
        or item.get("inference_fingerprint") != inference_fingerprint
        or int(item.get("scheduler_max_num_seqs", -1)) != spec["max_num_seqs"]
        or int(item.get("scheduler_max_num_batched_tokens", -1))
        != int(
            (runtime_contract.get("actual_vllm_runtime_config") or {}).get(
                "max_num_batched_tokens",
                -2,
            )
        )
        or item.get("scheduler_enable_chunked_prefill") is not False
        or item.get("scheduler_disable_chunked_mm_input") is not True
        or item.get("cache_enable_prefix_caching") is not False
        or item.get("vllm_engine_mode") != runtime_contract.get(
            "actual_vllm_engine_mode"
        )
        or not item.get("model_class")
        or not item.get("model_name")
        for item in handshakes
    ):
        failures.append("worker handshake mode/model family/runtime mismatch")

    if mode == "none":
        evidence = manifest.get("mechanism_evidence") or {}
        if (
            evidence.get("verified") is not True
            or evidence.get("control") is not True
            or evidence.get("mode") != "none"
            or evidence.get("run_id") != run_id
            or evidence.get("inference_fingerprint") != inference_fingerprint
            or int(evidence.get("validated_request_count", -1)) != MIN_BATCH_SIZE
            or evidence.get("actual_vllm_engine_mode")
            != runtime_contract.get("actual_vllm_engine_mode")
            or evidence.get("actual_vllm_runtime_config")
            != runtime_contract.get("actual_vllm_runtime_config")
        ):
            failures.append("matched none manifest lacks verified runtime evidence")
        if load_records(task_dir):
            failures.append("native none control unexpectedly produced shift records")
        if load_call_records(task_dir):
            failures.append("native none control unexpectedly produced shift call audits")
        return [], output_signature(frame), runtime_contract, failures

    evidence = manifest.get("mechanism_evidence") or {}
    if (
        evidence.get("verified") is not True
        or evidence.get("mode") != mode
        or evidence.get("inference_fingerprint") != inference_fingerprint
        or int(evidence.get("validated_request_count", -1)) != MIN_BATCH_SIZE
    ):
        failures.append("prediction manifest lacks matching mechanism evidence")
    evidence_run_id = evidence.get("run_id")
    if evidence_run_id != run_id:
        failures.append("mechanism evidence and runtime contract run ids differ")
    records = [record for record in load_records(task_dir) if record.get("run_id") == run_id]
    ranks = {int(record.get("rank", -1)) for record in records}
    if ranks != spec["ranks"]:
        failures.append(f"record ranks {sorted(ranks)} != {sorted(spec['ranks'])}")
    for rank in spec["ranks"]:
        rank_records = [record for record in records if int(record.get("rank", -1)) == rank]
        rank_records.sort(key=lambda item: int(item.get("call_index", -1)))
        if len(rank_records) < MIN_PREFILL_CALLS:
            failures.append(
                f"rank {rank} raw record count {len(rank_records)} < {MIN_PREFILL_CALLS}"
            )
            continue
        for record in rank_records:
            if record.get("model_family") != spec["family"]:
                failures.append(f"rank {rank} model family mismatch")
            failures.extend(
                f"rank {rank} call {record.get('call_index')}: {failure}"
                for failure in validate_raw_record(
                    record,
                    expected_mode=mode,
                    expected_stage=spec["stage"],
                    expected_family=spec["family"],
                    expected_fingerprint=inference_fingerprint,
                )
            )
        raw_pair_fingerprints = [
            (tuple(pair.get("shape") or []), pair.get("i1_before_sha256"))
            for record in rank_records
            for pair in (record.get("audit") or {}).get("pair_records", [])
        ]
        if len(raw_pair_fingerprints) != int(runtime_contract.get("request_count", -1)):
            failures.append(f"rank {rank} raw pair coverage differs from runtime contract")
        if len(set(raw_pair_fingerprints)) != len(raw_pair_fingerprints):
            failures.append(f"rank {rank} sentinel request visual fingerprints are not distinct")
        if spec["family"] == "minicpm_o_4_5":
            raw_pair_counts = [
                len((record.get("audit") or {}).get("pair_records", []))
                for record in rank_records
            ]
            if not contiguous_partition_refines(
                raw_pair_counts,
                runtime_contract.get("request_counts_by_call", []),
            ):
                failures.append(
                    f"rank {rank} MiniCPM worker partition {raw_pair_counts} "
                    "does not refine driver calls"
                )
            driver_slice_pairs = [
                tuple(
                    int(value)
                    for value in metadata.get("minicpm_num_slices", [])
                )
                for metadata in runtime_contract.get("request_metadata", [])
            ]
            worker_slice_pairs = []
            worker_pair_token_counts = []
            for record in rank_records:
                pair_records = (record.get("audit") or {}).get("pair_records", [])
                slice_counts = record.get("minicpm_placeholder_slice_counts") or []
                if len(slice_counts) != 2 * len(pair_records):
                    failures.append(
                        f"rank {rank} MiniCPM slice counts do not cover IQIQ pairs"
                    )
                    continue
                for pair_index, pair_record in enumerate(pair_records):
                    worker_slice_pairs.append(
                        tuple(
                            int(value)
                            for value in slice_counts[2 * pair_index : 2 * pair_index + 2]
                        )
                    )
                    worker_pair_token_counts.append(
                        int((pair_record.get("shape") or [0])[0])
                    )
            if worker_slice_pairs != driver_slice_pairs:
                failures.append(
                    f"rank {rank} MiniCPM driver and worker slices differ: "
                    f"driver={driver_slice_pairs} worker={worker_slice_pairs}"
                )
            if any(
                slices[0] != slices[1]
                or token_count != 64 * slices[1]
                for slices, token_count in zip(
                    worker_slice_pairs,
                    worker_pair_token_counts,
                )
            ):
                failures.append(
                    f"rank {rank} MiniCPM raw pair shape differs from driver slices"
                )
    call_records = [
        record for record in load_call_records(task_dir) if record.get("run_id") == run_id
    ]
    call_ranks = {int(record.get("rank", -1)) for record in call_records}
    if call_ranks != spec["ranks"]:
        failures.append(f"call-audit ranks {sorted(call_ranks)} != {sorted(spec['ranks'])}")
    pair_counts_by_rank = {}
    for rank in spec["ranks"]:
        rank_calls = [
            record for record in call_records if int(record.get("rank", -1)) == rank
        ]
        rank_calls.sort(key=lambda item: int(item.get("call_index", -1)))
        if len(rank_calls) < MIN_PREFILL_CALLS:
            failures.append(
                f"rank {rank} smoke call-audit count {len(rank_calls)} < {MIN_PREFILL_CALLS}"
            )
            continue
        expected_indices = list(range(1, len(rank_calls) + 1))
        if [int(call.get("call_index", -1)) for call in rank_calls] != expected_indices:
            failures.append(f"rank {rank} call indices are not contiguous")
        for call in rank_calls:
            if (
                call.get("all_iqiq_pairs_equal_exact") is not True
                or call.get("input_ids_unchanged_exact") is not True
                or call.get("is_multimodal_unchanged_exact") is not True
                or call.get("final_mm_scatter_exact") is not True
                or call.get("item_token_coverage_exact") is not True
                or call.get("item_span_grouping_exact") is not True
                or call.get("output_shape_matches_input_ids") is not True
                or call.get("full_tensor_validation") is not True
                or call.get("validation_level") != "full"
                or (
                    spec["family"] == "qwen2_5_vl"
                    and (
                        call.get("item_span_lengths_exact") is not True
                        or call.get("qwen_grid_token_count_exact") is not True
                    )
                )
                or (
                    spec["family"] == "minicpm_o_4_5"
                    and (
                        int(call.get("minicpm_query_num") or 0) != 64
                        or call.get("minicpm_placeholder_token_contract_exact")
                        is not True
                        or not call.get("minicpm_placeholder_slice_counts")
                        or any(
                            int(span_length) != 64
                            for group in call.get("item_span_groups") or []
                            for span_length in group
                        )
                    )
                )
                or call.get("real_request") is not True
                or call.get("recording_armed") is not True
                or call.get("inference_fingerprint") != inference_fingerprint
            ):
                failures.append(
                    f"rank {rank} smoke call {call.get('call_index')} audit is invalid"
                )
        pair_counts_by_rank[rank] = [
            int(call.get("request_pair_count", -1)) for call in rank_calls
        ]
        if not contiguous_partition_refines(
            pair_counts_by_rank[rank],
            runtime_contract.get("request_counts_by_call", []),
        ):
            failures.append(
                f"rank {rank} prefill partition does not refine driver calls"
            )
        raw_call_indices = sorted(
            int(record.get("call_index", -1))
            for record in records
            if int(record.get("rank", -1)) == rank
        )
        if raw_call_indices != expected_indices:
            failures.append(
                f"rank {rank} raw dump calls {raw_call_indices} != audited calls {expected_indices}"
            )
    if len({tuple(counts) for counts in pair_counts_by_rank.values()}) > 1:
        failures.append(f"TP prefill call partitions differ: {pair_counts_by_rank}")
    return records, output_signature(frame), runtime_contract, failures


def runtime_contract_signature(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_count": record.get("request_count"),
        "request_hashes": record.get("request_hashes"),
        "sampling_contract": record.get("sampling_contract"),
        "actual_vllm_engine_mode": record.get("actual_vllm_engine_mode"),
        "actual_vllm_engine_module": record.get("actual_vllm_engine_module"),
        "actual_vllm_engine_qualname": record.get("actual_vllm_engine_qualname"),
        "actual_vllm_runtime_config": record.get("actual_vllm_runtime_config"),
        "request_metadata": record.get("request_metadata"),
        "request_counts_by_call": record.get("request_counts_by_call"),
        "request_metadata_by_call": record.get("request_metadata_by_call"),
        "normalized_semantic_env": record.get("normalized_semantic_env"),
    }


def tp_hash_signature(record: dict[str, Any]) -> dict[str, Any]:
    raw_path = Path(record["raw_npz_path"])
    with np.load(raw_path) as raw:
        raw_hashes = {
            key: hashlib.sha256(
                np.ascontiguousarray(raw[key]).view(np.uint8).tobytes()
            ).hexdigest()
            for key in sorted(raw.files)
            if key == "input_ids"
            or key == "is_multimodal"
            or key == "final_mm_scatter_bytes"
            or key.endswith("_bytes")
            or key.endswith("_source_index_for_output")
        }
    return {
        "model_name": record.get("model_name"),
        "raw_hashes": raw_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures = []
    accepted: dict[str, Any] = {}
    certificate_entries: list[dict[str, Any]] = []
    for model_key, spec in MODEL_SPECS.items():
        signatures = {}
        runtime_contracts = {}
        mode_records = {}
        mode_bindings = {}
        for mode in ("none", "noop_vllm", "roll_right_1"):
            task_dir = args.root / POLICY_MODE_ROOT / mode / model_key / "MathVision"
            records, signature, runtime_contract, condition_failures = validate_condition(
                task_dir,
                model_key=model_key,
                mode=mode,
                spec=spec,
            )
            label = f"{spec['engine_mode']}/{model_key}/{mode}"
            accepted[label] = {
                "record_ranks": sorted({int(record["rank"]) for record in records}),
                "failures": condition_failures,
            }
            failures.extend(f"{label}: {failure}" for failure in condition_failures)
            try:
                actual_engine_mode = runtime_contract.get(
                    "actual_vllm_engine_mode"
                )
                if actual_engine_mode != spec["engine_mode"]:
                    raise ValueError(
                        "runtime engine differs from model specification: "
                        f"{actual_engine_mode!r} != {spec['engine_mode']!r}"
                    )
                entry = smoke_certificate_entry(
                    task_dir,
                    model_key=model_key,
                    model_family=spec["family"],
                    mode=mode,
                    engine_mode=actual_engine_mode,
                )
                mode_bindings[mode] = {
                    key: value
                    for key, value in entry["binding"].items()
                    if key != "mode"
                }
                if mode != "none":
                    certificate_entries.append(entry)
            except Exception as exc:
                failures.append(f"{label}: failed to build smoke certificate: {exc}")
            signatures[mode] = signature
            runtime_contracts[mode] = runtime_contract
            mode_records[mode] = records
        if signatures["none"] != signatures["noop_vllm"]:
            failures.append(
                f"{spec['engine_mode']}/{model_key}: native none and "
                "matched-runtime noop outputs differ"
            )
        if not (
            runtime_contract_signature(runtime_contracts["none"])
            == runtime_contract_signature(runtime_contracts["noop_vllm"])
            == runtime_contract_signature(runtime_contracts["roll_right_1"])
        ):
            failures.append(
                f"{spec['engine_mode']}/{model_key}: request or "
                "SamplingParams contracts differ"
            )
        if len(mode_bindings) != 3 or len(
            {
                json.dumps(binding, ensure_ascii=False, sort_keys=True)
                for binding in mode_bindings.values()
            }
        ) != 1:
            failures.append(
                f"{spec['engine_mode']}/{model_key}: "
                "code/runtime/model identity differs across controls"
            )
        if model_key == "qwen25vl_32b_token_roll":
            for mode in ("noop_vllm", "roll_right_1"):
                by_rank = {}
                for rank in (0, 1):
                    rank_map = {}
                    for record in mode_records[mode]:
                        if int(record["rank"]) != rank:
                            continue
                        call_index = int(record["call_index"])
                        if call_index in rank_map:
                            failures.append(
                                f"{spec['engine_mode']}/{model_key}/{mode}: "
                                f"duplicate TP call index {call_index} on rank {rank}"
                            )
                            continue
                        rank_map[call_index] = tp_hash_signature(record)
                    by_rank[rank] = rank_map
                if all(by_rank.values()) and by_rank[0] != by_rank[1]:
                    failures.append(
                        f"{spec['engine_mode']}/{model_key}/{mode}: "
                        "TP rank embedding hashes differ"
                    )

    summary = {
        "all_passed": not failures,
        "accepted_conditions": accepted,
        "failures": failures,
        "contract": {
            "minimum_actual_batch_size": MIN_BATCH_SIZE,
            "controls": ["none", "noop_vllm", "roll_right_1"],
            "qwen32_tp_hash_equality_required": True,
            "minimum_prefill_calls_per_rank": MIN_PREFILL_CALLS,
            "qwen32_v0_tp2_required": True,
        },
        "smoke_certificate": {
            "schema_version": 1,
            "valid": not failures,
            "entries": sorted(
                certificate_entries,
                key=lambda item: (
                    item["binding"]["vllm_engine_mode"],
                    item["binding"]["model_key"],
                    item["binding"]["mode"],
                ),
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
