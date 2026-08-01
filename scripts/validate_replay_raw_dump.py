#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import sys


EXPECTED_TYPES = {
    "iq": ["image", "text"],
    "iqiq": ["image", "text", "image", "text"],
}
EXPECTED_MODES = {
    "iq": "image_text",
    "iqiq": "image_text_image_text",
}


def _item_value(item):
    for key in ("value", "image", "text"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _sha256(path_text):
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(path_text)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_rgb_sha256(path_text):
    from PIL import Image

    path_text = str(path_text)
    if path_text.startswith("file://"):
        path_text = path_text[7:]
    with Image.open(path_text) as image:
        image = image.convert("RGB")
        width, height = image.size
        digest = hashlib.sha256()
        digest.update(f"PIL\0RGB\0{width}x{height}\0".encode("utf-8"))
        digest.update(image.tobytes())
        return digest.hexdigest()


def _decoded_rgb_sha256_at_size(path_text, size):
    from PIL import Image

    path_text = str(path_text)
    if path_text.startswith("file://"):
        path_text = path_text[7:]
    width = int(size["width"])
    height = int(size["height"])
    with Image.open(path_text) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height))
        digest = hashlib.sha256()
        digest.update(f"PIL\0RGB\0{width}x{height}\0".encode("utf-8"))
        digest.update(image.tobytes())
        return digest.hexdigest()


def _flatten_strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings = []
        for item in value.values():
            strings.extend(_flatten_strings(item))
        return strings
    if isinstance(value, (list, tuple)):
        strings = []
        for item in value:
            strings.extend(_flatten_strings(item))
        return strings
    return []


def _load_allowlist(path):
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError(f"No canonical indices in {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate canonical indices in {path}")
    return values


def _load_predictions(path):
    import pandas as pd

    suffix = path.suffix.lower()
    kwargs = {"dtype": str, "keep_default_na": False}
    if suffix == ".xlsx":
        data = pd.read_excel(path, **kwargs)
    elif suffix == ".csv":
        data = pd.read_csv(path, **kwargs)
    elif suffix == ".tsv":
        data = pd.read_csv(path, sep="\t", **kwargs)
    else:
        raise ValueError(f"Unsupported prediction file: {path}")
    if "index" not in data or "prediction" not in data:
        raise ValueError("Prediction file must contain index and prediction columns")
    data["index"] = data["index"].map(lambda value: str(value).strip())
    if data["index"].duplicated().any():
        raise ValueError("Prediction file contains duplicate canonical indices")
    return data


def validate_record(record, condition):
    """Validate the replay-policy stage for one sample."""
    errors = []
    before = record.get("before_content", [])
    after = record.get("after_content", [])
    if not isinstance(before, list) or not all(isinstance(item, dict) for item in before):
        before = []
        errors.append("before_content is not a list of objects")
    if not isinstance(after, list) or not all(isinstance(item, dict) for item in after):
        after = []
        errors.append("after_content is not a list of objects")

    if record.get("stage") not in (None, "replay_policy"):
        errors.append(f"record stage {record.get('stage')!r} != 'replay_policy'")
    if record.get("mode") != EXPECTED_MODES[condition]:
        errors.append(
            f"record mode {record.get('mode')!r} != {EXPECTED_MODES[condition]!r}"
        )

    before_types = [item.get("type") for item in before]
    after_types = [item.get("type") for item in after]
    if before_types != EXPECTED_TYPES["iq"]:
        errors.append(f"before types {before_types} != {EXPECTED_TYPES['iq']}")
    if after_types != EXPECTED_TYPES[condition]:
        errors.append(f"after types {after_types} != {EXPECTED_TYPES[condition]}")

    expected_before_counts = {"text": 1, "image": 1, "video": 0}
    expected_after_counts = {
        "text": 1 if condition == "iq" else 2,
        "image": 1 if condition == "iq" else 2,
        "video": 0,
    }
    if record.get("before_counts") != expected_before_counts:
        errors.append(f"before counts {record.get('before_counts')} != {expected_before_counts}")
    if record.get("after_counts") != expected_after_counts:
        errors.append(f"after counts {record.get('after_counts')} != {expected_after_counts}")

    image_hashes = []
    for item in after:
        if item.get("type") != "image":
            continue
        try:
            image_hashes.append(_decoded_rgb_sha256(_item_value(item)))
        except Exception as exc:
            errors.append(f"image hash failed: {type(exc).__name__}: {exc}")

    texts = [_item_value(item) for item in after if item.get("type") == "text"]
    before_image_hashes = []
    for item in before:
        if item.get("type") == "image":
            try:
                before_image_hashes.append(_decoded_rgb_sha256(_item_value(item)))
            except Exception as exc:
                errors.append(f"before image hash failed: {type(exc).__name__}: {exc}")
    before_texts = [_item_value(item) for item in before if item.get("type") == "text"]
    if len(before_image_hashes) == 1 and image_hashes and before_image_hashes[0] != image_hashes[0]:
        errors.append("source image differs between before and after")
    if len(before_texts) == 1 and texts and before_texts[0] != texts[0]:
        errors.append("source question differs between before and after")
    if condition == "iqiq" and len(image_hashes) == 2 and image_hashes[0] != image_hashes[1]:
        errors.append("I1 and I2 content hashes differ")
    if condition == "iqiq" and len(texts) == 2 and texts[0] != texts[1]:
        errors.append("Q1 and Q2 differ")

    text_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    return {
        "tag": record.get("tag"),
        "mode": record.get("mode"),
        "before_types": before_types,
        "after_types": after_types,
        "image_sha256": image_hashes,
        "image_refs": [_item_value(item) for item in after if item.get("type") == "image"],
        "texts": texts,
        "text_sha256": text_hashes,
        "sample_fingerprint": hashlib.sha256(
            "|".join(image_hashes[:1] + text_hashes[:1]).encode("utf-8")
        ).hexdigest(),
        "errors": errors,
    }


def validate_final_record(
    record,
    *,
    condition,
    expected_identity,
    model_family,
    backend,
    policy_check,
):
    errors = []
    if record.get("schema_version") != "final_model_input.v1":
        errors.append(f"schema_version {record.get('schema_version')!r} is not final_model_input.v1")
    if record.get("stage") != "final_model_input":
        errors.append(f"stage {record.get('stage')!r} is not final_model_input")
    if record.get("model_family") != model_family:
        errors.append(f"model_family {record.get('model_family')!r} != {model_family!r}")
    if record.get("backend") != backend:
        errors.append(f"backend {record.get('backend')!r} != {backend!r}")

    identity = record.get("task_identity")
    if not isinstance(identity, dict):
        identity = {}
        errors.append("task_identity is missing")
    for field, expected in expected_identity.items():
        if str(identity.get(field)) != str(expected):
            errors.append(f"task_identity.{field} {identity.get(field)!r} != {expected!r}")
    if record.get("missing_task_identity_fields"):
        errors.append(f"missing task identity fields: {record.get('missing_task_identity_fields')}")

    visuals = record.get("visual_inputs")
    if not isinstance(visuals, list) or not all(isinstance(item, dict) for item in visuals):
        visuals = []
        errors.append("visual_inputs is not a list of objects")
    expected_count = 1 if condition == "iq" else 2
    if record.get("visual_input_count") != expected_count or len(visuals) != expected_count:
        errors.append(
            f"final visual count {record.get('visual_input_count')}/{len(visuals)} != {expected_count}"
        )
    hashes = [item.get("sha256") for item in visuals]
    if any(not value for value in hashes):
        errors.append("one or more final visual inputs have no content hash")
    if any(item.get("unresolved") for item in visuals):
        errors.append("one or more final visual inputs are unresolved")
    if condition == "iqiq" and len(hashes) == 2 and hashes[0] != hashes[1]:
        errors.append("final I1 and I2 visual hashes differ")

    modalities = [item.get("modality") for item in visuals]
    expected_modalities = ["image"] * expected_count
    if modalities != expected_modalities:
        errors.append(f"final visual modalities {modalities} != {expected_modalities}")
    source_refs = [str(item.get("source_ref")) for item in visuals]
    if source_refs != policy_check["image_refs"]:
        errors.append(
            f"final visual source refs {source_refs} != policy refs {policy_check['image_refs']}"
        )
    source_hashes = [item.get("source_sha256") for item in visuals]
    if source_hashes != policy_check["image_sha256"]:
        errors.append(
            "final visual source hashes do not match replay-policy decoded image hashes"
        )
    expected_content_hashes = []
    for item, source_ref in zip(visuals, policy_check["image_refs"]):
        try:
            size = item.get("size")
            if not isinstance(size, dict) or set(size) != {"width", "height"}:
                raise ValueError(f"invalid final visual size: {size!r}")
            expected_content_hashes.append(
                _decoded_rgb_sha256_at_size(source_ref, size)
            )
        except Exception as exc:
            expected_content_hashes.append(None)
            errors.append(
                f"expected final visual hash failed: {type(exc).__name__}: {exc}"
            )
    if hashes != expected_content_hashes:
        errors.append(
            "final visual content hashes do not match source images resized to consumer dimensions"
        )

    sequence = record.get("content_sequence")
    if not isinstance(sequence, list) or not all(isinstance(item, dict) for item in sequence):
        sequence = []
        errors.append("content_sequence is not a list of objects")
    sequence_types = [item.get("type") for item in sequence]
    if sequence_types != EXPECTED_TYPES[condition]:
        errors.append(
            f"final content sequence {sequence_types} != {EXPECTED_TYPES[condition]}"
        )
    visual_positions = [item.get("visual_position") for item in sequence if item.get("type") == "image"]
    if visual_positions != list(range(expected_count)):
        errors.append(
            f"final content visual positions {visual_positions} != {list(range(expected_count))}"
        )
    sequence_text_hashes = [
        item.get("text_sha256") for item in sequence if item.get("type") == "text"
    ]
    if sequence_text_hashes != policy_check["text_sha256"]:
        errors.append("final content text hashes do not match replay-policy text hashes")

    strings = _flatten_strings(record.get("text_chat_representation"))
    question = policy_check["texts"][0] if policy_check["texts"] else ""
    question_occurrences = sum(value.count(question) for value in strings) if question else 0
    expected_occurrences = 1 if condition == "iq" else 2
    if question_occurrences != expected_occurrences:
        errors.append(
            f"final question occurrences {question_occurrences} != {expected_occurrences}"
        )
    generation_config = record.get("generation_config")
    if generation_config is not None and not isinstance(generation_config, dict):
        errors.append("generation_config is not an object")
        generation_config = None
    return {
        "canonical_index": identity.get("canonical_index"),
        "batch_position": record.get("batch_position"),
        "visual_sha256": hashes,
        "visual_source_sha256": source_hashes,
        "expected_visual_sha256": expected_content_hashes,
        "content_sequence_types": sequence_types,
        "question_occurrences": question_occurrences,
        "consumer_api": record.get("consumer_api"),
        "generation_config": generation_config,
        "errors": errors,
    }


def validate_dump(
    records,
    *,
    condition,
    expected_indices,
    run_uuid,
    matrix,
    task_tag,
    model_key,
    dataset,
    model_family,
    backend,
    prediction_indices,
):
    policy_records = [record for record in records if record.get("stage") in (None, "replay_policy")]
    final_records = [record for record in records if record.get("stage") == "final_model_input"]
    unknown = [record.get("stage") for record in records if record not in policy_records and record not in final_records]
    global_errors = []
    expected_count = len(expected_indices)
    if unknown:
        global_errors.append(f"unknown raw stages: {unknown}")
    if len(policy_records) != expected_count:
        global_errors.append(f"replay-policy records {len(policy_records)} != expected {expected_count}")
    if len(final_records) != expected_count:
        global_errors.append(f"final-input records {len(final_records)} != expected {expected_count}")
    if prediction_indices != expected_indices:
        global_errors.append(
            f"prediction indices {prediction_indices} != allowlist indices {expected_indices}"
        )

    policy_checks = []
    policy_batches = []
    for record in policy_records:
        check = validate_record(record, condition)
        expected_top_level = {
            "run_uuid": run_uuid,
            "matrix": matrix,
            "task_tag": task_tag,
            "model_key": model_key,
            "dataset": dataset,
            "condition": condition,
        }
        for field, expected in expected_top_level.items():
            if str(record.get(field)) != str(expected):
                check["errors"].append(f"{field} {record.get(field)!r} != {expected!r}")
        try:
            batch_indices = json.loads(record.get("sample_indices_json") or "null")
        except json.JSONDecodeError:
            batch_indices = None
        if not isinstance(batch_indices, list) or not batch_indices:
            check["errors"].append(f"invalid sample_indices_json: {batch_indices!r}")
        else:
            batch_indices = [str(index) for index in batch_indices]
            if not policy_batches or policy_batches[-1]["indices"] != batch_indices:
                policy_batches.append({"indices": batch_indices, "records": 0})
            policy_batches[-1]["records"] += 1
        policy_checks.append(check)

    reconstructed_indices = []
    for batch in policy_batches:
        reconstructed_indices.extend(batch["indices"])
        if batch["records"] != len(batch["indices"]):
            global_errors.append(
                f"policy batch {batch['indices']} emitted {batch['records']} records"
            )
    if reconstructed_indices != expected_indices:
        global_errors.append(
            f"policy batch indices {reconstructed_indices} != expected {expected_indices}"
        )

    final_by_index = {}
    for record in final_records:
        identity = record.get("task_identity")
        canonical_index = str(identity.get("canonical_index")) if isinstance(identity, dict) else "None"
        if canonical_index in final_by_index:
            global_errors.append(f"duplicate final-input canonical index: {canonical_index}")
        final_by_index[canonical_index] = record
    call_ids = [record.get("call_correlation_id") for record in final_records]
    if any(not call_id for call_id in call_ids):
        global_errors.append("one or more final-input call IDs are missing")
    if len(set(call_ids)) != len(call_ids):
        global_errors.append("final-input call IDs are not unique")

    parent_groups = {}
    for record in final_records:
        parent_id = record.get("parent_call_id")
        if not parent_id:
            global_errors.append("one or more final-input parent_call_id values are missing")
            continue
        parent_groups.setdefault(parent_id, []).append(record)
    observed_batch_indices = []
    for parent_id, group in parent_groups.items():
        positions = [record.get("batch_position") for record in group]
        if any(not isinstance(position, int) for position in positions):
            global_errors.append(f"parent {parent_id} has non-integer batch_position values")
            continue
        if len(set(positions)) != len(positions):
            global_errors.append(f"parent {parent_id} has duplicate batch_position values")
        ordered_group = sorted(group, key=lambda record: record["batch_position"])
        if sorted(positions) != list(range(len(group))):
            global_errors.append(
                f"parent {parent_id} batch positions {sorted(positions)} are not contiguous"
            )
        observed_batch_indices.append(
            [str(record.get("task_identity", {}).get("canonical_index")) for record in ordered_group]
        )
        for record in group:
            expected_call_id = f"{parent_id}:{record.get('batch_position')}"
            if record.get("call_correlation_id") != expected_call_id:
                global_errors.append(
                    f"call ID {record.get('call_correlation_id')!r} != {expected_call_id!r}"
                )
    expected_batch_indices = [batch["indices"] for batch in policy_batches]
    if sorted(observed_batch_indices) != sorted(expected_batch_indices):
        global_errors.append(
            f"final parent batches {observed_batch_indices} != policy batches {expected_batch_indices}"
        )
    ordered_final = [
        (index, final_by_index[index]) for index in expected_indices if index in final_by_index
    ]
    policy_question_by_index = {
        index: check["texts"][0]
        for index, check in zip(expected_indices, policy_checks)
        if check["texts"]
    }
    final_checks = []
    for canonical_index, record in ordered_final:
        policy_check = policy_checks[expected_indices.index(canonical_index)]
        final_checks.append(
            validate_final_record(
                record,
                condition=condition,
                expected_identity={
                    "run_uuid": run_uuid,
                    "matrix_tag": matrix,
                    "task_tag": task_tag,
                    "model_key": model_key,
                    "dataset": dataset,
                    "condition": condition,
                    "canonical_index": canonical_index,
                },
                model_family=model_family,
                backend=backend,
                policy_check=policy_check,
            )
        )

    final_indices = [str(check.get("canonical_index")) for check in final_checks]
    if final_indices != expected_indices:
        global_errors.append(f"final-input indices {final_indices} != expected {expected_indices}")
    fingerprints = [check["sample_fingerprint"] for check in policy_checks]
    if len(set(fingerprints)) != expected_count:
        global_errors.append(
            f"unique sample fingerprints {len(set(fingerprints))} != expected {expected_count}"
        )
    return policy_checks, final_checks, global_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_path", type=Path)
    parser.add_argument("--condition", choices=sorted(EXPECTED_TYPES), required=True)
    parser.add_argument("--expect-records", type=int, required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--run-uuid", required=True)
    parser.add_argument("--task-tag", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--allowlist-file", type=Path, required=True)
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--forbid-output-substring", action="append", default=[])
    parser.add_argument("--summary-path", type=Path)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.dump_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No records in {args.dump_path}")
    expected_indices = _load_allowlist(args.allowlist_file)
    if len(expected_indices) != args.expect_records:
        raise ValueError(
            f"Allowlist contains {len(expected_indices)} records; expected {args.expect_records}"
        )
    predictions = _load_predictions(args.prediction_file)
    prediction_indices = predictions["index"].tolist()
    policy_checks, final_checks, global_errors = validate_dump(
        records,
        condition=args.condition,
        expected_indices=expected_indices,
        run_uuid=args.run_uuid,
        matrix=args.matrix,
        task_tag=args.task_tag,
        model_key=args.model_key,
        dataset=args.dataset,
        model_family=args.model_family,
        backend=args.backend,
        prediction_indices=prediction_indices,
    )
    output_columns = [
        column
        for column in ("prediction", "detailed_prediction", "full_output", "description")
        if column in predictions
    ]
    for forbidden in args.forbid_output_substring:
        matches = []
        for row in predictions.to_dict("records"):
            combined = "\n".join(str(row.get(column, "")) for column in output_columns)
            if forbidden in combined:
                matches.append(str(row.get("index")))
        if matches:
            global_errors.append(
                f"forbidden output substring {forbidden!r} found for indices {matches}"
            )

    policy_error_count = sum(bool(check["errors"]) for check in policy_checks)
    final_error_count = sum(bool(check["errors"]) for check in final_checks)
    summary = {
        "dump_path": str(args.dump_path),
        "prediction_file": str(args.prediction_file),
        "prediction_sha256": _sha256(args.prediction_file),
        "raw_dump_sha256": _sha256(args.dump_path),
        "run_uuid": args.run_uuid,
        "condition": args.condition,
        "expected_indices": expected_indices,
        "raw_records": len(records),
        "policy_records": len(policy_checks),
        "final_input_records": len(final_checks),
        "valid_policy_records": len(policy_checks) - policy_error_count,
        "valid_final_input_records": len(final_checks) - final_error_count,
        "global_errors": global_errors,
        "policy_checks": policy_checks,
        "final_input_checks": final_checks,
    }
    encoded = json.dumps(summary, ensure_ascii=False, indent=2)
    print(encoded)
    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(encoded + "\n", encoding="utf-8")
    if policy_error_count or final_error_count or global_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
