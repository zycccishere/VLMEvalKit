import torch
import torch.distributed as dist
from vlmeval.config_runtime import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
import sys
import os
import json
from contextlib import contextmanager
from typing import Any

FAIL_MSG = 'Failed to obtain answer via API.'


@contextmanager
def _replay_sample_indices(indices):
    """Expose the exact batch/index association to opt-in runtime dumps."""
    key = "REPLAY_SAMPLE_INDICES_JSON"
    previous = os.environ.get(key)
    enabled = bool(os.environ.get("REPLAY_RAW_DUMP_PATH", "").strip())
    if enabled:
        os.environ[key] = json.dumps([str(index) for index in indices], ensure_ascii=False)
    try:
        yield
    finally:
        if enabled:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _is_overlong_prompt_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "longer than the maximum model length" in msg
        or ("maximum model length" in msg and "prompt" in msg)
        or "max_model_len" in msg
    )


def _record_overlong_skip(work_dir: str, model_name: str, dataset_name: str, idx, err: Exception) -> None:
    try:
        log_dir = osp.join(work_dir, "_logs", "overlong_skipped")
        os.makedirs(log_dir, exist_ok=True)
        log_file = osp.join(log_dir, f"{model_name}_{dataset_name}.jsonl")
        rec = {
            "index": str(idx),
            "reason": "overlong_prompt",
            "error_type": type(err).__name__,
            "error": str(err),
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _make_overlong_skip_result(err: Exception) -> dict:
    return {
        "prediction": "",
        "description": f"[SKIPPED_OVERLONG_PROMPT] {type(err).__name__}: {err}",
        "detailed_prediction": "",
        "full_output": "",
    }


def _make_failed_result(err: Exception) -> dict:
    return {
        "prediction": "",
        "description": f"[FAILED_INFER] {type(err).__name__}: {err}",
        "detailed_prediction": "",
        "full_output": "",
    }


def _is_failed_result(result) -> bool:
    if result is None:
        return True
    if isinstance(result, str):
        return FAIL_MSG in result or result.startswith("[FAILED_INFER]")
    if isinstance(result, dict):
        desc = str(result.get("description", ""))
        return desc.startswith("[FAILED_INFER]") or FAIL_MSG in desc
    return False


def _resume_failed_enabled() -> bool:
    return os.environ.get("INFER_RESUME_FAILED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _load_dataset_index_allowlist() -> set[Any] | None:
    raw = os.environ.get("DATASET_INDEX_ALLOWLIST_FILE", "").strip()
    if not raw:
        return None
    if not osp.exists(raw):
        raise FileNotFoundError(f"DATASET_INDEX_ALLOWLIST_FILE not found: {raw}")
    if raw.lower().endswith(".json"):
        payload = json.loads(open(raw, "r", encoding="utf-8").read())
        if isinstance(payload, dict):
            values = payload.get("indices", [])
        else:
            values = payload
    else:
        values = [line.strip() for line in open(raw, "r", encoding="utf-8").read().splitlines() if line.strip()]
    if not isinstance(values, list):
        raise ValueError(f"Invalid allowlist payload in {raw}")
    out = set()
    for value in values:
        normalized = _normalize_resume_index(value)
        if normalized is not None:
            out.add(normalized)
    return out


def _first_image_value_from_struct(struct) -> str:
    if not isinstance(struct, list):
        return ""
    for item in struct:
        if isinstance(item, dict) and item.get("type") == "image":
            return _cell_to_text(item.get("value", ""))
    return ""


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _build_common_prompt_struct(dataset, dataset_name: str, row) -> list[dict[str, Any]] | None:
    if not _truthy_env("REPLAY_FORCE_COMMON_PROMPT"):
        return None
    from vlmeval.dataset import DATASET_TYPE
    import pandas as pd
    import string

    dataset_type = DATASET_TYPE(dataset_name, default=getattr(dataset, "TYPE", None))
    if dataset_type not in {"MCQ", "VQA", "Y/N"}:
        return None

    if getattr(dataset, "meta_only", False):
        tgt_path = toliststr(row["image_path"])
    else:
        tgt_path = dataset.dump_image(row)

    question = str(row["question"])
    prompt = ""
    hint = row["hint"] if ("hint" in row and not pd.isna(row["hint"])) else None
    if hint is not None:
        prompt += f"Hint: {hint}\n"

    if dataset_type == "MCQ":
        options = {
            cand: row[cand]
            for cand in string.ascii_uppercase
            if cand in row and not pd.isna(row[cand])
        }
        prompt += f"Question: {question}\n"
        if len(options):
            prompt += "Options:\n"
            for key, item in options.items():
                prompt += f"{key}. {item}\n"
            prompt += "Please select the correct answer from the options above."
    elif dataset_type == "Y/N":
        prompt += question
        prompt += " Please answer yes or no."
    else:
        prompt += question
        prompt += "\nPlease try to answer the question with short words or phrases if possible."

    msgs = []
    if isinstance(tgt_path, list):
        msgs.extend([dict(type="image", value=p) for p in tgt_path])
    else:
        msgs = [dict(type="image", value=tgt_path)]
    msgs.append(dict(type="text", value=prompt.rstrip()))
    return msgs


def _maybe_build_prompt_struct(model, dataset, dataset_name: str, row) -> list[dict[str, Any]]:
    # Some benchmark protocols require a dataset-owned output grammar (for
    # example, an integer-only answer or a specific bounding-box coordinate
    # system). This opt-in leaves every existing dataset/model prompt path
    # unchanged while preventing model mixins from silently rewriting those
    # protocol-critical prompts.
    if getattr(dataset, "FORCE_DATASET_PROMPT", False):
        return dataset.build_prompt(row)
    common = _build_common_prompt_struct(dataset, dataset_name, row)
    if common is not None:
        return common
    if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
        struct = model.build_prompt(row, dataset=dataset_name)
    else:
        struct = dataset.build_prompt(row)
    return struct


def _attach_replay_meta(struct: list[dict[str, Any]], replay_meta: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(struct, list) or not struct:
        return struct
    out = []
    for idx, item in enumerate(struct):
        if not isinstance(item, dict):
            out.append(item)
            continue
        copied = dict(item)
        if idx == 0:
            copied["replay_meta"] = replay_meta
        out.append(copied)
    return out


def _needs_random_same_dataset_meta() -> bool:
    return os.environ.get("REPLAY_IMAGE_TRANSFORM", "").strip().lower() == "random_same_dataset"


def _build_random_same_dataset_meta(
    *,
    model,
    dataset,
    dataset_name: str,
    current_row,
    current_struct: list[dict[str, Any]],
    index_to_position: dict[Any, int],
    prompt_cache: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    current_index = _normalize_resume_index(current_row["index"])
    current_pos = index_to_position.get(current_index)
    source_ref = _first_image_value_from_struct(current_struct)
    if current_pos is None:
        return {}
    full_data = dataset.data
    total = len(full_data)
    if total <= 1:
        return {}
    donor_offset = max(1, int(os.environ.get("REPLAY_RANDOM_SAME_DATASET_OFFSET", "1")))
    for delta in range(donor_offset, total + donor_offset):
        donor_pos = (current_pos + delta) % total
        donor_row = full_data.iloc[donor_pos]
        donor_index = _normalize_resume_index(donor_row["index"])
        if donor_index == current_index:
            continue
        if donor_pos not in prompt_cache:
            prompt_cache[donor_pos] = _maybe_build_prompt_struct(model, dataset, dataset_name, donor_row)
        donor_ref = _first_image_value_from_struct(prompt_cache[donor_pos])
        if donor_ref and donor_ref != source_ref:
            return {
                "random_same_dataset_image_ref": donor_ref,
                "random_same_dataset_donor_index": donor_index,
            }
    return {}


def _build_replay_meta(
    *,
    model,
    dataset,
    dataset_name: str,
    current_row,
    current_struct: list[dict[str, Any]],
    index_to_position: dict[Any, int],
    prompt_cache: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    current_index = _normalize_resume_index(current_row["index"])
    replay_meta = {
        "sample_index": current_index,
        "dataset_name": dataset_name,
        "source_first_image_ref": _first_image_value_from_struct(current_struct),
    }
    if _needs_random_same_dataset_meta():
        replay_meta.update(
            _build_random_same_dataset_meta(
                model=model,
                dataset=dataset,
                dataset_name=dataset_name,
                current_row=current_row,
                current_struct=current_struct,
                index_to_position=index_to_position,
                prompt_cache=prompt_cache,
            )
        )
    return replay_meta


def _normalize_resume_index(raw_idx):
    if raw_idx is None:
        return None
    try:
        if pd.isna(raw_idx):
            return None
    except Exception:
        pass
    if hasattr(raw_idx, "item"):
        try:
            raw_idx = raw_idx.item()
        except Exception:
            pass
    if isinstance(raw_idx, str):
        stripped = raw_idx.strip()
        if stripped == "":
            return stripped
        try:
            return int(stripped)
        except Exception:
            try:
                maybe_float = float(stripped)
            except Exception:
                return stripped
            if maybe_float.is_integer():
                return int(maybe_float)
            return stripped
    if isinstance(raw_idx, float) and raw_idx.is_integer():
        return int(raw_idx)
    return raw_idx


def _cell_to_text(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if value is None:
        return ""
    return str(value)


def _load_existing_results_for_resume(result_file, dataset) -> dict:
    if not osp.exists(result_file):
        return {}
    try:
        table = load(result_file)
    except Exception as err:
        print(f"[RESUME] failed to read existing result file {result_file}: {err}", flush=True)
        return {}
    if not isinstance(table, pd.DataFrame) or "index" not in table.columns:
        return {}

    dataset_index_map = {_normalize_resume_index(idx): idx for idx in dataset.data["index"]}
    resumed = {}
    for _, row in table.iterrows():
        raw_idx = _normalize_resume_index(row.get("index"))
        if raw_idx not in dataset_index_map:
            continue
        idx = dataset_index_map[raw_idx]
        record = {
            "prediction": _cell_to_text(row.get("prediction", "")),
            "description": _cell_to_text(row.get("description", "")),
            "detailed_prediction": _cell_to_text(row.get("detailed_prediction", "")),
            "full_output": _cell_to_text(row.get("full_output", row.get("detailed_prediction", ""))),
        }
        if _is_failed_result(record):
            continue
        if not any(record.values()):
            continue
        resumed[idx] = record
    return resumed


def _load_existing_tmp_results_for_resume(tmp_pkl_file) -> dict:
    if not osp.exists(tmp_pkl_file):
        return {}
    try:
        data = load(tmp_pkl_file)
    except Exception as err:
        print(f"[RESUME] failed to read existing tmp file {tmp_pkl_file}: {err}", flush=True)
        return {}
    if not isinstance(data, dict):
        return {}
    if _resume_failed_enabled():
        data = {k: v for k, v in data.items() if not _is_failed_result(v)}
    return data


def _filter_data_by_allowlist(data: pd.DataFrame, allowlist: set[Any] | None) -> pd.DataFrame:
    if allowlist is None:
        return data
    normalized = data["index"].map(_normalize_resume_index)
    return data[normalized.isin(allowlist)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False, batch_size=4):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    allowlist = _load_dataset_index_allowlist()
    data = _filter_data_by_allowlist(data, allowlist)
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])
    index_to_position = {_normalize_resume_index(dataset.data.iloc[pos]['index']): pos for pos in range(len(dataset.data))}
    prompt_cache: dict[int, list[dict[str, Any]]] = {}

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        struct = _maybe_build_prompt_struct(model, dataset, dataset_name, item)
        replay_meta = _build_replay_meta(
            model=model,
            dataset=dataset,
            dataset_name=dataset_name,
            current_row=item,
            current_struct=struct,
            index_to_position=index_to_position,
            prompt_cache=prompt_cache,
        )
        structs.append(_attach_replay_meta(struct, replay_meta))

    # structs = [dataset.build_prompt(data.iloc[i]) for i in range(lt)]

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}
        elif _resume_failed_enabled():
            res = {k: v for k, v in res.items() if not _is_failed_result(v)}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, batch_size=4):
    print(f"processing batch_size {batch_size}", flush=True, file=sys.stderr)
    dataset_name = dataset.dataset_name
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if _resume_failed_enabled():
            res = {k: v for k, v in res.items() if not _is_failed_result(v)}

    data = dataset.data
    allowlist = _load_dataset_index_allowlist()
    data = _filter_data_by_allowlist(data, allowlist)

    lt = len(data)
    data_indices = list(data['index'])

    all_finished = all(idx in res for idx in data_indices)
    if all_finished:
        print("All tasks are already finished.")
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name

    stage_debug = os.environ.get("REPLAY_STAGE_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    stage_debug_max = int(os.environ.get("REPLAY_STAGE_DEBUG_SAMPLES", "3"))
    stage_debug_printed = 0
    skip_overlong = os.environ.get("SKIP_OVERLONG_SAMPLE", "1").strip().lower() in {"1", "true", "yes", "on"}

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model_name
    else:
        if hasattr(model, 'set_dump_image'):
            model.set_dump_image(dataset.dump_image)
    index_to_position = {_normalize_resume_index(dataset.data.iloc[pos]['index']): pos for pos in range(len(dataset.data))}
    prompt_cache: dict[int, list[dict[str, Any]]] = {}

    for i in tqdm(range(0, lt, batch_size)):
        # Get the mini-batch of data
        mini_batch_data = data.iloc[i:i + batch_size]

        # Prepare prompts and indices for the mini-batch
        prompts = []
        indices = []
        for j in range(len(mini_batch_data)):
            idx = mini_batch_data.iloc[j]['index']

            row = mini_batch_data.iloc[j]
            struct = _maybe_build_prompt_struct(model, dataset, dataset_name, row)
            replay_meta = _build_replay_meta(
                model=model,
                dataset=dataset,
                dataset_name=dataset_name,
                current_row=row,
                current_struct=struct,
                index_to_position=index_to_position,
                prompt_cache=prompt_cache,
            )
            struct = _attach_replay_meta(struct, replay_meta)

            if stage_debug and stage_debug_printed < stage_debug_max:
                try:
                    print(
                        "[STAGE_DEBUG] " + json.dumps(
                            {
                                "stage": "dataset_prompt_built",
                                "model": model_name,
                                "dataset": dataset_name,
                                "index": str(idx),
                                "struct": struct,
                                "replay_meta": replay_meta,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                except Exception:
                    print(
                        f"[STAGE_DEBUG] stage=dataset_prompt_built model={model_name} dataset={dataset_name} index={idx}",
                        flush=True,
                    )
                stage_debug_printed += 1

            prompts.append(struct)
            indices.append(idx)

        if not prompts:
            continue


        # Call generate_batch for the mini-batch
        if batch_size > 1:
            try:
                print(f"Processing batch {i // batch_size + 1} with {len(prompts)} prompts.", flush=True)
                with _replay_sample_indices(indices):
                    responses = model.generate_batch(messages=prompts, dataset=dataset_name)
            except Exception as e:
                strict_batch = os.environ.get('VLMEVAL_STRICT_BATCH', '0').strip().lower() in {
                    '1', 'true', 'yes', 'on'
                }
                if strict_batch:
                    raise RuntimeError(
                        f"Batch generation failed for {dataset_name} at offset {i} "
                        f"with batch_size={batch_size}"
                    ) from e
                print(f"Error during batch generation: {e}. Falling back to single-item generation for this batch.")
                # Fallback to single-item generation if batch fails
                responses = []
                for prompt, prompt_index in zip(prompts, indices):
                    try:
                        with _replay_sample_indices([prompt_index]):
                            responses.append(model.generate(message=prompt, dataset=dataset_name))
                    except Exception as e_single:
                        if skip_overlong and _is_overlong_prompt_error(e_single):
                            print(f"[WARN][OVERLONG] single-item fallback skipped due to max length: {e_single}")
                            responses.append(_make_overlong_skip_result(e_single))
                        else:
                            print(f"Error during single-item fallback: {e_single}. Marking this item for resume.")
                            responses.append(_make_failed_result(e_single))
                    # responses.append(model.generate(message=prompt, dataset=dataset_name))

        else:
            try:
                with _replay_sample_indices(indices):
                    responses = [model.generate(message=prompts[0], dataset=dataset_name)]
            except Exception as e_single:
                if skip_overlong and _is_overlong_prompt_error(e_single):
                    print(f"[WARN][OVERLONG] skipped due to max length: {e_single}")
                    responses = [_make_overlong_skip_result(e_single)]
                else:
                    print(f"Error during single-item generation: {e_single}. Marking this item for resume.")
                    responses = [_make_failed_result(e_single)]

        torch.cuda.empty_cache()

        # Map responses back to indices
        for idx, response in zip(indices, responses):
            res[idx] = response
            if (
                skip_overlong
                and isinstance(response, dict)
                and str(response.get("description", "")).startswith("[SKIPPED_OVERLONG_PROMPT]")
            ):
                _record_overlong_skip(work_dir, str(model_name), dataset_name, idx, Exception(response["description"]))
            if verbose:
                print(f"Index: {idx}\nResponse: {response}", flush=True)

        # Save intermediate results after each batch
        dump(res, out_file)

    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False, batch_size=4):
    dataset_name = dataset.dataset_name
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')
    tmp_pkl_file = osp.join(work_dir, f'{model_name}_{dataset_name}.pkl')
    allowlist = _load_dataset_index_allowlist()
    data = _filter_data_by_allowlist(dataset.data, allowlist)

    resumed = {}
    if osp.exists(result_file):
        dataset_for_resume = type("DatasetView", (), {"data": data})()
        resumed = _load_existing_results_for_resume(result_file, dataset_for_resume)
        expected = len(data)
        if len(resumed) == expected:
            return model_name
    tmp_resumed = _load_existing_tmp_results_for_resume(tmp_pkl_file)
    merged_resumed = dict(tmp_resumed)
    if resumed:
        merged_resumed.update(resumed)
    if merged_resumed and merged_resumed != tmp_resumed:
        print(
            f"[RESUME] seed {len(merged_resumed)}/{len(data)} finished samples from existing artifacts: "
            f"table={osp.exists(result_file)} tmp={osp.exists(tmp_pkl_file)}",
            flush=True,
        )
        dump(merged_resumed, tmp_pkl_file)

    out_file = tmp_pkl_file

    model = infer_data(
        model, work_dir=work_dir, dataset=dataset, out_file=out_file, verbose=verbose, api_nproc=api_nproc, batch_size=batch_size)

    data_all = load(tmp_pkl_file)
    unfinished = []
    for x in data['index']:
        if x not in data_all:
            print(f"Warning: Index {x} not found in inference results, will be filled with empty.")
            unfinished.append(x)
            data_all[x] = {}
        elif _is_failed_result(data_all[x]):
            unfinished.append(x)

    if unfinished:
        preview = ', '.join(map(str, unfinished[:10]))
        print(
            f"[RESUME] inference incomplete for {model_name} x {dataset_name}: "
            f"{len(unfinished)} sample(s) still pending. Example indices: {preview}",
            flush=True,
        )
        dump(data_all, tmp_pkl_file)
        raise RuntimeError(
            f"Inference incomplete for {model_name} x {dataset_name}; rerun will resume from pending samples."
        )

    predictions = []
    descriptions = []
    detailed_predictions = []
    full_outputs = []
    for x in data['index']:
        result_dict = data_all.get(x, {}) # Use .get for safety
        if isinstance(result_dict, dict):
            predictions.append(str(result_dict.get('prediction', '')))
            descriptions.append(str(result_dict.get('description', '')))
            detailed_predictions.append(str(result_dict.get('detailed_prediction', '')))
            full_outputs.append(str(result_dict.get('full_output', result_dict.get('detailed_prediction', ''))))
        else: # Fallback for non-dict results (like FAIL_MSG from older versions)
            predictions.append(str(result_dict))
            descriptions.append('')
            detailed_predictions.append('')
            full_outputs.append('')

    data['prediction'] = predictions
    data['description'] = descriptions
    data['detailed_prediction'] = detailed_predictions
    data['full_output'] = full_outputs

    if 'image' in data and 'image' in data.columns:
        data.pop('image')

    dump(data, result_file)

    if osp.exists(tmp_pkl_file):
        os.remove(tmp_pkl_file)

    return model
