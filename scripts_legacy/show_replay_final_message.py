#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
from typing import Any

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_DATALIST = (
    "SEEDBench2_Plus MathVista_MINI MMStar AI2D_TEST MMVet OCRBench "
    "MMMU_DEV_VAL MathVision DynaMath ObjHal MMHal"
)


def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REPLAY_POLICY_PATH = os.path.join(REPO_ROOT, "vlmeval", "vlm", "replay_policy.py")
PROMPT_TEMPLATE_PATH = os.path.join(REPO_ROOT, "vlmeval", "vlm", "qwen2_vl", "replay_prompt_template.py")

replay_policy = _load_module_from_path("replay_policy_runtime", REPLAY_POLICY_PATH)
prompt_templ = _load_module_from_path("replay_prompt_template_runtime", PROMPT_TEMPLATE_PATH)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


def ensure_image_url(image: str) -> str:
    prefixes = ("http://", "https://", "file://", "data:image;")
    if image.startswith(prefixes):
        return image
    if os.path.exists(image):
        return "file://" + image
    return image


def ensure_video_url(video: str) -> str:
    prefixes = ("http://", "https://", "file://", "data:video;")
    if video.startswith(prefixes):
        return video
    if os.path.exists(video):
        return "file://" + video
    return video


def prepare_content_vllm_like_qwen2_replay(
    inputs: list[dict[str, Any]],
    dataset: str,
    limit_mm_per_prompt: int,
    prompt_template_cfg: dict[str, str],
    replay_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    # Mirror the replay pipeline:
    # dataset.build_prompt(...) -> _prepare_content_vllm normalization -> template -> replay
    content: list[dict[str, Any]] = []
    cur_image_count = 0

    for item in inputs:
        typ = item.get("type")
        val = item.get("value")
        if typ == "image":
            out = {"type": "image", "image": ensure_image_url(str(val))}
            if dataset == "OCRBench":
                out["min_pixels"] = 10 * 10 * 28 * 28
            if cur_image_count < limit_mm_per_prompt:
                content.append(out)
                cur_image_count += 1
        elif typ == "video":
            out = {"type": "video", "video": ensure_video_url(str(val))}
            content.append(out)
        elif typ == "text":
            content.append({"type": "text", "text": str(val)})
        else:
            # Keep unknown types visible for debugging.
            content.append({"type": str(typ), "value": to_jsonable(val)})

    use_last_replay_text = bool(replay_cfg.get("template_on_last_replay_text", False))
    replay_mode = replay_policy.canonicalize_replay_mode(replay_cfg["mode"])
    if use_last_replay_text and not replay_policy.is_noop_replay_mode(replay_mode):
        replayed = replay_policy.apply_replay(
            content,
            mode=replay_mode,
            repeat_times=int(replay_cfg["repeat_times"]),
            image_copy_mode=str(replay_cfg["image_copy_mode"]),
        )
        final_content = prompt_templ.apply_prompt_template_to_content(replayed, prompt_template_cfg)
        debug_before = replayed
        debug_after = final_content
    else:
        templated = prompt_templ.apply_prompt_template_to_content(content, prompt_template_cfg)
        final_content = replay_policy.apply_replay(
            templated,
            mode=replay_mode,
            repeat_times=int(replay_cfg["repeat_times"]),
            image_copy_mode=str(replay_cfg["image_copy_mode"]),
        )
        debug_before = templated
        debug_after = final_content
    replay_policy.maybe_debug_print_replay(
        enabled=bool(replay_cfg["debug"]),
        mode=str(replay_cfg["mode"]),
        before=debug_before,
        after=debug_after,
        tag="show_replay_final_message",
    )
    return final_content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print final built inputs (after dataset prompt build + template + replay) without model forward."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Datasets to inspect. Default: standard run datalist.",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=1,
        help="How many rows to print per dataset.",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Starting row offset in each dataset.",
    )
    parser.add_argument("--replay-mode", type=str, default=os.environ.get("REPLAY_MODE", "image_text"))
    parser.add_argument("--replay-times", type=int, default=int(os.environ.get("REPLAY_TIMES", "1")))
    parser.add_argument(
        "--image-copy-mode",
        type=str,
        default=os.environ.get("REPLAY_IMAGE_COPY_MODE", "reuse_path"),
    )
    parser.add_argument(
        "--limit-mm-per-prompt",
        type=int,
        default=int(os.environ.get("REPLAY_LIMIT_MM_PER_PROMPT", "8")),
    )
    parser.add_argument(
        "--template-name",
        type=str,
        default=os.environ.get("REPLAY_PROMPT_TEMPLATE_NAME", "directly_answer"),
    )
    parser.add_argument("--template-file", type=str, default=os.environ.get("REPLAY_PROMPT_TEMPLATE_FILE", ""))
    parser.add_argument("--template-inline", type=str, default=os.environ.get("REPLAY_PROMPT_TEMPLATE", ""))
    parser.add_argument("--replay-debug", action="store_true", help="Enable replay debug print.")
    parser.add_argument(
        "--template-on-last-replay-text",
        type=int,
        default=int(os.environ.get("REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT", "0")),
        help="If 1 and replay is enabled, apply template only on the last replayed text.",
    )
    parser.add_argument(
        "--only-final-message",
        action="store_true",
        help="Only print final_message; omit dataset prompt struct and metadata.",
    )
    parser.add_argument(
        "--manual-image-path",
        type=str,
        default="",
        help="Optional manual mode. Requires both --manual-image-path and --manual-question.",
    )
    parser.add_argument(
        "--manual-question",
        type=str,
        default="",
        help="Optional manual mode question text.",
    )
    return parser.parse_args()


def resolve_prompt_template_cfg(args: argparse.Namespace) -> dict[str, str]:
    old_name = os.environ.get("REPLAY_PROMPT_TEMPLATE_NAME")
    old_file = os.environ.get("REPLAY_PROMPT_TEMPLATE_FILE")
    old_inline = os.environ.get("REPLAY_PROMPT_TEMPLATE")
    try:
        os.environ["REPLAY_PROMPT_TEMPLATE_NAME"] = args.template_name
        if args.template_file:
            os.environ["REPLAY_PROMPT_TEMPLATE_FILE"] = args.template_file
        else:
            os.environ.pop("REPLAY_PROMPT_TEMPLATE_FILE", None)
        if args.template_inline:
            os.environ["REPLAY_PROMPT_TEMPLATE"] = args.template_inline
        else:
            os.environ.pop("REPLAY_PROMPT_TEMPLATE", None)
        cfg = prompt_templ.read_prompt_template_config_from_env()
    finally:
        if old_name is None:
            os.environ.pop("REPLAY_PROMPT_TEMPLATE_NAME", None)
        else:
            os.environ["REPLAY_PROMPT_TEMPLATE_NAME"] = old_name
        if old_file is None:
            os.environ.pop("REPLAY_PROMPT_TEMPLATE_FILE", None)
        else:
            os.environ["REPLAY_PROMPT_TEMPLATE_FILE"] = old_file
        if old_inline is None:
            os.environ.pop("REPLAY_PROMPT_TEMPLATE", None)
        else:
            os.environ["REPLAY_PROMPT_TEMPLATE"] = old_inline
    return cfg


def build_manual_output(
    args: argparse.Namespace,
    prompt_template_cfg: dict[str, str],
    replay_cfg: dict[str, Any],
) -> dict[str, Any]:
    image_path = args.manual_image_path
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)

    inputs = [
        {"type": "image", "value": image_path},
        {"type": "text", "value": args.manual_question},
    ]
    final_content = prepare_content_vllm_like_qwen2_replay(
        inputs=inputs,
        dataset="manual",
        limit_mm_per_prompt=args.limit_mm_per_prompt,
        prompt_template_cfg=prompt_template_cfg,
        replay_cfg=replay_cfg,
    )
    final_message = [{"role": "user", "content": final_content}]
    return {
        "mode": "manual",
        "input_struct": inputs,
        "final_message": final_message,
    }


def build_dataset_output(
    args: argparse.Namespace,
    prompt_template_cfg: dict[str, str],
    replay_cfg: dict[str, Any],
) -> dict[str, Any]:
    from vlmeval.dataset import build_dataset

    datasets = args.datasets if args.datasets else DEFAULT_DATALIST.split()
    all_results = []

    for dataset_name in datasets:
        entry: dict[str, Any] = {"dataset": dataset_name}
        try:
            dataset = build_dataset(dataset_name)
            if dataset is None:
                raise RuntimeError("build_dataset returned None")
            data = getattr(dataset, "data", None)
            if data is None:
                raise RuntimeError("dataset has no .data")
            total = len(data)
            entry["total_rows"] = total

            sample_count = max(args.samples_per_dataset, 0)
            start = max(args.start_offset, 0)
            end = min(total, start + sample_count)
            row_positions = list(range(start, end))
            # For single-sample debug, prefer MCQ-like rows if available so options are visible.
            if sample_count == 1 and "answer_type" in getattr(data, "columns", []):
                try:
                    mcq_positions = [i for i, v in enumerate(data["answer_type"].tolist()) if str(v).lower() == "multiple choice"]
                    if mcq_positions:
                        row_positions = [mcq_positions[0]]
                except Exception:
                    pass

            samples = []
            for ridx in row_positions:
                row = data.iloc[ridx]
                raw_struct = dataset.build_prompt(row)
                final_content = prepare_content_vllm_like_qwen2_replay(
                    inputs=raw_struct,
                    dataset=dataset_name,
                    limit_mm_per_prompt=args.limit_mm_per_prompt,
                    prompt_template_cfg=prompt_template_cfg,
                    replay_cfg=replay_cfg,
                )
                sample = {
                    "row_offset": ridx,
                    "index": to_jsonable(row.get("index", ridx)),
                    "dataset_struct": to_jsonable(raw_struct),
                    "final_message": to_jsonable([{"role": "user", "content": final_content}]),
                }
                samples.append(sample)
            entry["samples"] = samples
        except Exception as err:
            entry["error"] = str(err)

        all_results.append(entry)

    return {
        "mode": "dataset",
        "datasets": datasets,
        "results": all_results,
    }


def main() -> None:
    args = parse_args()

    prompt_template_cfg = resolve_prompt_template_cfg(args)
    replay_cfg = {
        "mode": args.replay_mode,
        "repeat_times": args.replay_times,
        "debug": args.replay_debug,
        "image_copy_mode": args.image_copy_mode,
        "template_on_last_replay_text": bool(args.template_on_last_replay_text),
    }

    old_tpl = os.environ.get("REPLAY_PROMPT_TEMPLATE_NAME")
    try:
        # Keep dataset.build_prompt behavior aligned with the template under test.
        os.environ["REPLAY_PROMPT_TEMPLATE_NAME"] = str(prompt_template_cfg.get("name", ""))
        manual_mode = bool(args.manual_image_path and args.manual_question)
        if manual_mode:
            output = build_manual_output(args, prompt_template_cfg, replay_cfg)
        else:
            output = build_dataset_output(args, prompt_template_cfg, replay_cfg)
    finally:
        if old_tpl is None:
            os.environ.pop("REPLAY_PROMPT_TEMPLATE_NAME", None)
        else:
            os.environ["REPLAY_PROMPT_TEMPLATE_NAME"] = old_tpl

    if args.only_final_message:
        if manual_mode:
            print(json.dumps(output["final_message"], ensure_ascii=False, indent=2))
            return
        compact = []
        for ds in output.get("results", []):
            item = {"dataset": ds.get("dataset")}
            if "error" in ds:
                item["error"] = ds["error"]
            else:
                item["final_message"] = [s.get("final_message", []) for s in ds.get("samples", [])]
            compact.append(item)
        print(json.dumps(compact, ensure_ascii=False, indent=2))
        return

    payload = {
        "replay_cfg": replay_cfg,
        "prompt_template_cfg": prompt_template_cfg,
        "output": output,
    }
    print(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
