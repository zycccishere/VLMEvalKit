#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-VL topology ablations on a sliced dataset while reusing one loaded model."
    )
    parser.add_argument("--model", default="Qwen2VLChatReplay")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="SEEDBench2_Plus")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--index-file", default="")
    parser.add_argument("--policy", default="identity", choices=["identity", "directly_answer"])
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--api-nproc", type=int, default=4)
    parser.add_argument("--eval-nproc", type=int, default=4)
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def ensure_blank_image(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.new("RGB", (224, 224), color=(255, 255, 255)).save(path)


def set_common_env(args: argparse.Namespace) -> None:
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["REPLAY_PROMPT_TEMPLATE_NAME"] = args.policy
    os.environ["REPLAY_TIMES"] = "1"
    os.environ["REPLAY_IMAGE_COPY_MODE"] = "reuse_path"
    os.environ["REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT"] = "1"
    os.environ["REPLAY_LIMIT_MM_PER_PROMPT"] = "2"
    os.environ["REPLAY_STAGE_DEBUG"] = "0"
    os.environ["REPLAY_PROMPT_AUDIT"] = "0"
    os.environ["REPLAY_DEBUG"] = "0"
    os.environ["VLLM_TP_SIZE"] = str(args.tp_size)
    os.environ["VLLM_MAX_MODEL_LEN"] = str(args.max_model_len)
    os.environ["VLLM_MAX_NUM_SEQS"] = str(args.max_num_seqs)
    os.environ["INFER_BATCH_SIZE"] = str(args.batch_size)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def configure_variant(model, blank_path: Path, mode: str, blank_positions: str) -> None:
    os.environ["REPLAY_MODE"] = mode
    if hasattr(model, "replay_cfg") and isinstance(model.replay_cfg, dict):
        model.replay_cfg["mode"] = mode
        model.replay_cfg["repeat_times"] = 1
        model.replay_cfg["image_copy_mode"] = "reuse_path"

    blank_positions = blank_positions.strip()
    if blank_positions:
        os.environ["REPLAY_BLANK_IMAGE_POSITIONS"] = blank_positions
        os.environ["REPLAY_BLANK_IMAGE_PATH"] = str(blank_path)
    else:
        os.environ.pop("REPLAY_BLANK_IMAGE_POSITIONS", None)
        os.environ.pop("REPLAY_BLANK_IMAGE_PATH", None)


def cleanup_outputs(pred_root: Path, model_name: str, dataset_name: str) -> None:
    for path in pred_root.glob(f"{model_name}_{dataset_name}*"):
        if path.is_file():
            path.unlink()


def variant_work_dir(base: Path, label: str) -> Path:
    return base / label


def load_indices(path: Path) -> list[int | str]:
    indices: list[int | str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            indices.append(int(raw))
        except Exception:
            indices.append(raw)
    return indices


def slice_dataset(dataset, args: argparse.Namespace):
    if args.index_file.strip():
        index_file = Path(args.index_file).expanduser().resolve()
        requested = load_indices(index_file)
        if args.limit > 0:
            requested = requested[: args.limit]
        order = {idx: i for i, idx in enumerate(requested)}
        sliced = dataset.data[dataset.data["index"].isin(requested)].copy()
        sliced["__order__"] = sliced["index"].map(order)
        sliced = sliced.sort_values("__order__").drop(columns="__order__").reset_index(drop=True)
        dataset.data = sliced
        return
    dataset.data = dataset.data.iloc[: args.limit].copy().reset_index(drop=True)


def build_sliced_dataset(args: argparse.Namespace):
    from vlmeval.dataset import build_dataset

    dataset = build_dataset(args.dataset)
    if dataset is None:
        raise RuntimeError(f"Failed to build dataset: {args.dataset}")
    slice_dataset(dataset, args)
    return dataset


def main() -> int:
    args = build_parser().parse_args()
    set_common_env(args)
    work_dir = Path(args.work_dir).expanduser().resolve()
    blank_path = work_dir / "_assets" / "blank_224.png"
    ensure_blank_image(blank_path)

    from vlmeval.config_runtime import supported_VLM
    from vlmeval.inference import infer_data_job

    dataset_for_model_init = build_sliced_dataset(args)

    # Instantiate once, then mutate replay settings between variants.
    os.environ["REPLAY_MODE"] = "image_text_image"
    model = supported_VLM[args.model]()

    variants = [
        ("image_text", "image_text", ""),
        ("image_text_image", "image_text_image", ""),
        ("image_text_image_blank_first", "image_text_image", "1"),
        ("image_text_image_blank_second", "image_text_image", "2"),
    ]

    summaries = []
    for label, mode, blank_positions in variants:
        dataset = build_sliced_dataset(args)
        configure_variant(model, blank_path, mode=mode, blank_positions=blank_positions)
        run_root = variant_work_dir(work_dir, label)
        pred_root = run_root / args.model
        pred_root.mkdir(parents=True, exist_ok=True)
        if args.force_clean and run_root.exists():
            shutil.rmtree(run_root)
            pred_root.mkdir(parents=True, exist_ok=True)
        elif args.force_clean:
            pred_root.mkdir(parents=True, exist_ok=True)

        infer_data_job(
            model=model,
            work_dir=str(pred_root),
            model_name=args.model,
            dataset=dataset,
            verbose=args.verbose,
            api_nproc=args.api_nproc,
            batch_size=args.batch_size,
        )

        result_file = pred_root / f"{args.model}_{dataset.dataset_name}.xlsx"
        acc = dataset.evaluate(
            str(result_file),
            model="exact_matching",
            nproc=args.eval_nproc,
        )
        summaries.append(
            {
                "label": label,
                "mode": mode,
                "blank_image_positions": blank_positions,
                "result_file": str(result_file),
                "acc_rows": acc.to_dict(orient="records"),
                "num_samples": len(dataset.data),
            }
        )
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)

    final_summary = {
        "model": args.model,
        "model_path": args.model_path,
        "dataset": dataset_for_model_init.dataset_name,
        "limit": int(args.limit),
        "policy": args.policy,
        "work_dir": str(work_dir),
        "variants": summaries,
    }
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
