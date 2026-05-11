#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small replay subset for Qwen2.5-VL with exact-matching evaluation."
    )
    parser.add_argument("--model", default="Qwen2VLChatReplay")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="SEEDBench2_Plus")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--index-file", default="")
    parser.add_argument(
        "--policy",
        default="identity",
        choices=["identity", "directly_answer", "blank_like_problem"],
    )
    parser.add_argument("--mode", default="image_text_image")
    parser.add_argument("--blank-image-positions", default="")
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


def set_runtime_env(args: argparse.Namespace, blank_path: Path | None) -> None:
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["REPLAY_MODE"] = args.mode
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

    blank_positions = args.blank_image_positions.strip()
    if blank_positions and blank_path is not None:
        os.environ["REPLAY_BLANK_IMAGE_POSITIONS"] = blank_positions
        os.environ["REPLAY_BLANK_IMAGE_PATH"] = str(blank_path)
    else:
        os.environ.pop("REPLAY_BLANK_IMAGE_POSITIONS", None)
        os.environ.pop("REPLAY_BLANK_IMAGE_PATH", None)


def cleanup_outputs(pred_root: Path, model_name: str, dataset_name: str) -> None:
    for path in pred_root.glob(f"{model_name}_{dataset_name}*"):
        if path.is_file():
            path.unlink()


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


def slice_dataset(dataset, args: argparse.Namespace) -> None:
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


def main() -> int:
    args = build_parser().parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    blank_path = None
    if args.blank_image_positions.strip():
        blank_path = work_dir / "_assets" / "blank_224.png"
        ensure_blank_image(blank_path)
    set_runtime_env(args, blank_path)

    from vlmeval.dataset import build_dataset
    from vlmeval.inference import infer_data_job

    dataset = build_dataset(args.dataset)
    if dataset is None:
        raise RuntimeError(f"Failed to build dataset: {args.dataset}")
    slice_dataset(dataset, args)
    pred_root = work_dir / args.model
    pred_root.mkdir(parents=True, exist_ok=True)

    if args.force_clean:
        cleanup_outputs(pred_root, args.model, dataset.dataset_name)

    infer_data_job(
        model=args.model,
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
    acc_file = result_file.with_name(f"{result_file.stem}_acc.csv")
    summary = {
        "model": args.model,
        "model_path": args.model_path,
        "dataset": dataset.dataset_name,
        "limit": int(args.limit),
        "index_file": args.index_file,
        "policy": args.policy,
        "mode": args.mode,
        "blank_image_positions": args.blank_image_positions,
        "result_file": str(result_file),
        "acc_file": str(acc_file),
        "acc_rows": acc.to_dict(orient="records"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
