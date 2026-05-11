#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vlmeval.dataset import build_dataset

from qwen25vl_image2_probe import (
    PROMPT_TEMPLATE_DIRECTLY_ANSWER,
    PROMPT_TEMPLATE_IDENTITY,
    RopeAlignPatch,
    build_base_content,
    build_inputs,
    build_replayed_content,
    configure_blank_image_env,
    load_model_and_processor,
    resolve_blank_image_path_for_sample,
    resolve_input_device,
    set_seed,
    tensor_to_device,
)


CONDITION_TO_BLANK_POSITIONS = {
    "full": [],
    "no_image1": [1],
    "no_image2": [2],
    "no_both": [1, 2],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute output-level image1/image2 attribution for Qwen2.5-VL on a fixed decode trajectory."
    )
    parser.add_argument("--model-path", default="/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset", default="DynaMath")
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--mode", default="image_text_image")
    parser.add_argument(
        "--policy",
        default="identity",
        choices=[PROMPT_TEMPLATE_IDENTITY, PROMPT_TEMPLATE_DIRECTLY_ANSWER],
    )
    parser.add_argument("--template-on-last-replay-text", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--rope-align", action="store_true")
    parser.add_argument(
        "--blank-image-match-source-size",
        action="store_true",
        help="Create a white blank image with the same width/height as a source prompt image for each sample.",
    )
    parser.add_argument(
        "--blank-image-source-position",
        type=int,
        default=1,
        help="1-based image position in the base prompt used to determine blank image size.",
    )
    parser.add_argument("--blank-image-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    return parser


def make_condition_inputs(
    *,
    processor,
    dataset_name: str,
    base_content: list[dict[str, Any]],
    mode: str,
    policy: str,
    template_on_last_replay_text: bool,
    blank_positions: list[int],
    blank_image_path: str,
    blank_image_match_source_size: bool,
    blank_image_source_position: int,
    blank_cache_dir: Path,
    input_device: str,
) -> tuple[str, dict[str, torch.Tensor]]:
    resolved_blank = resolve_blank_image_path_for_sample(
        base_content=base_content,
        blank_positions=blank_positions,
        blank_image_path=blank_image_path,
        blank_image_match_source_size=blank_image_match_source_size,
        blank_image_source_position=blank_image_source_position,
        blank_cache_dir=blank_cache_dir,
    )
    configure_blank_image_env(blank_positions, resolved_blank)
    content = build_replayed_content(
        base_content,
        dataset_name,
        mode=mode,
        policy=policy,
        template_on_last_replay_text=template_on_last_replay_text,
    )
    _, prompt_text, model_inputs = build_inputs(processor, content)
    model_inputs = tensor_to_device(model_inputs, input_device)
    return prompt_text, model_inputs


def build_teacher_forced_inputs(
    model_inputs: dict[str, torch.Tensor],
    generated_ids: list[int],
) -> dict[str, torch.Tensor]:
    if not generated_ids:
        return model_inputs

    prompt_input_ids = model_inputs["input_ids"]
    prompt_attention_mask = model_inputs["attention_mask"]
    prefix_ids = generated_ids[:-1]
    if not prefix_ids:
        return model_inputs

    prefix_tensor = torch.tensor(prefix_ids, dtype=prompt_input_ids.dtype, device=prompt_input_ids.device).unsqueeze(0)
    prefix_mask = torch.ones((1, len(prefix_ids)), dtype=prompt_attention_mask.dtype, device=prompt_attention_mask.device)

    out = dict(model_inputs)
    out["input_ids"] = torch.cat([prompt_input_ids, prefix_tensor], dim=1)
    out["attention_mask"] = torch.cat([prompt_attention_mask, prefix_mask], dim=1)
    return out


def collect_condition_scores(
    *,
    model,
    model_inputs: dict[str, torch.Tensor],
    generated_ids: list[int],
) -> list[dict[str, Any]]:
    if not generated_ids:
        return []

    forced_inputs = build_teacher_forced_inputs(model_inputs, generated_ids)
    prompt_len = model_inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        outputs = model(
            **forced_inputs,
            use_cache=False,
            return_dict=True,
        )

    logits = outputs.logits[0, prompt_len - 1 : prompt_len - 1 + len(generated_ids), :].float().detach().cpu()
    log_probs = F.log_softmax(logits, dim=-1)
    target_ids = torch.tensor(generated_ids, dtype=torch.long)

    target_logits = logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    target_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    masked_logits = logits.clone()
    masked_logits.scatter_(1, target_ids.unsqueeze(1), float("-inf"))
    alt_logits, alt_ids = masked_logits.max(dim=1)
    margins = target_logits - alt_logits

    rows: list[dict[str, Any]] = []
    for step, token_id in enumerate(generated_ids):
        rows.append(
            {
                "step": step,
                "token_id": int(token_id),
                "target_logit": float(target_logits[step].item()),
                "target_logprob": float(target_log_probs[step].item()),
                "alt_token_id": int(alt_ids[step].item()),
                "alt_logit": float(alt_logits[step].item()),
                "target_margin": float(margins[step].item()),
            }
        )
    return rows


def shapley_rows_from_condition_scores(
    *,
    sample_index: int,
    token_texts: list[str],
    condition_scores: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    full_rows = condition_scores["full"]
    no_image1_rows = condition_scores["no_image1"]
    no_image2_rows = condition_scores["no_image2"]
    no_both_rows = condition_scores["no_both"]

    out: list[dict[str, Any]] = []
    for step in range(len(full_rows)):
        full = full_rows[step]
        no_image1 = no_image1_rows[step]
        no_image2 = no_image2_rows[step]
        no_both = no_both_rows[step]

        phi_image1_logprob = 0.5 * (
            (full["target_logprob"] - no_image2["target_logprob"])
            + (no_image1["target_logprob"] - no_both["target_logprob"])
        )
        phi_image2_logprob = 0.5 * (
            (full["target_logprob"] - no_image1["target_logprob"])
            + (no_image2["target_logprob"] - no_both["target_logprob"])
        )
        interaction_logprob = (
            full["target_logprob"]
            - no_image1["target_logprob"]
            - no_image2["target_logprob"]
            + no_both["target_logprob"]
        )

        phi_image1_margin = 0.5 * (
            (full["target_margin"] - no_image2["target_margin"])
            + (no_image1["target_margin"] - no_both["target_margin"])
        )
        phi_image2_margin = 0.5 * (
            (full["target_margin"] - no_image1["target_margin"])
            + (no_image2["target_margin"] - no_both["target_margin"])
        )
        interaction_margin = (
            full["target_margin"]
            - no_image1["target_margin"]
            - no_image2["target_margin"]
            + no_both["target_margin"]
        )

        out.append(
            {
                "sample_index": sample_index,
                "step": step,
                "token_id": int(full["token_id"]),
                "token_text": token_texts[step],
                "phi_image1_logprob": phi_image1_logprob,
                "phi_image2_logprob": phi_image2_logprob,
                "interaction_logprob": interaction_logprob,
                "phi_image1_margin": phi_image1_margin,
                "phi_image2_margin": phi_image2_margin,
                "interaction_margin": interaction_margin,
            }
        )
    return out


def summarize_sample_row(
    *,
    sample_index: int,
    row: pd.Series,
    prompt_text: str,
    generated_ids: list[int],
    generated_text: str,
) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "question_id": str(row.get("index", sample_index)),
        "subcategory": row.get("subcategory", None),
        "category": row.get("category", None),
        "question": row.get("question", None),
        "prompt_preview": prompt_text[:500],
        "generated_token_count": len(generated_ids),
        "generated_text": generated_text,
    }


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blank_cache_dir = output_dir / "_blank_cache"

    processor, model = load_model_and_processor(args.model_path, args.device)
    input_device = resolve_input_device(model, args.device)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    rope_patch = RopeAlignPatch(model, image_token_id, enabled=args.rope_align)
    dataset = build_dataset(args.dataset)

    sample_summary_rows: list[dict[str, Any]] = []
    condition_score_rows: list[dict[str, Any]] = []
    shapley_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    run_start = time.perf_counter()
    try:
        rope_patch.__enter__()
        for sample_index in args.indices:
            sample_start = time.perf_counter()
            row = dataset.data.iloc[sample_index]
            base_content = build_base_content(dataset, row)

            prompt_text, full_model_inputs = make_condition_inputs(
                processor=processor,
                dataset_name=args.dataset,
                base_content=base_content,
                mode=args.mode,
                policy=args.policy,
                template_on_last_replay_text=args.template_on_last_replay_text,
                blank_positions=[],
                blank_image_path=args.blank_image_path,
                blank_image_match_source_size=args.blank_image_match_source_size,
                blank_image_source_position=args.blank_image_source_position,
                blank_cache_dir=blank_cache_dir,
                input_device=input_device,
            )

            with torch.inference_mode():
                sequences = model.generate(
                    **full_model_inputs,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                )

            prompt_len = full_model_inputs["input_ids"].shape[-1]
            generated_ids = sequences[0, prompt_len:].detach().cpu().tolist()
            generated_text = processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            token_texts = [
                processor.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                for token_id in generated_ids
            ]
            del sequences
            torch.cuda.empty_cache()

            sample_summary_rows.append(
                summarize_sample_row(
                    sample_index=sample_index,
                    row=row,
                    prompt_text=prompt_text,
                    generated_ids=generated_ids,
                    generated_text=generated_text,
                )
            )

            condition_scores: dict[str, list[dict[str, Any]]] = {}
            for condition, blank_positions in CONDITION_TO_BLANK_POSITIONS.items():
                _, condition_inputs = make_condition_inputs(
                    processor=processor,
                    dataset_name=args.dataset,
                    base_content=base_content,
                    mode=args.mode,
                    policy=args.policy,
                    template_on_last_replay_text=args.template_on_last_replay_text,
                    blank_positions=blank_positions,
                    blank_image_path=args.blank_image_path,
                    blank_image_match_source_size=args.blank_image_match_source_size,
                    blank_image_source_position=args.blank_image_source_position,
                    blank_cache_dir=blank_cache_dir,
                    input_device=input_device,
                )
                rows = collect_condition_scores(
                    model=model,
                    model_inputs=condition_inputs,
                    generated_ids=generated_ids,
                )
                condition_scores[condition] = rows
                for row_item in rows:
                    condition_score_rows.append(
                        {
                            "sample_index": sample_index,
                            "condition": condition,
                            **row_item,
                            "token_text": token_texts[row_item["step"]],
                        }
                    )
                torch.cuda.empty_cache()

            shapley_rows.extend(
                shapley_rows_from_condition_scores(
                    sample_index=sample_index,
                    token_texts=token_texts,
                    condition_scores=condition_scores,
                )
            )

            sample_seconds = time.perf_counter() - sample_start
            timing_rows.append(
                {
                    "sample_index": sample_index,
                    "generated_token_count": len(generated_ids),
                    "sample_seconds": sample_seconds,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "sample_complete",
                        "sample_index": sample_index,
                        "generated_token_count": len(generated_ids),
                        "sample_seconds": sample_seconds,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        configure_blank_image_env([], "")
        rope_patch.__exit__(None, None, None)

    sample_summary_df = pd.DataFrame(sample_summary_rows)
    condition_score_df = pd.DataFrame(condition_score_rows)
    shapley_df = pd.DataFrame(shapley_rows)
    timing_df = pd.DataFrame(timing_rows)

    sample_summary_df.to_csv(output_dir / "sample_summary.csv", index=False)
    condition_score_df.to_csv(output_dir / "condition_scores.csv", index=False)
    shapley_df.to_csv(output_dir / "shapley_per_step.csv", index=False)
    timing_df.to_csv(output_dir / "timing.csv", index=False)

    if not shapley_df.empty:
        shapley_summary = (
            shapley_df.groupby("step", as_index=False)[
                [
                    "phi_image1_logprob",
                    "phi_image2_logprob",
                    "interaction_logprob",
                    "phi_image1_margin",
                    "phi_image2_margin",
                    "interaction_margin",
                ]
            ]
            .mean(numeric_only=True)
        )
        shapley_summary.to_csv(output_dir / "shapley_step_mean.csv", index=False)
        phi_image1_logprob_auc = float(shapley_summary["phi_image1_logprob"].sum())
        phi_image2_logprob_auc = float(shapley_summary["phi_image2_logprob"].sum())
        phi_image1_margin_auc = float(shapley_summary["phi_image1_margin"].sum())
        phi_image2_margin_auc = float(shapley_summary["phi_image2_margin"].sum())
    else:
        shapley_summary = pd.DataFrame()
        phi_image1_logprob_auc = float("nan")
        phi_image2_logprob_auc = float("nan")
        phi_image1_margin_auc = float("nan")
        phi_image2_margin_auc = float("nan")

    summary = {
        "dataset": args.dataset,
        "indices": args.indices,
        "mode": args.mode,
        "policy": args.policy,
        "template_on_last_replay_text": args.template_on_last_replay_text,
        "rope_align": bool(args.rope_align),
        "max_new_tokens": args.max_new_tokens,
        "sample_count": len(args.indices),
        "processed_sample_count": len(sample_summary_df),
        "run_seconds": time.perf_counter() - run_start,
        "mean_sample_seconds": float(timing_df["sample_seconds"].mean()) if not timing_df.empty else float("nan"),
        "mean_generated_token_count": (
            float(sample_summary_df["generated_token_count"].mean()) if not sample_summary_df.empty else float("nan")
        ),
        "phi_image1_logprob_auc": phi_image1_logprob_auc,
        "phi_image2_logprob_auc": phi_image2_logprob_auc,
        "phi_image1_margin_auc": phi_image1_margin_auc,
        "phi_image2_margin_auc": phi_image2_margin_auc,
        "rope_align_log": rope_patch.align_log,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
