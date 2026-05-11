#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from qwen25vl_image2_probe import (
    RopeAlignPatch,
    build_base_content,
    build_inputs,
    build_replayed_content,
    configure_blank_image_env,
    find_image_spans,
    find_mid_text_positions,
    load_model_and_processor,
    resolve_blank_image_path_for_sample,
    resolve_input_device,
    sanitize_single_process_env,
    set_seed,
    tensor_to_device,
)
from vlmeval.dataset import build_dataset
from vlmeval.vlm.qwen2_vl.replay_prompt_template import (
    PROMPT_TEMPLATE_DIRECTLY_ANSWER,
    PROMPT_TEMPLATE_IDENTITY,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Faithful output-level attribution probe for Qwen2.5-VL image_text_image replay."
    )
    parser.add_argument("--model-path", default="/models/Qwen2.5-VL-32B-Instruct")
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
    parser.add_argument(
        "--score-kind",
        default="logprob",
        choices=["logprob", "margin", "both"],
        help="Scalar score used for Shapley. 'both' stores both logprob and logit-margin variants.",
    )
    parser.add_argument(
        "--blank-image-match-source-size",
        action="store_true",
        help="Use a same-size white image as the corruption for blanked image slots.",
    )
    parser.add_argument(
        "--corruption-family",
        default="blank",
        choices=["blank", "dataset_swap"],
        help="Corruption family used to construct no-image conditions.",
    )
    parser.add_argument("--blank-image-path", default="")
    parser.add_argument(
        "--blank-image-source-position",
        type=int,
        default=1,
        help="1-based source image position used to size the blank image when matching source size.",
    )
    parser.add_argument(
        "--swap-source-offset",
        type=int,
        default=1,
        help="Dataset offset used to find a replacement sample for dataset_swap corruption.",
    )
    parser.add_argument("--rope-align", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    return parser


CONDITIONS: dict[str, list[int]] = {
    "full": [],
    "no_image1": [1],
    "no_image2": [2],
    "no_both": [1, 2],
}


def make_metric_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def extend_teacher_forced_inputs(
    model_inputs: dict[str, Any],
    suffix_ids: list[int],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    suffix = None
    if suffix_ids:
        input_ids = model_inputs["input_ids"]
        suffix = torch.tensor([suffix_ids], dtype=input_ids.dtype, device=input_ids.device)
    for key, value in model_inputs.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        if key == "input_ids" and suffix is not None:
            out[key] = torch.cat([value, suffix], dim=1)
        elif key == "attention_mask" and suffix is not None:
            ones = torch.ones((value.shape[0], len(suffix_ids)), dtype=value.dtype, device=value.device)
            out[key] = torch.cat([value, ones], dim=1)
        else:
            out[key] = value
    return out


def prepare_condition_inputs(
    *,
    processor,
    dataset,
    sample_index: int,
    dataset_name: str,
    base_content: list[dict[str, Any]],
    mode: str,
    policy: str,
    template_on_last_replay_text: bool,
    blank_positions: list[int],
    corruption_family: str,
    blank_image_path: str,
    blank_image_match_source_size: bool,
    blank_image_source_position: int,
    swap_source_offset: int,
    blank_cache_dir: Path,
    image_token_id: int,
) -> tuple[str, dict[str, torch.Tensor], list[int], list[dict[str, int]], list[int], str]:
    resolved_corruption_ref = ""
    if corruption_family == "blank":
        resolved_corruption_ref = resolve_blank_image_path_for_sample(
            base_content=base_content,
            blank_positions=blank_positions,
            blank_image_path=blank_image_path,
            blank_image_match_source_size=blank_image_match_source_size,
            blank_image_source_position=blank_image_source_position,
            blank_cache_dir=blank_cache_dir,
        )
        configure_blank_image_env(blank_positions, resolved_corruption_ref)
    else:
        configure_blank_image_env([], "")
        resolved_corruption_ref = resolve_replacement_image_ref_for_sample(
            dataset=dataset,
            sample_index=sample_index,
            base_content=base_content,
            swap_source_offset=swap_source_offset,
        )
    content = build_replayed_content(
        base_content,
        dataset_name,
        mode=mode,
        policy=policy,
        template_on_last_replay_text=template_on_last_replay_text,
    )
    if corruption_family == "dataset_swap":
        content = replace_selected_images_with_ref(
            content,
            positions=blank_positions,
            image_ref=resolved_corruption_ref,
        )
    _, prompt_text, model_inputs = build_inputs(processor, content)
    input_ids = model_inputs["input_ids"][0].tolist()
    image_spans = [span.__dict__ for span in find_image_spans(input_ids, image_token_id)]
    special_token_ids = set(processor.tokenizer.all_special_ids)
    text_positions = find_mid_text_positions(
        input_ids,
        [type("SpanLike", (), span)() for span in image_spans],
        special_token_ids,
    )
    return prompt_text, model_inputs, input_ids, image_spans, text_positions, resolved_corruption_ref


def replace_selected_images_with_ref(
    content: list[dict[str, Any]],
    *,
    positions: list[int],
    image_ref: str,
) -> list[dict[str, Any]]:
    if not positions or not image_ref:
        return content
    selected = set(positions)
    out = []
    image_idx = 0
    for item in content:
        copied = dict(item)
        if copied.get("type") == "image":
            image_idx += 1
            if image_idx in selected:
                copied["image"] = image_ref
        out.append(copied)
    return out


def first_image_ref(content: list[dict[str, Any]]) -> str:
    for item in content:
        if item.get("type") == "image":
            return str(item.get("image", "")).strip()
    return ""


def resolve_replacement_image_ref_for_sample(
    *,
    dataset,
    sample_index: int,
    base_content: list[dict[str, Any]],
    swap_source_offset: int,
) -> str:
    source_ref = first_image_ref(base_content)
    total = len(dataset.data)
    if total <= 1:
        raise ValueError("dataset_swap corruption requires at least 2 dataset rows.")
    offset = max(1, int(swap_source_offset))
    for delta in range(offset, total + offset):
        cand_idx = (sample_index + delta) % total
        if cand_idx == sample_index:
            continue
        cand_row = dataset.data.iloc[cand_idx]
        cand_content = build_base_content(dataset, cand_row)
        cand_ref = first_image_ref(cand_content)
        if cand_ref and cand_ref != source_ref:
            return cand_ref
    raise ValueError(f"Could not find replacement image ref for sample {sample_index}.")


def decode_generated_text(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def score_sequence(
    *,
    model,
    model_inputs: dict[str, Any],
    prompt_len: int,
    generated_ids: list[int],
) -> list[dict[str, Any]]:
    if not generated_ids:
        return []
    full_inputs = extend_teacher_forced_inputs(model_inputs, generated_ids)
    with torch.inference_mode():
        outputs = model(**full_inputs, use_cache=False, return_dict=True)
    logits = outputs.logits[0]
    rows: list[dict[str, Any]] = []
    for step, token_id in enumerate(generated_ids):
        score_pos = prompt_len - 1 + step
        next_logits = logits[score_pos]
        log_probs = F.log_softmax(next_logits.float(), dim=-1)
        target_logprob = float(log_probs[token_id].item())
        target_logit = float(next_logits[token_id].item())
        masked_logits = next_logits.clone().float()
        masked_logits[token_id] = float("-inf")
        best_alt_logit = float(masked_logits.max().item())
        rows.append(
            {
                "step": step,
                "token_id": int(token_id),
                "target_logprob": target_logprob,
                "target_logit": target_logit,
                "best_alt_logit": best_alt_logit,
                "logit_margin": target_logit - best_alt_logit,
            }
        )
    del outputs
    torch.cuda.empty_cache()
    return rows


def compute_two_player_shapley(
    *,
    full_score: float,
    no_image1_score: float,
    no_image2_score: float,
    no_both_score: float,
) -> tuple[float, float, float]:
    phi_image1 = 0.5 * ((full_score - no_image2_score) + (no_image1_score - no_both_score))
    phi_image2 = 0.5 * ((full_score - no_image1_score) + (no_image2_score - no_both_score))
    interaction = full_score - no_image1_score - no_image2_score + no_both_score
    return phi_image1, phi_image2, interaction


def build_summary(
    *,
    args: argparse.Namespace,
    generated_df: pd.DataFrame,
    step_scores_df: pd.DataFrame,
    shapley_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    run_seconds: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "indices": args.indices,
        "mode": args.mode,
        "policy": args.policy,
        "template_on_last_replay_text": bool(args.template_on_last_replay_text),
        "rope_align": bool(args.rope_align),
        "score_kind": args.score_kind,
        "corruption_family": args.corruption_family,
        "max_new_tokens": int(args.max_new_tokens),
        "sample_count": len(args.indices),
        "processed_sample_count": int(len(generated_df)),
        "run_seconds": run_seconds,
    }
    if not timing_df.empty:
        summary["mean_generate_seconds"] = float(timing_df["generate_seconds"].mean())
        summary["mean_scoring_seconds"] = float(timing_df["scoring_seconds"].mean())
        summary["mean_sample_seconds"] = float(timing_df["sample_seconds"].mean())
    if not generated_df.empty:
        summary["mean_generated_token_count"] = float(generated_df["generated_token_count"].mean())
    if not shapley_df.empty:
        numeric_cols = [col for col in shapley_df.columns if col.startswith("phi_") or col.startswith("interaction_")]
        summary["mean_shapley"] = shapley_df[numeric_cols].mean(numeric_only=True).to_dict()
    if not step_scores_df.empty:
        summary["condition_counts"] = step_scores_df["condition"].value_counts().to_dict()
    return summary


def main() -> int:
    args = build_parser().parse_args()
    sanitize_single_process_env()
    set_seed(args.seed)
    if not args.blank_image_match_source_size and not args.blank_image_path:
        args.blank_image_match_source_size = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blank_cache_dir = output_dir / "_blank_cache"

    processor, model = load_model_and_processor(args.model_path, args.device)
    input_device = resolve_input_device(model, args.device)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    dataset = build_dataset(args.dataset)

    rope_patch = RopeAlignPatch(model, image_token_id, enabled=args.rope_align)
    rope_patch.__enter__()

    sample_summaries: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    step_score_rows: list[dict[str, Any]] = []
    shapley_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    run_start = time.perf_counter()

    try:
        for sample_index in args.indices:
            sample_start = time.perf_counter()
            row = dataset.data.iloc[sample_index]
            base_content = build_base_content(dataset, row)

            condition_payloads: dict[str, dict[str, Any]] = {}
            for condition, blank_positions in CONDITIONS.items():
                (
                    prompt_text,
                    model_inputs,
                    input_ids,
                    image_spans,
                    text_positions,
                    resolved_blank_path,
                ) = prepare_condition_inputs(
                    processor=processor,
                    dataset=dataset,
                    sample_index=sample_index,
                    dataset_name=args.dataset,
                    base_content=base_content,
                    mode=args.mode,
                    policy=args.policy,
                    template_on_last_replay_text=args.template_on_last_replay_text,
                    blank_positions=blank_positions,
                    corruption_family=args.corruption_family,
                    blank_image_path=args.blank_image_path,
                    blank_image_match_source_size=args.blank_image_match_source_size,
                    blank_image_source_position=args.blank_image_source_position,
                    swap_source_offset=args.swap_source_offset,
                    blank_cache_dir=blank_cache_dir,
                    image_token_id=image_token_id,
                )
                condition_payloads[condition] = {
                    "prompt_text": prompt_text,
                    "model_inputs": tensor_to_device(model_inputs, input_device),
                    "input_ids": input_ids,
                    "prompt_len": len(input_ids),
                    "image_spans": image_spans,
                    "text_positions": text_positions,
                    "blank_positions": blank_positions,
                    "corruption_ref": resolved_blank_path or None,
                }

            full_payload = condition_payloads["full"]
            generate_start = time.perf_counter()
            with torch.inference_mode():
                sequences = model.generate(
                    **full_payload["model_inputs"],
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                )
            generate_seconds = time.perf_counter() - generate_start
            prompt_len = int(full_payload["model_inputs"]["input_ids"].shape[-1])
            generated_ids = sequences[0, prompt_len:].detach().cpu().tolist()
            generated_text = decode_generated_text(processor.tokenizer, generated_ids)
            generated_rows.append(
                {
                    "sample_index": sample_index,
                    "generated_token_count": len(generated_ids),
                    "generated_text": generated_text,
                }
            )
            del sequences
            torch.cuda.empty_cache()

            sample_summaries.append(
                {
                    "sample_index": sample_index,
                    "question_id": str(row.get("index", sample_index)),
                    "question": row.get("question", None),
                    "generated_token_count": len(generated_ids),
                    "conditions": {
                        name: {
                            "prompt_len": payload["prompt_len"],
                            "image_spans": payload["image_spans"],
                            "text_token_count": len(payload["text_positions"]),
                            "blank_positions": payload["blank_positions"],
                            "corruption_ref": payload["corruption_ref"],
                        }
                        for name, payload in condition_payloads.items()
                    },
                }
            )

            scoring_start = time.perf_counter()
            score_table_by_condition: dict[str, list[dict[str, Any]]] = {}
            for condition, payload in condition_payloads.items():
                score_rows = score_sequence(
                    model=model,
                    model_inputs=payload["model_inputs"],
                    prompt_len=payload["prompt_len"],
                    generated_ids=generated_ids,
                )
                score_table_by_condition[condition] = score_rows
                for score_row in score_rows:
                    token_text = processor.tokenizer.decode(
                        [score_row["token_id"]],
                        clean_up_tokenization_spaces=False,
                    )
                    step_score_rows.append(
                        {
                            "sample_index": sample_index,
                            "condition": condition,
                            "blank_positions": ",".join(str(x) for x in payload["blank_positions"]),
                            "corruption_family": args.corruption_family,
                            "step": score_row["step"],
                            "token_id": score_row["token_id"],
                            "token_text": token_text,
                            "target_logprob": score_row["target_logprob"],
                            "target_logit": score_row["target_logit"],
                            "best_alt_logit": score_row["best_alt_logit"],
                            "logit_margin": score_row["logit_margin"],
                        }
                    )

            for step, token_id in enumerate(generated_ids):
                per_condition = {name: score_table_by_condition[name][step] for name in CONDITIONS}
                row_out = {
                    "sample_index": sample_index,
                    "step": step,
                    "token_id": int(token_id),
                    "token_text": processor.tokenizer.decode(
                        [token_id], clean_up_tokenization_spaces=False
                    ),
                }
                if args.score_kind in {"logprob", "both"}:
                    phi1, phi2, interaction = compute_two_player_shapley(
                        full_score=per_condition["full"]["target_logprob"],
                        no_image1_score=per_condition["no_image1"]["target_logprob"],
                        no_image2_score=per_condition["no_image2"]["target_logprob"],
                        no_both_score=per_condition["no_both"]["target_logprob"],
                    )
                    row_out.update(
                        {
                            "phi_image1_logprob": phi1,
                            "phi_image2_logprob": phi2,
                            "interaction_logprob": interaction,
                            "full_logprob": per_condition["full"]["target_logprob"],
                            "no_image1_logprob": per_condition["no_image1"]["target_logprob"],
                            "no_image2_logprob": per_condition["no_image2"]["target_logprob"],
                            "no_both_logprob": per_condition["no_both"]["target_logprob"],
                        }
                    )
                if args.score_kind in {"margin", "both"}:
                    phi1, phi2, interaction = compute_two_player_shapley(
                        full_score=per_condition["full"]["logit_margin"],
                        no_image1_score=per_condition["no_image1"]["logit_margin"],
                        no_image2_score=per_condition["no_image2"]["logit_margin"],
                        no_both_score=per_condition["no_both"]["logit_margin"],
                    )
                    row_out.update(
                        {
                            "phi_image1_margin": phi1,
                            "phi_image2_margin": phi2,
                            "interaction_margin": interaction,
                            "full_margin": per_condition["full"]["logit_margin"],
                            "no_image1_margin": per_condition["no_image1"]["logit_margin"],
                            "no_image2_margin": per_condition["no_image2"]["logit_margin"],
                            "no_both_margin": per_condition["no_both"]["logit_margin"],
                        }
                    )
                shapley_rows.append(row_out)
            scoring_seconds = time.perf_counter() - scoring_start

            sample_seconds = time.perf_counter() - sample_start
            timing_row = {
                "sample_index": sample_index,
                "generated_token_count": len(generated_ids),
                "generate_seconds": generate_seconds,
                "scoring_seconds": scoring_seconds,
                "sample_seconds": sample_seconds,
            }
            timing_rows.append(timing_row)
            print(json.dumps({"event": "sample_complete", **timing_row}, ensure_ascii=False), flush=True)

            (output_dir / "sample_summary.json").write_text(
                json.dumps(sample_summaries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            make_metric_frame(generated_rows).to_csv(output_dir / "generated_text.csv", index=False)
            make_metric_frame(step_score_rows).to_csv(output_dir / "step_scores.csv", index=False)
            make_metric_frame(shapley_rows).to_csv(output_dir / "step_shapley.csv", index=False)
            make_metric_frame(timing_rows).to_csv(output_dir / "timing.csv", index=False)

    finally:
        configure_blank_image_env([], "")
        rope_patch.__exit__(None, None, None)

    generated_df = make_metric_frame(generated_rows)
    step_scores_df = make_metric_frame(step_score_rows)
    shapley_df = make_metric_frame(shapley_rows)
    timing_df = make_metric_frame(timing_rows)

    generated_df.to_csv(output_dir / "generated_text.csv", index=False)
    step_scores_df.to_csv(output_dir / "step_scores.csv", index=False)
    shapley_df.to_csv(output_dir / "step_shapley.csv", index=False)
    timing_df.to_csv(output_dir / "timing.csv", index=False)

    summary = build_summary(
        args=args,
        generated_df=generated_df,
        step_scores_df=step_scores_df,
        shapley_df=shapley_df,
        timing_df=timing_df,
        run_seconds=time.perf_counter() - run_start,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
