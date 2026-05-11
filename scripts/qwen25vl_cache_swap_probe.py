#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from qwen25vl_image2_probe import (
    LastLayerAttentionTracer,
    aggregate_attention,
    build_base_content,
    build_replayed_content,
    build_inputs,
    find_image_spans,
    find_mid_text_positions,
    load_model_and_processor,
    set_seed,
    tensor_to_device,
)
from vlmeval.dataset import build_dataset


def flush_cache_swap_outputs(
    *,
    output_dir: Path,
    compare_rows: list[dict],
    sample_rows: list[dict],
    generated_rows: list[dict],
    timing_rows: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compare_df = pd.DataFrame(compare_rows)
    samples_df = pd.DataFrame(sample_rows)
    forced_df = pd.DataFrame(generated_rows)
    timing_df = pd.DataFrame(timing_rows)
    compare_df.to_csv(output_dir / "cache_swap_compare.csv", index=False)
    samples_df.to_csv(output_dir / "sample_summary.csv", index=False)
    forced_df.to_csv(output_dir / "forced_tokens.csv", index=False)
    timing_df.to_csv(output_dir / "timing.csv", index=False)
    return compare_df, samples_df, forced_df, timing_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare original vs swapped image cache slices after prefill for Qwen2.5-VL."
    )
    parser.add_argument("--model-path", default="/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset", default="SEEDBench2_Plus")
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--mode", default="image_text_image")
    parser.add_argument("--policy", default="identity")
    parser.add_argument("--template-on-last-replay-text", action="store_true")
    parser.add_argument(
        "--swap-mode",
        default="kv",
        choices=["kv", "k", "v"],
        help="Which cache tensors to swap between image_1 and image_2 spans.",
    )
    parser.add_argument("--teacher-force-steps", type=int, default=4)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Greedy-generate up to this many tokens to build the teacher-forced decode path.",
    )
    parser.add_argument(
        "--skip-short-samples",
        action="store_true",
        help="Skip samples whose greedy decode is shorter than teacher-force-steps.",
    )
    parser.add_argument(
        "--head-reduction",
        default="per_head",
        choices=["per_head", "mean"],
        help="Store per-head attention rows or mean-reduced rows for each sample/step.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    return parser


def clone_cache(cache):
    return copy.deepcopy(cache)


def swap_cache_slices(cache, span1: tuple[int, int], span2: tuple[int, int], swap_mode: str) -> None:
    s1, e1 = span1
    s2, e2 = span2
    len1 = e1 - s1 + 1
    len2 = e2 - s2 + 1
    if len1 != len2:
        raise ValueError(f"Image span lengths differ: {len1} vs {len2}")
    if swap_mode not in {"kv", "k", "v"}:
        raise ValueError(f"Unsupported swap_mode: {swap_mode}")
    for layer_idx in range(len(cache.key_cache)):
        key = cache.key_cache[layer_idx]
        val = cache.value_cache[layer_idx]
        if swap_mode in {"kv", "k"}:
            key1 = key[:, :, s1 : e1 + 1, :].clone()
            key2 = key[:, :, s2 : e2 + 1, :].clone()
            key[:, :, s1 : e1 + 1, :] = key2
            key[:, :, s2 : e2 + 1, :] = key1
        if swap_mode in {"kv", "v"}:
            val1 = val[:, :, s1 : e1 + 1, :].clone()
            val2 = val[:, :, s2 : e2 + 1, :].clone()
            val[:, :, s1 : e1 + 1, :] = val2
            val[:, :, s2 : e2 + 1, :] = val1


def run_one_variant_step(
    *,
    model,
    tracer,
    token_input: torch.Tensor,
    cache,
    attention_mask: torch.Tensor,
    cache_position_value: int,
    spans_for_attention: dict[str, list[int]],
    head_reduction: str,
):
    tracer.reset()
    model_inputs = model.prepare_inputs_for_generation(
        token_input,
        past_key_values=cache,
        attention_mask=attention_mask,
        cache_position=torch.tensor([cache_position_value], device=token_input.device),
        use_cache=True,
    )
    with torch.inference_mode():
        outputs = model(**model_inputs, return_dict=True)
    decode_record = None
    for record in tracer.records:
        if record["q_len"] == 1:
            decode_record = record
    if decode_record is None:
        raise RuntimeError("Failed to capture q_len=1 decode attention record.")
    rows = aggregate_attention(
        decode_record["attn_last"],
        spans_for_attention,
        head_reduction=head_reduction,
    )
    return outputs, rows


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args.dataset)
    processor, model = load_model_and_processor(args.model_path, args.device)

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    special_token_ids = set(processor.tokenizer.all_special_ids)

    tracer = LastLayerAttentionTracer(model.model.language_model.layers[-1].self_attn)
    tracer.patch()

    compare_rows = []
    sample_rows = []
    generated_rows = []
    timing_rows = []
    run_start = time.perf_counter()
    eos_token_id = processor.tokenizer.eos_token_id

    try:
        for sample_index in args.indices:
            sample_start = time.perf_counter()
            row = dataset.data.iloc[sample_index]
            base_content = build_base_content(dataset, row)
            content = build_replayed_content(
                base_content,
                args.dataset,
                mode=args.mode,
                policy=args.policy,
                template_on_last_replay_text=args.template_on_last_replay_text,
            )
            _, prompt_text, model_inputs = build_inputs(processor, content)
            input_ids = model_inputs["input_ids"][0].tolist()
            image_spans = find_image_spans(input_ids, image_token_id)
            if len(image_spans) < 2:
                raise ValueError(
                    f"Need two image spans for cache-swap probe, got {len(image_spans)} on sample {sample_index}."
                )
            text_positions = find_mid_text_positions(input_ids, image_spans, special_token_ids)
            model_inputs = tensor_to_device(model_inputs, args.device)

            sample_rows.append(
                {
                    "sample_index": sample_index,
                    "question_id": str(row.get("index", sample_index)),
                    "subcategory": row.get("subcategory", None),
                    "prompt_len": len(input_ids),
                    "image1_span": [image_spans[0].start, image_spans[0].end],
                    "image2_span": [image_spans[1].start, image_spans[1].end],
                    "text_token_count": len(text_positions),
                    "prompt_preview": prompt_text[:500],
                }
            )

            spans_for_attention = {
                "image1": list(range(image_spans[0].start, image_spans[0].end + 1)),
                "image2": list(range(image_spans[1].start, image_spans[1].end + 1)),
                "text": text_positions,
            }

            tracer.reset()
            prefill_start = time.perf_counter()
            with torch.inference_mode():
                prefill_outputs = model(**model_inputs, use_cache=True, return_dict=True)
            prefill_seconds = time.perf_counter() - prefill_start

            base_cache = prefill_outputs.past_key_values
            original_cache = clone_cache(base_cache)
            swapped_cache = clone_cache(base_cache)
            swap_cache_slices(
                swapped_cache,
                (image_spans[0].start, image_spans[0].end),
                (image_spans[1].start, image_spans[1].end),
                args.swap_mode,
            )

            prompt_len = int(model_inputs["input_ids"].shape[-1])
            path_start = time.perf_counter()
            tracer.reset()
            with torch.inference_mode():
                path_sequences = model.generate(
                    **model_inputs,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=args.max_new_tokens or args.teacher_force_steps,
                )
            path_seconds = time.perf_counter() - path_start
            generated_ids = path_sequences[0, prompt_len:].detach().cpu().tolist()
            if eos_token_id is not None and eos_token_id in generated_ids:
                eos_pos = generated_ids.index(eos_token_id) + 1
                generated_ids = generated_ids[:eos_pos]
            if len(generated_ids) < args.teacher_force_steps:
                generated_rows.append(
                    {
                        "sample_index": sample_index,
                        "generated_token_count": len(generated_ids),
                        "teacher_force_steps": args.teacher_force_steps,
                        "used_for_compare": False,
                        "forced_token_ids": generated_ids,
                        "forced_text": processor.tokenizer.decode(
                            generated_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ),
                    }
                )
                sample_rows[-1]["generated_token_count"] = len(generated_ids)
                sample_rows[-1]["used_for_compare"] = False
                sample_rows[-1]["skip_reason"] = "too_short_for_teacher_force"
                timing_row = {
                    "sample_index": sample_index,
                    "prompt_len": len(input_ids),
                    "teacher_force_steps": args.teacher_force_steps,
                    "generated_token_count": len(generated_ids),
                    "prefill_seconds": prefill_seconds,
                    "path_seconds": path_seconds,
                    "decode_seconds": 0.0,
                    "sample_seconds": time.perf_counter() - sample_start,
                    "used_for_compare": False,
                }
                timing_rows.append(timing_row)
                print(
                    json.dumps(
                        {
                            "event": "sample_skipped",
                            **timing_row,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                del prefill_outputs, path_sequences
                torch.cuda.empty_cache()
                flush_cache_swap_outputs(
                    output_dir=output_dir,
                    compare_rows=compare_rows,
                    sample_rows=sample_rows,
                    generated_rows=generated_rows,
                    timing_rows=timing_rows,
                )
                if args.skip_short_samples:
                    continue
                raise ValueError(
                    f"Sample {sample_index} generated only {len(generated_ids)} tokens, "
                    f"shorter than teacher-force-steps={args.teacher_force_steps}."
                )

            forced_token_ids = generated_ids[: args.teacher_force_steps]
            token_input = torch.tensor([[forced_token_ids[0]]], device=model_inputs["input_ids"].device)
            original_prefix_attention_mask = model_inputs["attention_mask"].clone()
            swapped_prefix_attention_mask = model_inputs["attention_mask"].clone()
            cache_position_value = prompt_len
            decode_start = time.perf_counter()

            for step in range(args.teacher_force_steps):
                original_attention_mask = torch.cat(
                    [
                        original_prefix_attention_mask,
                        torch.ones(
                            (original_prefix_attention_mask.shape[0], token_input.shape[-1]),
                            dtype=original_prefix_attention_mask.dtype,
                            device=original_prefix_attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                swapped_attention_mask = torch.cat(
                    [
                        swapped_prefix_attention_mask,
                        torch.ones(
                            (swapped_prefix_attention_mask.shape[0], token_input.shape[-1]),
                            dtype=swapped_prefix_attention_mask.dtype,
                            device=swapped_prefix_attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                original_outputs, original_rows = run_one_variant_step(
                    model=model,
                    tracer=tracer,
                    token_input=token_input,
                    cache=original_cache,
                    attention_mask=original_attention_mask,
                    cache_position_value=cache_position_value,
                    spans_for_attention=spans_for_attention,
                    head_reduction=args.head_reduction,
                )
                swapped_outputs, swapped_rows = run_one_variant_step(
                    model=model,
                    tracer=tracer,
                    token_input=token_input,
                    cache=swapped_cache,
                    attention_mask=swapped_attention_mask,
                    cache_position_value=cache_position_value,
                    spans_for_attention=spans_for_attention,
                    head_reduction=args.head_reduction,
                )

                token_id = int(token_input.item())
                token_text = processor.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                for variant, rows in [("original", original_rows), ("swapped", swapped_rows)]:
                    for head_row in rows:
                        head_row.update(
                            {
                                "sample_index": sample_index,
                                "step": step,
                                "variant": variant,
                                "swap_mode": args.swap_mode,
                                "token_id": token_id,
                                "token_text": token_text,
                            }
                        )
                        compare_rows.append(head_row)

                original_prefix_attention_mask = original_attention_mask
                swapped_prefix_attention_mask = swapped_attention_mask
                cache_position_value += 1
                if step + 1 < args.teacher_force_steps:
                    token_input = torch.tensor(
                        [[forced_token_ids[step + 1]]],
                        device=model_inputs["input_ids"].device,
                    )

            decode_seconds = time.perf_counter() - decode_start

            generated_rows.append(
                {
                    "sample_index": sample_index,
                    "generated_token_count": len(generated_ids),
                    "teacher_force_steps": args.teacher_force_steps,
                    "used_for_compare": True,
                    "forced_token_ids": forced_token_ids,
                    "forced_text": processor.tokenizer.decode(
                        forced_token_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                }
            )
            sample_rows[-1]["generated_token_count"] = len(generated_ids)
            sample_rows[-1]["used_for_compare"] = True
            sample_rows[-1]["skip_reason"] = None
            del prefill_outputs, path_sequences, original_outputs, swapped_outputs
            torch.cuda.empty_cache()
            sample_seconds = time.perf_counter() - sample_start
            timing_row = {
                "sample_index": sample_index,
                "prompt_len": len(input_ids),
                "teacher_force_steps": args.teacher_force_steps,
                "generated_token_count": len(generated_ids),
                "prefill_seconds": prefill_seconds,
                "path_seconds": path_seconds,
                "decode_seconds": decode_seconds,
                "sample_seconds": sample_seconds,
                "used_for_compare": True,
            }
            timing_rows.append(timing_row)
            print(
                json.dumps(
                    {
                        "event": "sample_complete",
                        **timing_row,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            flush_cache_swap_outputs(
                output_dir=output_dir,
                compare_rows=compare_rows,
                sample_rows=sample_rows,
                generated_rows=generated_rows,
                timing_rows=timing_rows,
            )

    finally:
        tracer.restore()

    compare_df, samples_df, forced_df, timing_df = flush_cache_swap_outputs(
        output_dir=output_dir,
        compare_rows=compare_rows,
        sample_rows=sample_rows,
        generated_rows=generated_rows,
        timing_rows=timing_rows,
    )

    step0 = compare_df[compare_df["step"] == 0]
    if step0.empty:
        mean_by_variant = {}
        original_prefers_image2 = 0
        swapped_prefers_image1 = 0
    else:
        mean_by_variant = (
            step0.groupby("variant")[
                ["image1_mass", "image2_mass", "text_mass", "image1_l2", "image2_l2", "text_l2"]
            ]
            .mean()
            .round(6)
            .to_dict(orient="index")
        )
        sample_variant = step0.groupby(["sample_index", "variant"])[["image1_mass", "image2_mass", "text_mass"]].mean()
        pivot = sample_variant.reset_index().pivot(index="sample_index", columns="variant")
        original_prefers_image2 = int(
            (
                pivot[("image2_mass", "original")] > pivot[("image1_mass", "original")]
            ).sum()
        )
        swapped_prefers_image1 = int(
            (
                pivot[("image1_mass", "swapped")] > pivot[("image2_mass", "swapped")]
            ).sum()
        )

    summary = {
        "dataset": args.dataset,
        "indices": args.indices,
        "mode": args.mode,
        "policy": args.policy,
        "swap_mode": args.swap_mode,
        "teacher_force_steps": args.teacher_force_steps,
        "max_new_tokens": args.max_new_tokens,
        "head_reduction": args.head_reduction,
        "sample_count": len(args.indices),
        "processed_sample_count": len(timing_df),
        "compared_sample_count": int(timing_df["used_for_compare"].sum()) if not timing_df.empty else 0,
        "run_seconds": time.perf_counter() - run_start,
        "mean_sample_seconds": (
            float(timing_df["sample_seconds"].mean()) if not timing_df.empty else float("nan")
        ),
        "mean_prefill_seconds": (
            float(timing_df["prefill_seconds"].mean()) if not timing_df.empty else float("nan")
        ),
        "mean_path_seconds": (
            float(timing_df["path_seconds"].mean()) if not timing_df.empty else float("nan")
        ),
        "mean_decode_seconds": (
            float(timing_df["decode_seconds"].mean()) if not timing_df.empty else float("nan")
        ),
        "compare_row_count": int(len(compare_df)),
        "step0_mean_by_variant": mean_by_variant,
        "original_step0_image2_gt_image1_count": original_prefers_image2,
        "swapped_step0_image1_gt_image2_count": swapped_prefers_image1,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
