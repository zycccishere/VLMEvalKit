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

from qwen25vl_faithful_attribution_probe import extend_teacher_forced_inputs
from qwen25vl_image2_probe import (
    RopeAlignPatch,
    build_base_content,
    build_inputs,
    build_replayed_content,
    find_image_spans,
    load_model_and_processor,
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
        description="Gradient-style attribution probe for Qwen2.5-VL image1/image2 contribution."
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
        "--max-attribution-steps",
        type=int,
        default=64,
        help="Maximum number of generated decode steps used in the aggregate target score.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["saliency", "grad_x_input", "integrated_gradients"],
        choices=["saliency", "grad_x_input", "integrated_gradients"],
    )
    parser.add_argument(
        "--score-kinds",
        nargs="+",
        default=["logprob", "margin"],
        choices=["logprob", "margin"],
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=4,
        help="Number of interpolation steps for integrated gradients.",
    )
    parser.add_argument("--rope-align", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    return parser


def decode_generated_text(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def prepare_model_forward_with_embeds(
    *,
    model,
    full_inputs: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor]:
    input_ids = full_inputs["input_ids"]
    inputs_embeds = model.get_input_embeddings()(input_ids)

    pixel_values = full_inputs.get("pixel_values")
    image_grid_thw = full_inputs.get("image_grid_thw")
    if pixel_values is not None:
        image_embeds = model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)
        n_image_tokens = int((input_ids == model.config.image_token_id).sum().item())
        if n_image_tokens != image_embeds.shape[0]:
            raise ValueError(
                f"Image features and image tokens do not match: tokens={n_image_tokens}, features={image_embeds.shape[0]}"
            )
        mask = input_ids == model.config.image_token_id
        image_mask = mask.unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    pixel_values_videos = full_inputs.get("pixel_values_videos")
    video_grid_thw = full_inputs.get("video_grid_thw")
    if pixel_values_videos is not None:
        video_embeds = model.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0)
        n_video_tokens = int((input_ids == model.config.video_token_id).sum().item())
        if n_video_tokens != video_embeds.shape[0]:
            raise ValueError(
                f"Video features and video tokens do not match: tokens={n_video_tokens}, features={video_embeds.shape[0]}"
            )
        mask = input_ids == model.config.video_token_id
        video_mask = mask.unsqueeze(-1).expand_as(inputs_embeds)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    inputs_embeds = inputs_embeds.detach().clone().requires_grad_(True)
    model_forward = {
        "input_ids": input_ids,
        "attention_mask": full_inputs.get("attention_mask"),
        "position_ids": full_inputs.get("position_ids"),
        "past_key_values": full_inputs.get("past_key_values"),
        "inputs_embeds": inputs_embeds,
        "pixel_values": None,
        "pixel_values_videos": None,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
        "rope_deltas": full_inputs.get("rope_deltas"),
        "cache_position": full_inputs.get("cache_position"),
        "second_per_grid_ts": full_inputs.get("second_per_grid_ts"),
        "use_cache": False,
        "return_dict": True,
    }
    return model_forward, inputs_embeds


def compute_target_scores(
    *,
    logits: torch.Tensor,
    prompt_len: int,
    generated_ids: list[int],
    max_attribution_steps: int,
) -> dict[str, torch.Tensor]:
    if not generated_ids:
        raise ValueError("No generated tokens to score.")
    n_steps = min(len(generated_ids), int(max_attribution_steps))
    logprob_terms: list[torch.Tensor] = []
    margin_terms: list[torch.Tensor] = []
    for step, token_id in enumerate(generated_ids[:n_steps]):
        score_pos = prompt_len - 1 + step
        next_logits = logits[score_pos]
        log_probs = F.log_softmax(next_logits.float(), dim=-1)
        target_logprob = log_probs[token_id]
        target_logit = next_logits[token_id].float()
        masked_logits = next_logits.float().clone()
        masked_logits[token_id] = float("-inf")
        best_alt_logit = masked_logits.max()
        logprob_terms.append(target_logprob)
        margin_terms.append(target_logit - best_alt_logit)
    return {
        "logprob": torch.stack(logprob_terms).mean(),
        "margin": torch.stack(margin_terms).mean(),
        "n_steps": torch.tensor(n_steps),
    }


def span_mean_abs_grad(grad: torch.Tensor, start: int, end: int) -> float:
    chunk = grad[start : end + 1]
    return float(chunk.abs().sum(dim=-1).mean().item())


def span_mean_grad_x_input(grad: torch.Tensor, embeds: torch.Tensor, start: int, end: int) -> float:
    chunk = (grad[start : end + 1] * embeds[start : end + 1]).sum(dim=-1)
    return float(chunk.mean().item())


def aggregate_metric(
    *,
    metric: str,
    grad: torch.Tensor,
    actual_embeds: torch.Tensor,
    span1: tuple[int, int],
    span2: tuple[int, int],
) -> tuple[float, float]:
    if metric == "saliency":
        return (
            span_mean_abs_grad(grad, *span1),
            span_mean_abs_grad(grad, *span2),
        )
    if metric == "grad_x_input":
        return (
            span_mean_grad_x_input(grad, actual_embeds, *span1),
            span_mean_grad_x_input(grad, actual_embeds, *span2),
        )
    raise ValueError(metric)


def zero_image_baseline(actual_embeds: torch.Tensor, span1: tuple[int, int], span2: tuple[int, int]) -> torch.Tensor:
    baseline = actual_embeds.detach().clone()
    baseline[span1[0] : span1[1] + 1] = 0.0
    baseline[span2[0] : span2[1] + 1] = 0.0
    return baseline


def integrated_gradients_metric(
    *,
    model,
    model_forward: dict[str, Any],
    actual_embeds: torch.Tensor,
    span1: tuple[int, int],
    span2: tuple[int, int],
    prompt_len: int,
    generated_ids: list[int],
    score_kind: str,
    max_attribution_steps: int,
    ig_steps: int,
) -> tuple[float, float]:
    baseline = zero_image_baseline(actual_embeds, span1, span2).unsqueeze(0)
    actual_batch = actual_embeds.unsqueeze(0)
    diff = actual_batch - baseline
    grad_accum = torch.zeros_like(actual_batch)
    for alpha in torch.linspace(
        1.0 / ig_steps, 1.0, ig_steps, device=actual_batch.device, dtype=actual_batch.dtype
    ):
        interp = (baseline + alpha * diff).detach().clone().requires_grad_(True)
        forward_kwargs = dict(model_forward)
        forward_kwargs["inputs_embeds"] = interp
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            outputs = model(**forward_kwargs)
            scalar = compute_target_scores(
                logits=outputs.logits[0],
                prompt_len=prompt_len,
                generated_ids=generated_ids,
                max_attribution_steps=max_attribution_steps,
            )[score_kind]
        if not scalar.requires_grad:
            raise RuntimeError(f"Integrated gradients scalar lost grad_fn for score_kind={score_kind}.")
        scalar.backward()
        if interp.grad is None:
            raise RuntimeError(f"Integrated gradients batch tensor has no grad for score_kind={score_kind}.")
        grad_accum = grad_accum + interp.grad.detach()
        del outputs
    integrated = (diff * (grad_accum / float(ig_steps)))[0]
    image1 = span_mean_grad_x_input(
        torch.ones_like(integrated),
        integrated,
        *span1,
    )
    image2 = span_mean_grad_x_input(
        torch.ones_like(integrated),
        integrated,
        *span2,
    )
    return image1, image2


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    sanitize_single_process_env()

    dataset = build_dataset(args.dataset)
    processor, model = load_model_and_processor(args.model_path, args.device)
    device = resolve_input_device(model, args.device)
    image_token_id = int(model.config.image_token_id)
    for param in model.parameters():
        param.requires_grad_(False)

    generated_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    rope_patch = RopeAlignPatch(model, image_token_id, enabled=args.rope_align)
    rope_patch.__enter__()
    try:
        for sample_index in args.indices:
            row = dataset.data.iloc[sample_index]
            base_content = build_base_content(dataset, row)
            replayed_content = build_replayed_content(
                base_content,
                args.dataset,
                mode=args.mode,
                policy=args.policy,
                template_on_last_replay_text=args.template_on_last_replay_text,
            )
            _, prompt_text, model_inputs = build_inputs(processor, replayed_content)
            model_inputs = tensor_to_device(model_inputs, device)
            prompt_input_ids = model_inputs["input_ids"][0].tolist()
            image_spans = find_image_spans(prompt_input_ids, image_token_id)
            if len(image_spans) < 2:
                raise ValueError(f"Need two image spans, got {len(image_spans)} for sample {sample_index}.")
            span1 = (image_spans[0].start, image_spans[0].end)
            span2 = (image_spans[1].start, image_spans[1].end)

            with torch.inference_mode():
                sequences = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            prompt_len = int(model_inputs["input_ids"].shape[1])
            generated_ids = sequences[0, prompt_len:].detach().cpu().tolist()
            generated_text = decode_generated_text(processor.tokenizer, generated_ids)
            generated_rows.append(
                {
                    "sample_index": sample_index,
                    "generated_token_count": len(generated_ids),
                    "attributed_step_count": min(len(generated_ids), int(args.max_attribution_steps)),
                    "generated_text": generated_text,
                    "prompt_len": prompt_len,
                    "image1_span_start": span1[0],
                    "image1_span_end": span1[1],
                    "image2_span_start": span2[0],
                    "image2_span_end": span2[1],
                    "prompt_preview": prompt_text[:500],
                }
            )
            if not generated_ids:
                continue

            full_inputs = extend_teacher_forced_inputs(model_inputs, generated_ids)
            model_forward, inputs_embeds = prepare_model_forward_with_embeds(model=model, full_inputs=full_inputs)
            actual_embeds = inputs_embeds[0].detach().clone()

            for score_kind in args.score_kinds:
                if any(metric in {"saliency", "grad_x_input"} for metric in args.metrics):
                    model.zero_grad(set_to_none=True)
                    forward_inputs = dict(model_forward)
                    forward_inputs["inputs_embeds"] = inputs_embeds
                    with torch.enable_grad():
                        outputs = model(**forward_inputs)
                        score_map = compute_target_scores(
                            logits=outputs.logits[0],
                            prompt_len=prompt_len,
                            generated_ids=generated_ids,
                            max_attribution_steps=args.max_attribution_steps,
                        )
                        scalar = score_map[score_kind]
                    if not scalar.requires_grad:
                        raise RuntimeError(
                            f"Scalar score lost grad_fn for score_kind={score_kind} sample={sample_index}."
                        )
                    scalar.backward()
                    grad = inputs_embeds.grad[0].detach().clone()
                    for metric in args.metrics:
                        if metric == "integrated_gradients":
                            continue
                        image1, image2 = aggregate_metric(
                            metric=metric,
                            grad=grad,
                            actual_embeds=actual_embeds,
                            span1=span1,
                            span2=span2,
                        )
                        metric_rows.append(
                            {
                                "sample_index": sample_index,
                                "score_kind": score_kind,
                                "metric": metric,
                                "generated_token_count": len(generated_ids),
                                "attributed_step_count": int(score_map["n_steps"].item()),
                                "image1": image1,
                                "image2": image2,
                                "image2_over_image1": image2 - image1,
                            }
                        )
                    del outputs
                    inputs_embeds.grad = None
                    torch.cuda.empty_cache()

                if "integrated_gradients" in args.metrics:
                    image1, image2 = integrated_gradients_metric(
                        model=model,
                        model_forward=model_forward,
                        actual_embeds=actual_embeds,
                        span1=span1,
                        span2=span2,
                        prompt_len=prompt_len,
                        generated_ids=generated_ids,
                        score_kind=score_kind,
                        max_attribution_steps=args.max_attribution_steps,
                        ig_steps=args.ig_steps,
                    )
                    metric_rows.append(
                        {
                            "sample_index": sample_index,
                            "score_kind": score_kind,
                            "metric": "integrated_gradients",
                            "generated_token_count": len(generated_ids),
                            "attributed_step_count": min(len(generated_ids), int(args.max_attribution_steps)),
                            "image1": image1,
                            "image2": image2,
                            "image2_over_image1": image2 - image1,
                        }
                    )
                    torch.cuda.empty_cache()
    finally:
        rope_patch.__exit__(None, None, None)

    pd.DataFrame(generated_rows).to_csv(output_dir / "generated_text.csv", index=False)
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(output_dir / "metric_rows.csv", index=False)
    summary = {}
    if not metric_df.empty:
        for (score_kind, metric), group in metric_df.groupby(["score_kind", "metric"], dropna=False):
            summary[f"{score_kind}::{metric}"] = {
                "n_samples": int(group["sample_index"].nunique()),
                "mean_image1": float(group["image1"].mean()),
                "mean_image2": float(group["image2"].mean()),
                "mean_gap": float(group["image2_over_image1"].mean()),
            }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
