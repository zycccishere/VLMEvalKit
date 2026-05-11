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

from qwen25vl_faithful_attribution_probe import (
    build_summary,
    compute_two_player_shapley,
    decode_generated_text,
    extend_teacher_forced_inputs,
    make_metric_frame,
    prepare_condition_inputs,
)
from qwen25vl_image2_probe import (
    RopeAlignPatch,
    build_base_content,
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
        description="Grad×input token-group attribution probe for Qwen2.5-VL image_text_image replay."
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
        default="both",
        choices=["logprob", "margin", "both"],
    )
    parser.add_argument(
        "--max-scored-steps",
        type=int,
        default=32,
        help="Maximum number of decode steps to run teacher-forced grad×input attribution on.",
    )
    parser.add_argument("--rope-align", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    return parser


class LanguageModelInputCapture:
    def __init__(self, language_model):
        self.language_model = language_model
        self.handle = None
        self.inputs_embeds: torch.Tensor | None = None

    def _hook(self, module, args, kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None and len(args) >= 5:
            inputs_embeds = args[4]
        if inputs_embeds is None:
            raise ValueError("LanguageModelInputCapture did not receive inputs_embeds.")
        inputs_embeds.retain_grad()
        self.inputs_embeds = inputs_embeds

    def __enter__(self):
        self.handle = self.language_model.register_forward_pre_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def score_step_with_gradxinput(
    *,
    model,
    language_model,
    model_inputs: dict[str, Any],
    prefix_ids: list[int],
    target_token_id: int,
    image1_positions: list[int],
    image2_positions: list[int],
    text_positions: list[int],
    score_kind: str,
) -> dict[str, float]:
    full_inputs = extend_teacher_forced_inputs(model_inputs, prefix_ids)
    model.zero_grad(set_to_none=True)

    with LanguageModelInputCapture(language_model) as capture:
        outputs = model(**full_inputs, use_cache=False, return_dict=True)
        logits = outputs.logits[0, -1]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        target_logprob = log_probs[target_token_id]
        target_logit = logits[target_token_id]
        masked_logits = logits.float().clone()
        masked_logits[target_token_id] = float("-inf")
        best_alt_logit = masked_logits.max()
        logit_margin = target_logit.float() - best_alt_logit

        metrics: dict[str, float] = {
            "target_logprob": float(target_logprob.item()),
            "target_logit": float(target_logit.item()),
            "best_alt_logit": float(best_alt_logit.item()),
            "logit_margin": float(logit_margin.item()),
        }
        score_specs: list[tuple[str, torch.Tensor]] = []
        if score_kind in {"logprob", "both"}:
            score_specs.append(("logprob", target_logprob))
        if score_kind in {"margin", "both"}:
            score_specs.append(("margin", logit_margin))

        embeds = capture.inputs_embeds
        if embeds is None:
            raise ValueError("LanguageModelInputCapture failed to capture inputs_embeds.")
        token_embeds = embeds[0]

        for idx, (suffix, score_tensor) in enumerate(score_specs):
            grads = torch.autograd.grad(
                score_tensor,
                embeds,
                retain_graph=idx < len(score_specs) - 1,
                allow_unused=False,
            )[0][0]
            contribution = (grads * token_embeds).sum(dim=-1)
            image1_value = float(contribution[image1_positions].sum().item())
            image2_value = float(contribution[image2_positions].sum().item())
            text_value = float(contribution[text_positions].sum().item()) if text_positions else 0.0
            metrics[f"phi_image1_{suffix}"] = image1_value
            metrics[f"phi_image2_{suffix}"] = image2_value
            metrics[f"text_{suffix}"] = text_value
            metrics[f"interaction_{suffix}"] = float("nan")

    del outputs
    torch.cuda.empty_cache()
    return metrics


def main() -> int:
    args = build_parser().parse_args()
    sanitize_single_process_env()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blank_cache_dir = output_dir / "_blank_cache"

    processor, model = load_model_and_processor(args.model_path, args.device)
    input_device = resolve_input_device(model, args.device)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    dataset = build_dataset(args.dataset)
    language_model = model.model.language_model

    rope_patch = RopeAlignPatch(model, image_token_id, enabled=args.rope_align)
    rope_patch.__enter__()

    sample_summaries: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    step_score_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    run_start = time.perf_counter()

    try:
        for sample_index in args.indices:
            sample_start = time.perf_counter()
            row = dataset.data.iloc[sample_index]
            base_content = build_base_content(dataset, row)

            (
                prompt_text,
                model_inputs,
                input_ids,
                image_spans,
                text_positions,
                _,
            ) = prepare_condition_inputs(
                processor=processor,
                dataset=dataset,
                sample_index=sample_index,
                dataset_name=args.dataset,
                base_content=base_content,
                mode=args.mode,
                policy=args.policy,
                template_on_last_replay_text=args.template_on_last_replay_text,
                blank_positions=[],
                corruption_family="blank",
                blank_image_path="",
                blank_image_match_source_size=True,
                blank_image_source_position=1,
                swap_source_offset=1,
                blank_cache_dir=blank_cache_dir,
                image_token_id=image_token_id,
            )
            if len(image_spans) < 2:
                raise ValueError(f"Expected two image spans on sample {sample_index}, got {len(image_spans)}")
            model_inputs = tensor_to_device(model_inputs, input_device)
            image1_positions = list(range(image_spans[0]["start"], image_spans[0]["end"] + 1))
            image2_positions = list(range(image_spans[1]["start"], image_spans[1]["end"] + 1))

            generate_start = time.perf_counter()
            with torch.inference_mode():
                sequences = model.generate(
                    **model_inputs,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                )
            generate_seconds = time.perf_counter() - generate_start
            prompt_len = int(model_inputs["input_ids"].shape[-1])
            generated_ids = sequences[0, prompt_len:].detach().cpu().tolist()
            scored_ids = generated_ids[: max(0, min(args.max_scored_steps, len(generated_ids)))]
            generated_text = decode_generated_text(processor.tokenizer, generated_ids)
            generated_rows.append(
                {
                    "sample_index": sample_index,
                    "generated_token_count": len(generated_ids),
                    "scored_token_count": len(scored_ids),
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
                    "scored_token_count": len(scored_ids),
                    "metric_family": "gradxinput",
                    "conditions": {
                        "full": {
                            "prompt_len": len(input_ids),
                            "image_spans": image_spans,
                            "text_token_count": len(text_positions),
                        }
                    },
                }
            )

            scoring_start = time.perf_counter()
            for step, token_id in enumerate(scored_ids):
                metrics = score_step_with_gradxinput(
                    model=model,
                    language_model=language_model,
                    model_inputs=model_inputs,
                    prefix_ids=scored_ids[:step],
                    target_token_id=token_id,
                    image1_positions=image1_positions,
                    image2_positions=image2_positions,
                    text_positions=text_positions,
                    score_kind=args.score_kind,
                )
                token_text = processor.tokenizer.decode(
                    [token_id],
                    clean_up_tokenization_spaces=False,
                )
                step_score_rows.append(
                    {
                        "sample_index": sample_index,
                        "condition": "full",
                        "metric_family": "gradxinput",
                        "step": step,
                        "token_id": int(token_id),
                        "token_text": token_text,
                        "target_logprob": metrics["target_logprob"],
                        "target_logit": metrics["target_logit"],
                        "best_alt_logit": metrics["best_alt_logit"],
                        "logit_margin": metrics["logit_margin"],
                    }
                )
                row_out = {
                    "sample_index": sample_index,
                    "step": step,
                    "token_id": int(token_id),
                    "token_text": token_text,
                    "metric_family": "gradxinput",
                }
                if args.score_kind in {"logprob", "both"}:
                    row_out.update(
                        {
                            "phi_image1_logprob": metrics["phi_image1_logprob"],
                            "phi_image2_logprob": metrics["phi_image2_logprob"],
                            "interaction_logprob": metrics["interaction_logprob"],
                            "text_logprob": metrics["text_logprob"],
                        }
                    )
                if args.score_kind in {"margin", "both"}:
                    row_out.update(
                        {
                            "phi_image1_margin": metrics["phi_image1_margin"],
                            "phi_image2_margin": metrics["phi_image2_margin"],
                            "interaction_margin": metrics["interaction_margin"],
                            "text_margin": metrics["text_margin"],
                        }
                    )
                attribution_rows.append(row_out)
            scoring_seconds = time.perf_counter() - scoring_start

            sample_seconds = time.perf_counter() - sample_start
            timing_row = {
                "sample_index": sample_index,
                "generated_token_count": len(generated_ids),
                "scored_token_count": len(scored_ids),
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
            make_metric_frame(attribution_rows).to_csv(output_dir / "step_attribution.csv", index=False)
            make_metric_frame(timing_rows).to_csv(output_dir / "timing.csv", index=False)

    finally:
        rope_patch.__exit__(None, None, None)

    generated_df = make_metric_frame(generated_rows)
    step_scores_df = make_metric_frame(step_score_rows)
    attribution_df = make_metric_frame(attribution_rows)
    timing_df = make_metric_frame(timing_rows)

    generated_df.to_csv(output_dir / "generated_text.csv", index=False)
    step_scores_df.to_csv(output_dir / "step_scores.csv", index=False)
    attribution_df.to_csv(output_dir / "step_attribution.csv", index=False)
    timing_df.to_csv(output_dir / "timing.csv", index=False)

    summary = build_summary(
        args=args,
        generated_df=generated_df,
        step_scores_df=step_scores_df,
        shapley_df=attribution_df,
        timing_df=timing_df,
        run_seconds=time.perf_counter() - run_start,
    )
    summary["metric_family"] = "gradxinput"
    summary["max_scored_steps"] = int(args.max_scored_steps)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
