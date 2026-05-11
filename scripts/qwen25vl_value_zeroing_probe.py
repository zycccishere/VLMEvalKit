#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from qwen25vl_faithful_attribution_probe import (
    CONDITIONS,
    build_base_content,
    build_summary,
    compute_two_player_shapley,
    decode_generated_text,
    extend_teacher_forced_inputs,
    make_metric_frame,
    prepare_condition_inputs,
)
from qwen25vl_image2_probe import (
    ALL_ATTENTION_FUNCTIONS,
    RopeAlignPatch,
    apply_multimodal_rotary_pos_emb,
    eager_attention_forward,
    load_model_and_processor,
    repeat_kv,
    resolve_input_device,
    sanitize_single_process_env,
    set_seed,
    tensor_to_device,
)
from vlmeval.dataset import build_dataset
from vlmeval.probe_attention import parse_attention_layers
from vlmeval.vlm.qwen2_vl.replay_prompt_template import (
    PROMPT_TEMPLATE_DIRECTLY_ANSWER,
    PROMPT_TEMPLATE_IDENTITY,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Value-zeroing attribution probe for Qwen2.5-VL image_text_image replay."
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
        "--zero-layers",
        default="all",
        help="Layers whose value vectors are zeroed: all, last4, last, or comma-separated indices.",
    )
    parser.add_argument("--rope-align", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    return parser


class ValueZeroingPatch:
    def __init__(self, attn_modules: dict[int, Any], zero_positions: list[int]):
        self.attn_modules = attn_modules
        self.zero_positions = sorted(set(int(x) for x in zero_positions))
        self.original_forward: dict[int, Any] = {}

    def patch(self) -> None:
        zero_positions = self.zero_positions
        for layer_idx, attn_module in self.attn_modules.items():
            self.original_forward[layer_idx] = attn_module.forward

            def instrumented_forward(
                module,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None,
                past_key_value=None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: torch.LongTensor | None = None,
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()

                query_states = module.q_proj(hidden_states)
                key_states = module.k_proj(hidden_states)
                value_states = module.v_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)

                if position_embeddings is None:
                    raise ValueError("position_embeddings is required for value-zeroing attention.")
                cos, sin = position_embeddings
                query_states, key_states = apply_multimodal_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    module.rope_scaling["mrope_section"],
                )

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, module.layer_idx, cache_kwargs
                    )

                if zero_positions:
                    valid_positions = [pos for pos in zero_positions if 0 <= pos < value_states.shape[2]]
                    if valid_positions:
                        value_states = value_states.clone()
                        value_states[:, :, valid_positions, :] = 0.0

                attention_interface = eager_attention_forward
                if module.config._attn_implementation != "eager":
                    attention_interface = ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not module.training else module.attention_dropout,
                    scaling=module.scaling,
                    sliding_window=module.sliding_window,
                    **kwargs,
                )

                attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
                attn_output = module.o_proj(attn_output)
                if not output_attentions:
                    attn_weights = None
                return attn_output, attn_weights, past_key_value

            attn_module.forward = types.MethodType(instrumented_forward, attn_module)

    def restore(self) -> None:
        for layer_idx, attn_module in self.attn_modules.items():
            attn_module.forward = self.original_forward[layer_idx]

    def __enter__(self):
        self.patch()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.restore()


def score_sequence_with_value_zeroing(
    *,
    model,
    attn_modules: dict[int, Any],
    zero_positions: list[int],
    model_inputs: dict[str, Any],
    prompt_len: int,
    generated_ids: list[int],
) -> list[dict[str, Any]]:
    if not generated_ids:
        return []
    full_inputs = extend_teacher_forced_inputs(model_inputs, generated_ids)
    with ValueZeroingPatch(attn_modules, zero_positions), torch.inference_mode():
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

    selected_layers = parse_attention_layers(args.zero_layers, len(model.model.language_model.layers))
    attn_modules = {
        layer_idx: model.model.language_model.layers[layer_idx].self_attn for layer_idx in selected_layers
    }

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
                    blank_positions=blank_positions,
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
                spans = {
                    "image1": list(range(image_spans[0]["start"], image_spans[0]["end"] + 1)),
                    "image2": list(range(image_spans[1]["start"], image_spans[1]["end"] + 1)),
                }
                zero_positions: list[int] = []
                if 1 in blank_positions:
                    zero_positions.extend(spans["image1"])
                if 2 in blank_positions:
                    zero_positions.extend(spans["image2"])
                condition_payloads[condition] = {
                    "prompt_text": prompt_text,
                    "model_inputs": tensor_to_device(model_inputs, input_device),
                    "input_ids": input_ids,
                    "prompt_len": len(input_ids),
                    "image_spans": image_spans,
                    "text_positions": text_positions,
                    "zero_positions": sorted(set(zero_positions)),
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
                    "metric_family": "value_zeroing_all_layers" if args.zero_layers == "all" else "value_zeroing",
                    "zero_layers": selected_layers,
                    "conditions": {
                        name: {
                            "prompt_len": payload["prompt_len"],
                            "image_spans": payload["image_spans"],
                            "text_token_count": len(payload["text_positions"]),
                            "zero_positions": payload["zero_positions"],
                        }
                        for name, payload in condition_payloads.items()
                    },
                }
            )

            scoring_start = time.perf_counter()
            score_table_by_condition: dict[str, list[dict[str, Any]]] = {}
            for condition, payload in condition_payloads.items():
                score_rows = score_sequence_with_value_zeroing(
                    model=model,
                    attn_modules=attn_modules,
                    zero_positions=payload["zero_positions"],
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
                            "metric_family": "value_zeroing",
                            "zero_layers": args.zero_layers,
                            "zero_positions": ",".join(str(x) for x in payload["zero_positions"]),
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
                    "metric_family": "value_zeroing",
                    "zero_layers": args.zero_layers,
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
    summary["metric_family"] = "value_zeroing"
    summary["zero_layers"] = selected_layers
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
