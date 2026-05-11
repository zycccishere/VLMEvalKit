#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ALL_ATTENTION_FUNCTIONS
except Exception:  # pragma: no cover
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vlmeval.dataset import build_dataset
from vlmeval.probe_attention import (
    Span,
    append_attention_rows,
    parse_attention_layers,
    plot_decode,
    plot_decode_ratio_by_layer,
    plot_prefill,
)
from vlmeval.vlm.qwen2_vl.replay_prompt_template import (
    PROMPT_TEMPLATE_DIRECTLY_ANSWER,
    PROMPT_TEMPLATE_IDENTITY,
    apply_prompt_template_to_content,
    strip_prompt_template_from_content_for_direct_answer,
)
from vlmeval.vlm.replay_policy import apply_replay, canonicalize_replay_mode


SINGLE_PROCESS_DISTRIBUTED_ENV_KEYS = (
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe Qwen2.5-VL image_2 usage with last-layer attention and layer-wise alignment."
    )
    parser.add_argument(
        "--model-path",
        default="/models/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--dataset", default="SEEDBench2_Plus")
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--mode", default="image_text_image")
    parser.add_argument(
        "--policy",
        default="identity",
        choices=[PROMPT_TEMPLATE_IDENTITY, PROMPT_TEMPLATE_DIRECTLY_ANSWER],
    )
    parser.add_argument("--template-on-last-replay-text", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--subspace-dim", type=int, default=8)
    parser.add_argument(
        "--attn-layers",
        default="last",
        help="Attention layers to trace: last, all, last4, or comma-separated indices.",
    )
    parser.add_argument(
        "--head-reduction",
        default="per_head",
        choices=["per_head", "mean"],
        help="Store per-head attention rows or mean-reduced rows for each sample/step.",
    )
    parser.add_argument(
        "--rope-align",
        action="store_true",
        help=(
            "Override image2 position_ids to exactly match image1's RoPE positions during prefill. "
            "This eliminates the positional-distance bias so that attention differences between "
            "image1 and image2 reflect only the contextual/information-flow bottleneck effect."
        ),
    )
    parser.add_argument("--blank-image-positions", nargs="*", type=int, default=[])
    parser.add_argument("--blank-image-path", default="")
    parser.add_argument(
        "--blank-image-match-source-size",
        action="store_true",
        help="Create a white blank image with the same width/height as a source prompt image for each sample.",
    )
    parser.add_argument(
        "--blank-image-source-position",
        type=int,
        default=1,
        help="1-based image position in the base prompt used to determine the blank image size.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    return parser


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_blank_image_env(blank_positions: list[int], blank_image_path: str) -> None:
    if blank_positions and blank_image_path:
        os.environ["REPLAY_BLANK_IMAGE_POSITIONS"] = ",".join(str(x) for x in blank_positions)
        os.environ["REPLAY_BLANK_IMAGE_PATH"] = blank_image_path
        return
    os.environ.pop("REPLAY_BLANK_IMAGE_POSITIONS", None)
    os.environ.pop("REPLAY_BLANK_IMAGE_PATH", None)


def sanitize_single_process_env() -> None:
    for key in SINGLE_PROCESS_DISTRIBUTED_ENV_KEYS:
        os.environ.pop(key, None)


def load_model_and_processor(model_path: str, device: str):
    sanitize_single_process_env()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    device_map = "auto" if str(device).strip().lower() == "auto" else device
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


def prompt_template_cfg(name: str) -> dict[str, str]:
    if name == PROMPT_TEMPLATE_DIRECTLY_ANSWER:
        return {
            "name": PROMPT_TEMPLATE_DIRECTLY_ANSWER,
            "template": "{problem}\nAnswer directly with a single word or short phrase.\nDo not output any explanation, derivation, words, or extra symbols.",
            "source": "probe",
        }
    return {
        "name": PROMPT_TEMPLATE_IDENTITY,
        "template": "{problem}",
        "source": "probe",
    }


def build_base_content(dataset, row: pd.Series) -> list[dict[str, Any]]:
    base_message = dataset.build_prompt(row)
    content = []
    for item in base_message:
        if item["type"] == "image":
            content.append({"type": "image", "image": item["value"]})
        elif item["type"] == "text":
            content.append({"type": "text", "text": item["value"]})
        else:
            raise ValueError(f"Unsupported item type: {item['type']}")
    return content


def resolve_blank_image_path_for_sample(
    *,
    base_content: list[dict[str, Any]],
    blank_positions: list[int],
    blank_image_path: str,
    blank_image_match_source_size: bool,
    blank_image_source_position: int,
    blank_cache_dir: Path,
) -> str:
    if not blank_positions:
        return ""
    if blank_image_match_source_size:
        if blank_image_source_position <= 0:
            raise ValueError("--blank-image-source-position must be >= 1.")
        image_items = [item for item in base_content if item.get("type") == "image"]
        if len(image_items) < blank_image_source_position:
            raise ValueError(
                f"Requested blank image source position {blank_image_source_position}, "
                f"but prompt only has {len(image_items)} image item(s)."
            )
        source_ref = str(image_items[blank_image_source_position - 1].get("image", "")).strip()
        if source_ref.startswith("file://"):
            source_path = Path(source_ref[len("file://") :])
        else:
            source_path = Path(source_ref)
        if not source_path.exists():
            raise FileNotFoundError(f"Blank-image source path not found: {source_path}")
        with Image.open(source_path) as source_image:
            width, height = source_image.size
        blank_cache_dir.mkdir(parents=True, exist_ok=True)
        blank_path = blank_cache_dir / f"blank_{width}x{height}.png"
        if not blank_path.exists():
            Image.new("RGB", (width, height), color=(255, 255, 255)).save(blank_path)
        return str(blank_path)
    return blank_image_path


def build_replayed_content(
    content: list[dict[str, Any]],
    dataset_name: str,
    *,
    mode: str,
    policy: str,
    template_on_last_replay_text: bool,
) -> list[dict[str, Any]]:
    cfg = prompt_template_cfg(policy)
    replay_mode = canonicalize_replay_mode(mode)

    if template_on_last_replay_text and replay_mode != "image_text":
        replay_source = content
        if policy == PROMPT_TEMPLATE_DIRECTLY_ANSWER:
            replay_source = strip_prompt_template_from_content_for_direct_answer(
                content,
                dataset=dataset_name,
                text_key="text",
            )
        replayed = apply_replay(
            replay_source,
            mode=replay_mode,
            repeat_times=1,
            image_copy_mode="reuse_path",
        )
        return apply_prompt_template_to_content(replayed, cfg, dataset=dataset_name)

    templated = apply_prompt_template_to_content(content, cfg, dataset=dataset_name)
    return apply_replay(
        templated,
        mode=replay_mode,
        repeat_times=1,
        image_copy_mode="reuse_path",
    )


def build_inputs(
    processor,
    content: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, dict[str, torch.Tensor]]:
    messages = [{"role": "user", "content": content}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    model_inputs = processor(
        text=prompt_text,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )
    return messages, prompt_text, model_inputs


def find_image_spans(input_ids: list[int], image_token_id: int) -> list[Span]:
    spans = []
    start = None
    for idx, token_id in enumerate(input_ids):
        if token_id == image_token_id and start is None:
            start = idx
        elif token_id != image_token_id and start is not None:
            spans.append(Span(name=f"image_{len(spans) + 1}", start=start, end=idx - 1))
            start = None
    if start is not None:
        spans.append(Span(name=f"image_{len(spans) + 1}", start=start, end=len(input_ids) - 1))
    return spans


def find_mid_text_positions(
    input_ids: list[int],
    image_spans: list[Span],
    special_token_ids: set[int],
) -> list[int]:
    if len(image_spans) < 1:
        return []
    text_positions = []
    if len(image_spans) >= 2:
        position_range = range(image_spans[0].end + 1, image_spans[1].start)
    else:
        position_range = range(image_spans[0].end + 1, len(input_ids))
    for pos in position_range:
        if input_ids[pos] not in special_token_ids:
            text_positions.append(pos)
    return text_positions


def tensor_to_device(inputs: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def resolve_input_device(model, requested_device: str) -> str:
    if str(requested_device).strip().lower() != "auto":
        return requested_device
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cuda:0"


def mean_pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return float((a @ b.T).mean().item())


def mean_centroid_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    a = F.normalize(a.float(), dim=-1)
    centroid = F.normalize(b.float().mean(dim=0, keepdim=True), dim=-1)
    return float((a * centroid).sum(dim=-1).mean().item())


def text_subspace_projection_ratio(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    a = a.float()
    b = b.float()
    a_centered = a - a.mean(dim=0, keepdim=True)
    b_centered = b - b.mean(dim=0, keepdim=True)
    max_rank = min(k, b_centered.shape[0], b_centered.shape[1])
    if max_rank <= 0:
        return float("nan")
    try:
        _, _, vh = torch.linalg.svd(b_centered, full_matrices=False)
    except RuntimeError:
        return float("nan")
    basis = vh[:max_rank].T
    proj = a_centered @ basis @ basis.T
    denom = torch.square(a_centered).sum()
    if denom <= 0:
        return float("nan")
    ratio = torch.square(proj).sum() / denom
    return float(ratio.item())


class QwenAttentionTracer:
    def __init__(self, attn_modules: dict[int, Any]):
        self.attn_modules = attn_modules
        self.original_forward: dict[int, Any] = {}
        self.records: list[dict[str, Any]] = []

    def patch(self) -> None:
        tracer = self
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
                _layer_idx: int = layer_idx,
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
                    raise ValueError("position_embeddings is required for traced attention.")
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

                key_states_for_scores = repeat_kv(key_states, module.num_key_value_groups)
                last_query = query_states[:, :, -1:, :]
                last_scores = torch.matmul(last_query, key_states_for_scores.transpose(2, 3)) * module.scaling
                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, -1:, : key_states_for_scores.shape[-2]]
                    last_scores = last_scores + causal_mask
                last_attn = F.softmax(last_scores, dim=-1, dtype=torch.float32)

                tracer.records.append(
                    {
                        "layer": int(_layer_idx),
                        "q_len": int(q_len),
                        "kv_len": int(key_states_for_scores.shape[-2]),
                        "attn_last": last_attn.detach().cpu(),
                    }
                )

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

    def reset(self) -> None:
        self.records.clear()


class RopeAlignPatch:
    """
    Context manager that patches model.get_rope_index so that image2 tokens
    receive exactly the same M-RoPE position indices as image1 tokens during
    the prefill pass.

    Motivation
    ----------
    In image_text_image, image2 occupies a later position in the sequence than
    image1.  Because RoPE encodes absolute position into key vectors, a decode
    query is inherently "closer" (in RoPE distance) to image2 than to image1,
    which can inflate image2 attention mass independently of content.  This
    patch eliminates that positional bias so that any remaining image2 > image1
    attention difference reflects only the information-flow-bottleneck effect
    (i.e. image2 having been contextualised by the intervening text block).

    Implementation
    --------------
    The patch intercepts model.get_rope_index at the prefill stage (seq_len > 1
    and image tokens present).  It locates the image1 and image2 spans in
    input_ids and overwrites image2's slice in position_ids with the values from
    image1's slice.  Because both images are the same image (replay), the two
    spans always have the same length, so a direct slice copy is valid.

    During decode steps (seq_len == 1, no image tokens in the new token), the
    patch is a strict no-op.  The KV cache already holds image2 keys that were
    RoPE-rotated with image1's positional encoding from the patched prefill, so
    decode attention is computed against correctly aligned cached keys.

    Caveat
    ------
    Aligning image2 positions reduces the max position_id in the prefill, which
    may shift the model's internal rope_deltas by a small constant.  This affects
    the absolute position of decode tokens but does not change the relative
    alignment between image1 and image2 in the KV cache.  The prefill analysis
    (attention mass, layer alignment) is unaffected.
    """

    def __init__(self, model, image_token_id: int, enabled: bool = True):
        self.model = model
        self.image_token_id = image_token_id
        self.enabled = enabled
        self._original = None
        self._target = None
        self._align_log: list[dict] = []

    def __enter__(self):
        if not self.enabled:
            return self
        token_id = self.image_token_id
        align_log = self._align_log

        # get_rope_index lives on Qwen2_5_VLModel (model.model) in transformers >=4.50
        # and on Qwen2_5_VLForConditionalGeneration (model) in older versions.
        if hasattr(self.model, "get_rope_index"):
            target = self.model
        elif hasattr(self.model, "model") and hasattr(self.model.model, "get_rope_index"):
            target = self.model.model
        else:
            raise AttributeError(
                "RopeAlignPatch: cannot find get_rope_index on model or model.model. "
                "Check the installed transformers version."
            )
        self._target = target
        original = target.get_rope_index
        self._original = original

        def _patched(*args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]

            position_ids, rope_deltas = original(*args, **kwargs)
            if input_ids is None:
                return position_ids, rope_deltas
            seq_len = input_ids.shape[-1]
            if seq_len > 1:
                ids = (
                    input_ids[0].tolist()
                    if input_ids.dim() == 2
                    else input_ids.tolist()
                )
                spans = find_image_spans(ids, token_id)
                if len(spans) >= 2:
                    s1, s2 = spans[0], spans[1]
                    n1, n2 = s1.length, s2.length
                    if n1 == n2:
                        pid = position_ids.clone()
                        if pid.dim() == 3:
                            # [batch, 3, seq_len]
                            pid[:, :, s2.start : s2.end + 1] = pid[
                                :, :, s1.start : s1.end + 1
                            ]
                        else:
                            # [3, seq_len]
                            pid[:, s2.start : s2.end + 1] = pid[
                                :, s1.start : s1.end + 1
                            ]
                        position_ids = pid
                        align_log.append(
                            {
                                "seq_len": seq_len,
                                "image1_span": (s1.start, s1.end),
                                "image2_span": (s2.start, s2.end),
                                "span_length": n1,
                                "aligned": True,
                            }
                        )
                    else:
                        align_log.append(
                            {
                                "seq_len": seq_len,
                                "image1_span": (s1.start, s1.end),
                                "image2_span": (s2.start, s2.end),
                                "span_length_mismatch": (n1, n2),
                                "aligned": False,
                            }
                        )
            return position_ids, rope_deltas

        self._target.get_rope_index = _patched
        return self

    def __exit__(self, *_):
        if self.enabled and self._original is not None:
            self._target.get_rope_index = self._original
            self._original = None
            self._target = None

    @property
    def align_log(self) -> list[dict]:
        return list(self._align_log)


def summarize_sample(
    *,
    sample_index: int,
    row: pd.Series,
    prompt_text: str,
    input_ids: list[int],
    image_spans: list[Span],
    text_positions: list[int],
) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "question_id": str(row.get("index", sample_index)),
        "subcategory": row.get("subcategory", None),
        "category": row.get("category", None),
        "question": row.get("question", None),
        "prompt_len": len(input_ids),
        "prompt_preview": prompt_text[:500],
        "image_spans": [span.__dict__ for span in image_spans],
        "text_token_count": len(text_positions),
    }


def make_metric_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def flush_probe_outputs(
    *,
    output_dir: Path,
    sample_summaries: list[dict[str, Any]],
    prefill_rows: list[dict[str, Any]],
    decode_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefill_df = make_metric_frame(prefill_rows)
    decode_df = make_metric_frame(decode_rows)
    alignment_df = make_metric_frame(alignment_rows)
    generated_df = make_metric_frame(generated_rows)
    timing_df = make_metric_frame(timing_rows)

    (output_dir / "sample_summary.json").write_text(
        json.dumps(sample_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    prefill_df.to_csv(output_dir / "prefill_attention.csv", index=False)
    decode_df.to_csv(output_dir / "decode_attention.csv", index=False)
    alignment_df.to_csv(output_dir / "layer_alignment.csv", index=False)
    generated_df.to_csv(output_dir / "generated_text.csv", index=False)
    timing_df.to_csv(output_dir / "timing.csv", index=False)
    return prefill_df, decode_df, alignment_df, generated_df, timing_df


def plot_alignment(alignment_df: pd.DataFrame, out_path: Path) -> None:
    if alignment_df.empty:
        return
    grouped = alignment_df.groupby("layer", as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(grouped["layer"], grouped["image1_pairwise_cos"], label="image1")
    axes[0].plot(grouped["layer"], grouped["image2_pairwise_cos"], label="image2")
    axes[0].set_title("Layer-wise Pairwise Cosine To Text")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("cosine")
    axes[0].legend()

    axes[1].plot(grouped["layer"], grouped["image1_text_subspace_ratio"], label="image1")
    axes[1].plot(grouped["layer"], grouped["image2_text_subspace_ratio"], label="image2")
    axes[1].set_title("Layer-wise Text-Subspace Projection Ratio")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("projection ratio")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)
    configure_blank_image_env(args.blank_image_positions, args.blank_image_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blank_cache_dir = output_dir / "_blank_cache"

    processor, model = load_model_and_processor(args.model_path, args.device)
    input_device = resolve_input_device(model, args.device)

    selected_layers = parse_attention_layers(args.attn_layers, len(model.model.language_model.layers))
    tracer = QwenAttentionTracer(
        {layer_idx: model.model.language_model.layers[layer_idx].self_attn for layer_idx in selected_layers}
    )
    tracer.patch()

    dataset = build_dataset(args.dataset)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    rope_patch = RopeAlignPatch(model, image_token_id, enabled=args.rope_align)
    rope_patch.__enter__()
    special_token_ids = set(processor.tokenizer.all_special_ids)

    sample_summaries = []
    prefill_rows = []
    decode_rows = []
    alignment_rows = []
    generated_rows = []
    timing_rows = []
    run_start = time.perf_counter()

    try:
        for sample_index in args.indices:

            sample_start = time.perf_counter()
            row = dataset.data.iloc[sample_index]
            base_content = build_base_content(dataset, row)
            blank_image_path = resolve_blank_image_path_for_sample(
                base_content=base_content,
                blank_positions=args.blank_image_positions,
                blank_image_path=args.blank_image_path,
                blank_image_match_source_size=args.blank_image_match_source_size,
                blank_image_source_position=args.blank_image_source_position,
                blank_cache_dir=blank_cache_dir,
            )
            configure_blank_image_env(args.blank_image_positions, blank_image_path)
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
            if len(image_spans) < 1:
                raise ValueError(
                    f"Expected at least one image span for mode={args.mode}, got {len(image_spans)} on sample {sample_index}."
                )
            text_positions = find_mid_text_positions(input_ids, image_spans, special_token_ids)
            if not text_positions:
                raise ValueError(f"Failed to recover mid-text positions on sample {sample_index}.")

            sample_summaries.append(
                summarize_sample(
                    sample_index=sample_index,
                    row=row,
                    prompt_text=prompt_text,
                    input_ids=input_ids,
                    image_spans=image_spans,
                    text_positions=text_positions,
                )
            )

            spans_for_attention = {
                "image1": list(range(image_spans[0].start, image_spans[0].end + 1)),
                "image2": (
                    list(range(image_spans[1].start, image_spans[1].end + 1))
                    if len(image_spans) >= 2
                    else []
                ),
                "text": text_positions,
            }

            model_inputs = tensor_to_device(model_inputs, input_device)

            tracer.reset()
            prefill_start = time.perf_counter()
            with torch.inference_mode():
                outputs = model(
                    **model_inputs,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            prefill_seconds = time.perf_counter() - prefill_start

            prefill_records = [record for record in tracer.records if record["q_len"] > 1]
            if not prefill_records:
                raise RuntimeError(f"No prefill attention record captured for sample {sample_index}.")

            for record in prefill_records:
                append_attention_rows(
                    sink=prefill_rows,
                    attn_last=record["attn_last"],
                    spans=spans_for_attention,
                    sample_index=sample_index,
                    stage="prefill",
                    layer=record["layer"],
                    step=-1,
                    head_reduction=args.head_reduction,
                )

            hidden_states = outputs.hidden_states
            for layer_idx, layer_hidden in enumerate(hidden_states):
                layer_hidden = layer_hidden[0].float().detach().cpu()
                text_hidden = layer_hidden[text_positions]
                image1_hidden = layer_hidden[image_spans[0].start : image_spans[0].end + 1]
                image2_hidden = (
                    layer_hidden[image_spans[1].start : image_spans[1].end + 1]
                    if len(image_spans) >= 2
                    else layer_hidden.new_empty((0, layer_hidden.shape[-1]))
                )
                alignment_rows.append(
                    {
                        "sample_index": sample_index,
                        "layer": layer_idx,
                        "image1_pairwise_cos": mean_pairwise_cosine(image1_hidden, text_hidden),
                        "image2_pairwise_cos": mean_pairwise_cosine(image2_hidden, text_hidden),
                        "image1_centroid_cos": mean_centroid_cosine(image1_hidden, text_hidden),
                        "image2_centroid_cos": mean_centroid_cosine(image2_hidden, text_hidden),
                        "image1_text_subspace_ratio": text_subspace_projection_ratio(
                            image1_hidden, text_hidden, args.subspace_dim
                        ),
                        "image2_text_subspace_ratio": text_subspace_projection_ratio(
                            image2_hidden, text_hidden, args.subspace_dim
                        ),
                    }
                )
            del outputs
            torch.cuda.empty_cache()

            tracer.reset()
            decode_start = time.perf_counter()
            with torch.inference_mode():
                sequences = model.generate(
                    **model_inputs,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                )
            decode_seconds = time.perf_counter() - decode_start

            decode_records = [record for record in tracer.records if record["q_len"] == 1]
            prompt_len = model_inputs["input_ids"].shape[-1]
            generated_ids = sequences[0, prompt_len:].detach().cpu().tolist()
            generated_text = processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            generated_rows.append(
                {
                    "sample_index": sample_index,
                    "generated_token_count": len(generated_ids),
                    "generated_text": generated_text,
                }
            )

            decode_step_by_layer: dict[int, int] = {}
            for record in decode_records:
                layer = int(record["layer"])
                step = decode_step_by_layer.get(layer, 0)
                decode_step_by_layer[layer] = step + 1
                token_id = generated_ids[step] if step < len(generated_ids) else None
                token_text = (
                    processor.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                    if token_id is not None
                    else None
                )
                append_attention_rows(
                    sink=decode_rows,
                    attn_last=record["attn_last"],
                    spans=spans_for_attention,
                    sample_index=sample_index,
                    stage="decode",
                    layer=layer,
                    step=step,
                    head_reduction=args.head_reduction,
                    token_id=token_id,
                    token_text=token_text,
                )

            del sequences
            torch.cuda.empty_cache()

            sample_seconds = time.perf_counter() - sample_start
            timing_row = {
                "sample_index": sample_index,
                "prompt_len": len(input_ids),
                "generated_token_count": len(generated_ids),
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "sample_seconds": sample_seconds,
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
            flush_probe_outputs(
                output_dir=output_dir,
                sample_summaries=sample_summaries,
                prefill_rows=prefill_rows,
                decode_rows=decode_rows,
                alignment_rows=alignment_rows,
                generated_rows=generated_rows,
                timing_rows=timing_rows,
            )

    finally:
        tracer.restore()
        rope_patch.__exit__(None, None, None)

    prefill_df, decode_df, alignment_df, generated_df, timing_df = flush_probe_outputs(
        output_dir=output_dir,
        sample_summaries=sample_summaries,
        prefill_rows=prefill_rows,
        decode_rows=decode_rows,
        alignment_rows=alignment_rows,
        generated_rows=generated_rows,
        timing_rows=timing_rows,
    )

    summary_layer = max(selected_layers)
    plot_prefill(prefill_df, output_dir / "prefill_last_token_attention.png", summary_layer=summary_layer)
    plot_decode(decode_df, output_dir / "decode_attention_over_steps.png", summary_layer=summary_layer)
    plot_decode_ratio_by_layer(decode_df, output_dir / "decode_attention_ratio_by_layer.png")
    plot_alignment(alignment_df, output_dir / "layer_alignment_over_layers.png")

    prefill_summary_df = prefill_df[prefill_df["layer"] == summary_layer] if not prefill_df.empty else prefill_df
    decode_summary_df = decode_df[decode_df["layer"] == summary_layer] if not decode_df.empty else decode_df
    prefill_mean_mass = (
        prefill_summary_df[["image1_mass", "image2_mass", "text_mass"]].mean().to_dict()
        if not prefill_summary_df.empty
        else {}
    )
    decode_mean_mass = (
        decode_summary_df[["image1_mass", "image2_mass", "text_mass"]].mean().to_dict()
        if not decode_summary_df.empty
        else {}
    )
    decode_step0 = decode_summary_df[decode_summary_df["step"] == 0]
    decode_step0_mean_mass = (
        decode_step0[["image1_mass", "image2_mass", "text_mass"]].mean().to_dict()
        if not decode_step0.empty
        else {}
    )
    final_layer = alignment_df[alignment_df["layer"] == alignment_df["layer"].max()] if not alignment_df.empty else pd.DataFrame()

    summary = {
        "dataset": args.dataset,
        "indices": args.indices,
        "mode": args.mode,
        "policy": args.policy,
        "template_on_last_replay_text": args.template_on_last_replay_text,
        "rope_align": bool(args.rope_align),
        "rope_align_log": rope_patch.align_log,
        "blank_image_positions": args.blank_image_positions,
        "blank_image_path": args.blank_image_path or None,
        "blank_image_match_source_size": bool(args.blank_image_match_source_size),
        "blank_image_source_position": int(args.blank_image_source_position),
        "max_new_tokens": args.max_new_tokens,
        "attn_layers_spec": args.attn_layers,
        "selected_attention_layers": selected_layers,
        "summary_attention_layer": summary_layer,
        "head_reduction": args.head_reduction,
        "sample_count": len(args.indices),
        "processed_sample_count": len(timing_df),
        "run_seconds": time.perf_counter() - run_start,
        "mean_sample_seconds": (
            float(timing_df["sample_seconds"].mean()) if not timing_df.empty else float("nan")
        ),
        "mean_prefill_seconds": (
            float(timing_df["prefill_seconds"].mean()) if not timing_df.empty else float("nan")
        ),
        "mean_decode_seconds": (
            float(timing_df["decode_seconds"].mean()) if not timing_df.empty else float("nan")
        ),
        "prefill_row_count": int(len(prefill_df)),
        "decode_row_count": int(len(decode_df)),
        "prefill_mean_mass": prefill_mean_mass,
        "decode_mean_mass": decode_mean_mass,
        "decode_step0_mean_mass": decode_step0_mean_mass,
        "prefill_image2_over_image1_mass_ratio": (
            float(prefill_mean_mass["image2_mass"] / max(prefill_mean_mass["image1_mass"], 1e-8))
            if prefill_mean_mass
            else float("nan")
        ),
        "decode_image2_over_image1_mass_ratio": (
            float(decode_mean_mass["image2_mass"] / max(decode_mean_mass["image1_mass"], 1e-8))
            if decode_mean_mass
            else float("nan")
        ),
        "decode_step0_image2_over_image1_mass_ratio": (
            float(decode_step0_mean_mass["image2_mass"] / max(decode_step0_mean_mass["image1_mass"], 1e-8))
            if decode_step0_mean_mass
            else float("nan")
        ),
        "final_layer_image2_pairwise_cos": (
            float(final_layer["image2_pairwise_cos"].mean()) if not final_layer.empty else float("nan")
        ),
        "final_layer_image1_pairwise_cos": (
            float(final_layer["image1_pairwise_cos"].mean()) if not final_layer.empty else float("nan")
        ),
        "final_layer_image2_text_subspace_ratio": (
            float(final_layer["image2_text_subspace_ratio"].mean()) if not final_layer.empty else float("nan")
        ),
        "final_layer_image1_text_subspace_ratio": (
            float(final_layer["image1_text_subspace_ratio"].mean()) if not final_layer.empty else float("nan")
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
