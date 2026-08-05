import hashlib
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from vlmeval.probes.readout_carriers import (
    CARRIERS,
    MASK_CONDITIONS,
    PreparedSequence,
    _attention_backend,
    _candidate_parity,
    _candidate_values,
    _carrier_masks,
    _carrier_masks_from_base,
    _validate_corruption_control,
    _independent_expected_masks,
    _independent_expected_gemma_masks,
    _independent_gemma_native_masks,
    _insert_matched_text_carrier,
    _independent_qwen_mrope,
    _insert_matched_text_ids_carrier,
    _literal_token_id,
    _lorem_carrier_token_ids,
    _minicpm_per_image_vision_states,
    _minicpm_content_to_prompt,
    _minicpm_processor_call,
    _minicpm_v_per_image_vision_states,
    _gemma_image_spans,
    _model_family,
    _natural_noise_image,
    _nested_input_summary,
    _normalize_minicpm_iqi_content,
    _prepare_literal_blind,
    _prepare_minicpm_state,
    _qwen_per_image_features,
    _run_minicpm_allowed,
    _run_qwen_allowed,
    _run_qwen_embedded_standard,
    _score_values,
    _single_edit_spec,
    _splice_ids_2d,
    _tensor_sha256,
    _validate_matched_readout_counts,
    _validate_raw_score,
)


class _LiteralTokenizer:
    all_special_ids = [999]
    unk_token_id = -1

    def __init__(self):
        self.mapping = {".": 13, " ": 220, "?": 30}
        self.inverse = {value: key for key, value in self.mapping.items()}

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[self.mapping[text]])

    def decode(self, ids, **kwargs):
        del kwargs
        return "".join(self.inverse[int(item)] for item in ids)


class ReadoutCarrierMaskTest(unittest.TestCase):
    def test_exact_visibility_for_all_three_conditions(self):
        masks, checks = _carrier_masks(10, 7, [4, 5])
        self.assertEqual(tuple(masks.shape), (3, 10, 10))
        self.assertEqual(list(MASK_CONDITIONS), ["aware", "no_write", "position_null"])
        self.assertEqual(masks[0, 9].nonzero().flatten().tolist(), [4, 5, 7, 8, 9])
        self.assertEqual(masks[1, 4].nonzero().flatten().tolist(), [4])
        self.assertEqual(masks[1, 5].nonzero().flatten().tolist(), [4, 5])
        self.assertEqual(masks[1, 9].nonzero().flatten().tolist(), [4, 5, 7, 8, 9])
        self.assertEqual(masks[2, 9].nonzero().flatten().tolist(), [7, 8, 9])
        self.assertEqual(checks["no_write_readout_to_pre_readout_visible"], 0)
        self.assertEqual(checks["position_null_decode_prefill_visible"], 0)

    def test_numpy_oracle_is_independent_and_exact(self):
        masks, _ = _carrier_masks(13, 9, [5, 6, 7])
        oracle = _independent_expected_masks(13, 9, [5, 6, 7])
        np.testing.assert_array_equal(masks.numpy(), oracle)

    def test_native_mask_constraints_preserve_visual_bidirectionality(self):
        base = torch.tril(torch.ones((8, 8), dtype=torch.bool))
        base[3:5, 3:5] = True
        masks, checks = _carrier_masks_from_base(base, 6, [3, 4])

        self.assertTrue(masks[0, 3, 4])
        self.assertTrue(masks[1, 3, 4])
        self.assertFalse(masks[1, 3, 2])
        self.assertEqual(masks[0, 7].nonzero().flatten().tolist(), [3, 4, 6, 7])
        self.assertEqual(masks[2, 7].nonzero().flatten().tolist(), [6, 7])
        self.assertEqual(checks["no_write_readout_to_non_readout_prefill_visible"], 0)

    def test_gemma_effective_masks_match_topology_across_modalities(self):
        masks = _independent_expected_gemma_masks(12, 9, [6, 7, 8], 5)
        self.assertEqual(masks.shape, (3, 2, 12, 12))
        self.assertFalse(masks[:, :, 6, 8].any())
        self.assertFalse(masks[0, 0, 11, 5])
        self.assertTrue(masks[0, 0, 11, 8])
        self.assertFalse(masks[0, 1, 11, 6])

    def test_gemma_native_oracle_matches_official_cross_span_distance_rule(self):
        token_types = np.array([[0, 1, 1, 0, 0, 1, 1, 0]])
        masks = _independent_gemma_native_masks(token_types, 4, 4)
        self.assertTrue(masks[0, 1, 5])
        self.assertFalse(masks[0, 1, 6])
        self.assertTrue(masks[0, 1, 2])
        self.assertFalse(masks[0, 1, 3])
        self.assertTrue(masks[1, 5, 1])
        self.assertFalse(masks[1, 7, 1])


class ReadoutCarrierSequenceTest(unittest.TestCase):
    def test_expanded_model_keys_use_the_intended_semantic_family(self):
        self.assertEqual(_model_family("qwen25vl_32b"), "qwen25vl")
        self.assertEqual(_model_family("minicpm_v_45"), "minicpmv45")
        self.assertEqual(_model_family("minicpm_o_45"), "minicpmo45")
        self.assertEqual(_model_family("gemma3_12b"), "gemma3")

    def test_gemma_image_spans_are_exact_contiguous_runs(self):
        ids = torch.tensor([[7, 9, 9, 9, 4, 9, 9, 3]])
        self.assertEqual(_gemma_image_spans(ids, 9), [(1, 4), (5, 7)])

    def test_gemma_text_splice_updates_only_sequence_token_types(self):
        inputs = {
            "input_ids": torch.tensor([[10, 11, 12, 13]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "token_type_ids": torch.tensor([[0, 1, 1, 0]]),
            "pixel_values": torch.zeros((1, 3, 2, 2)),
        }
        out = _splice_ids_2d(
            inputs,
            1,
            3,
            [21, 22],
            recompute_position_ids=False,
            sequence_replacements={"token_type_ids": [0, 0]},
        )
        self.assertEqual(out["input_ids"].tolist(), [[10, 21, 22, 13]])
        self.assertEqual(out["token_type_ids"].tolist(), [[0, 0, 0, 0]])
        self.assertIs(out["pixel_values"], inputs["pixel_values"])

    def test_qwen_embedded_standard_reuses_precomputed_state(self):
        calls = []

        class LanguageModel:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"] + 1)

        model = SimpleNamespace(
            model=SimpleNamespace(language_model=LanguageModel()),
            lm_head=lambda value: value.sum(dim=-1, keepdim=True),
        )
        embeds = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        positions = torch.arange(3).reshape(1, 1, 3).repeat(3, 1, 1)
        attention = torch.ones((1, 3), dtype=torch.long)
        logits = _run_qwen_embedded_standard(
            model,
            {"attention_mask": attention},
            {"inputs_embeds": embeds, "position_ids": positions},
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["inputs_embeds"], embeds)
        self.assertIs(calls[0]["position_ids"], positions)
        self.assertIs(calls[0]["attention_mask"], attention)
        self.assertIsNone(calls[0]["input_ids"])
        torch.testing.assert_close(logits, (embeds[:, -1, :] + 1).sum(dim=-1))

    def test_qwen_masks_run_as_independent_single_examples(self):
        class FakeLanguageModel:
            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                visible = float((kwargs["attention_mask"] == 0).sum())
                return SimpleNamespace(
                    last_hidden_state=kwargs["inputs_embeds"] + visible
                )

        language_model = FakeLanguageModel()
        model = SimpleNamespace(
            model=SimpleNamespace(language_model=language_model),
            lm_head=lambda value: value.sum(dim=-1, keepdim=True),
        )
        embeds = torch.zeros((1, 3, 2))
        positions = torch.arange(3).reshape(1, 1, 3).repeat(3, 1, 1)
        state = {
            "inputs_embeds": embeds,
            "position_ids": positions,
            "public_meta": {},
        }
        allowed = torch.stack(
            [
                torch.eye(3, dtype=torch.bool),
                torch.tril(torch.ones((3, 3), dtype=torch.bool)),
                torch.ones((3, 3), dtype=torch.bool),
            ]
        )
        logits, _, _ = _run_qwen_allowed(model, {}, allowed, state=state)

        self.assertEqual(len(language_model.calls), 3)
        self.assertEqual(
            [tuple(call["inputs_embeds"].shape) for call in language_model.calls],
            [(1, 3, 2)] * 3,
        )
        self.assertEqual(
            [tuple(call["attention_mask"].shape) for call in language_model.calls],
            [(1, 1, 3, 3)] * 3,
        )
        self.assertTrue(
            all(call["inputs_embeds"] is embeds for call in language_model.calls)
        )
        self.assertTrue(
            all(call["position_ids"] is positions for call in language_model.calls)
        )
        expected_masks = torch.where(
            allowed,
            torch.tensor(0.0),
            torch.tensor(torch.finfo(torch.float32).min),
        ).unsqueeze(1)
        for idx, call in enumerate(language_model.calls):
            torch.testing.assert_close(call["attention_mask"], expected_masks[idx : idx + 1])
        self.assertEqual(tuple(logits.shape), (3, 1))
        torch.testing.assert_close(logits[:, 0], torch.tensor([6.0, 12.0, 18.0]))

    def test_qwen_images_are_encoded_one_at_a_time(self):
        class FakeModel:
            def __init__(self):
                self.calls = []
                self.config = SimpleNamespace(
                    vision_config=SimpleNamespace(spatial_merge_size=2)
                )

            def get_image_features(self, pixels, grid):
                self.calls.append((pixels.clone(), grid.clone()))
                value = float(pixels[0, 0].item())
                feature_count = int(grid.prod() // 4)
                return (torch.full((feature_count, 3), value),)

        model = FakeModel()
        pixels = torch.tensor([[1.0]] * 4 + [[2.0]] * 8)
        grid = torch.tensor([[1, 2, 2], [1, 2, 4]])
        features = _qwen_per_image_features(model, pixels, grid)

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(tuple(model.calls[0][0].shape), (4, 1))
        self.assertEqual(tuple(model.calls[1][0].shape), (8, 1))
        self.assertEqual(model.calls[0][1].tolist(), [[1, 2, 2]])
        self.assertEqual(model.calls[1][1].tolist(), [[1, 2, 4]])
        self.assertEqual([tuple(part.shape) for part in features], [(1, 3), (2, 3)])
        self.assertTrue(torch.all(features[0] == 1))
        self.assertTrue(torch.all(features[1] == 2))

    def test_qwen_per_image_encoding_rejects_incomplete_grid(self):
        model = SimpleNamespace(
            config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
        )
        with self.assertRaisesRegex(RuntimeError, "exactly consume"):
            _qwen_per_image_features(
                model,
                torch.zeros((5, 1)),
                torch.tensor([[1, 2, 2]]),
            )

    def test_qwen_per_image_encoding_rejects_wrong_feature_length(self):
        model = SimpleNamespace(
            config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)),
            get_image_features=lambda pixels, grid: (torch.zeros((2, 3)),),
        )
        with self.assertRaisesRegex(RuntimeError, "feature length mismatch"):
            _qwen_per_image_features(
                model,
                torch.zeros((4, 1)),
                torch.tensor([[1, 2, 2]]),
            )

    def test_minicpm_images_are_encoded_one_at_a_time(self):
        class FakeModel:
            def __init__(self):
                self.calls = []
                self.config = SimpleNamespace(query_num=2)

            def get_vision_embedding(self, inputs):
                pixels = inputs["pixel_values"][0]
                sizes = inputs["tgt_sizes"][0]
                self.calls.append((len(pixels), tuple(sizes.shape)))
                value = float(pixels[0].item())
                return [torch.full((1, 2, 3), value)]

        model = FakeModel()
        inputs = {
            "pixel_values": [[torch.tensor(1.0), torch.tensor(2.0)]],
            "tgt_sizes": [torch.tensor([[10, 11], [20, 21]])],
            "image_bound": [torch.tensor([[5, 7], [9, 11]])],
        }
        states = _minicpm_per_image_vision_states(model, inputs)
        self.assertEqual(model.calls, [(1, (1, 2)), (1, (1, 2))])
        self.assertEqual(tuple(states[0].shape), (2, 2, 3))
        self.assertTrue(torch.all(states[0][0] == 1))
        self.assertTrue(torch.all(states[0][1] == 2))

    def test_minicpm_v_images_and_temporal_ids_are_encoded_one_at_a_time(self):
        class FakeModel:
            device = torch.device("cpu")

            def __init__(self):
                self.config = SimpleNamespace(query_num=2)
                self.calls = []

            def get_vllm_embedding(self, inputs):
                self.calls.append(
                    (
                        len(inputs["pixel_values"][0]),
                        inputs["temporal_ids"][0],
                    )
                )
                value = float(inputs["pixel_values"][0][0].item())
                states = torch.full((1, 2, 3), value)
                return torch.zeros((1, 2, 3)), [states]

        model = FakeModel()
        inputs = {
            "pixel_values": [[torch.tensor(1.0), torch.tensor(2.0)]],
            "tgt_sizes": [torch.tensor([[10, 11], [20, 21]])],
            "temporal_ids": [[[-1], [-1]]],
            "image_bound": [torch.tensor([[5, 7], [9, 11]])],
        }
        states = _minicpm_v_per_image_vision_states(model, inputs)
        self.assertEqual(model.calls, [(1, [[-1]]), (1, [[-1]])])
        self.assertEqual(tuple(states[0].shape), (2, 2, 3))
        self.assertTrue(torch.all(states[0][0] == 1))
        self.assertTrue(torch.all(states[0][1] == 2))

    def test_attention_backend_is_scoped(self):
        config = SimpleNamespace(_attn_implementation="sdpa")
        with _attention_backend(config, "eager"):
            self.assertEqual(config._attn_implementation, "eager")
        self.assertEqual(config._attn_implementation, "sdpa")

    def test_minicpm_masks_run_as_independent_single_examples(self):
        class FakeLLM:
            def __init__(self):
                self.batch_sizes = []

            def __call__(self, **kwargs):
                self.batch_sizes.append(int(kwargs["inputs_embeds"].shape[0]))
                visible = float((kwargs["attention_mask"] == 0).sum())
                return SimpleNamespace(logits=torch.tensor([[[visible, -visible]]]))

        model = SimpleNamespace(llm=FakeLLM())
        state = {
            "inputs_embeds": torch.zeros((1, 3, 2)),
            "position_ids": torch.arange(3).unsqueeze(0),
            "cache_position": torch.arange(3),
            "public_meta": {},
        }
        allowed = torch.stack(
            [
                torch.eye(3, dtype=torch.bool),
                torch.tril(torch.ones((3, 3), dtype=torch.bool)),
            ]
        )
        logits, _, _ = _run_minicpm_allowed(
            model, {}, allowed, state=state, apply_omni=False
        )
        self.assertEqual(model.llm.batch_sizes, [1, 1])
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertNotEqual(float(logits[0, 0]), float(logits[1, 0]))

    def test_minicpm_v_skips_omni_while_minicpm_o_applies_identity_path(self):
        class FakeModel:
            device = torch.device("cpu")

            def __init__(self):
                self.config = SimpleNamespace(query_num=2, audio_chunk_length=1)
                self.llm = SimpleNamespace(
                    config=SimpleNamespace(_attn_implementation="sdpa")
                )
                self.omni_calls = 0

            def get_vision_embedding(self, inputs):
                del inputs
                return [torch.ones((1, 2, 3))]

            def get_vllm_embedding(self, inputs):
                states = inputs.get("vision_hidden_states")
                if states is None:
                    value = float(inputs["pixel_values"][0][0].item())
                    states = [torch.full((1, 2, 3), value)]
                bound = inputs["image_bound"][0][0]
                embeds = torch.zeros((1, inputs["input_ids"].shape[-1], 3))
                embeds[0, int(bound[0]) : int(bound[1])] = states[0][0]
                return embeds, states

            def get_omni_embedding(self, inputs, *, input_embeddings, chunk_length):
                del inputs, chunk_length
                self.omni_calls += 1
                return input_embeddings

        inputs = {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "pixel_values": [torch.ones((1, 1))],
            "tgt_sizes": [torch.tensor([[1, 1]])],
            "image_bound": [torch.tensor([[1, 3]])],
            "position_ids": torch.arange(4).unsqueeze(0),
            "temporal_ids": [[[-1]]],
        }
        model = FakeModel()
        visual_state = _prepare_minicpm_state(model, inputs, apply_omni=False)
        self.assertEqual(model.omni_calls, 0)
        self.assertEqual(
            visual_state["public_meta"]["embedding_postprocess"], "not_applicable"
        )
        omni_state = _prepare_minicpm_state(model, inputs, apply_omni=True)
        self.assertEqual(model.omni_calls, 1)
        self.assertEqual(
            omni_state["public_meta"]["embedding_postprocess"],
            "applied_no_audio_identity",
        )

    def test_minicpm_processor_uses_model_specific_positional_inputs(self):
        class Batch(dict):
            def to(self, device):
                self["moved_to"] = device
                return self

        class Processor:
            def __init__(self):
                self.calls = []

            def __call__(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return Batch(input_ids=torch.tensor([[1]]))

        processor = Processor()
        model = SimpleNamespace(device="cuda:0")
        wrapper = SimpleNamespace(model=model, processor=processor, tokenizer=object())
        _minicpm_processor_call(wrapper, "prompt", [], family="minicpmv45")
        self.assertEqual(len(processor.calls[0][0]), 2)
        self.assertNotIn("stream_input", processor.calls[0][1])

        prepare_calls = []
        model.prepare_processor = lambda **kwargs: prepare_calls.append(kwargs)
        _minicpm_processor_call(wrapper, "prompt", [], family="minicpmo45")
        self.assertEqual(len(processor.calls[1][0]), 4)
        self.assertFalse(processor.calls[1][1]["stream_input"])
        self.assertEqual(len(prepare_calls), 1)

    def test_minicpm_prompt_uses_checkpoint_native_image_marker(self):
        calls = []

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                calls.append((messages, kwargs))
                return messages[0]["content"]

        wrapper = SimpleNamespace(
            processor=SimpleNamespace(tokenizer=Tokenizer())
        )
        image = Image.new("RGB", (2, 2))
        v_prompt, _ = _minicpm_content_to_prompt(
            wrapper,
            [image, "question"],
            family="minicpmv45",
            add_generation_prompt=True,
        )
        o_prompt, _ = _minicpm_content_to_prompt(
            wrapper,
            [image, "question"],
            family="minicpmo45",
            add_generation_prompt=True,
        )
        self.assertTrue(v_prompt.startswith("(<image>./</image>)"))
        self.assertTrue(o_prompt.startswith("<image>./</image>"))
        self.assertNotIn("use_tts_template", calls[0][1])
        self.assertFalse(calls[1][1]["use_tts_template"])

    def test_splice_recomputes_unpadded_minicpm_positions(self):
        inputs = {
            "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.bool),
            "position_ids": torch.arange(3).unsqueeze(0),
            "image_grid_thw": torch.tensor([[1, 2, 3, 4]]),
        }
        out = _splice_ids_2d(
            inputs,
            2,
            2,
            [20, 21],
            recompute_position_ids=True,
        )
        self.assertEqual(out["input_ids"].tolist(), [[10, 11, 20, 21, 12]])
        self.assertEqual(out["attention_mask"].tolist(), [[True] * 5])
        self.assertEqual(out["position_ids"].tolist(), [[0, 1, 2, 3, 4]])
        self.assertEqual(out["image_grid_thw"].tolist(), [[1, 2, 3, 4]])

    def test_dot_and_space_are_distinct_reversible_single_tokens(self):
        tokenizer = _LiteralTokenizer()
        self.assertEqual(_literal_token_id(tokenizer, "."), 13)
        self.assertEqual(_literal_token_id(tokenizer, " "), 220)

    def test_literal_blind_contains_only_forced_answer_prefix(self):
        wrapper = SimpleNamespace(model=torch.nn.Linear(2, 2, bias=False))
        tokenizer = _LiteralTokenizer()
        blind = _prepare_literal_blind(
            wrapper,
            "qwen25vl",
            tokenizer,
            [13, 220],
        )
        self.assertEqual(blind.inputs["input_ids"].tolist(), [[13, 220]])
        self.assertEqual(blind.prefill_len, 0)
        self.assertEqual(blind.token_roles, ["answer_prefix", "answer_prefix"])
        self.assertEqual(blind.generation_text, ". ")

    def test_nested_input_summary_never_materializes_tensor_values(self):
        value = [[torch.arange(12).reshape(3, 4)]]
        summary = _nested_input_summary(value)
        self.assertEqual(summary[0][0]["shape"], [3, 4])
        self.assertIn("sha256", summary[0][0])
        self.assertNotIn("value", summary[0][0])

    def test_nested_input_summary_supports_scalar_temporal_ids(self):
        summary = _nested_input_summary([[torch.tensor(-1, dtype=torch.long)]])
        self.assertEqual(summary[0][0]["shape"], [])
        self.assertEqual(summary[0][0]["dtype"], "torch.int64")
        self.assertIn("sha256", summary[0][0])

    def test_text_core_replaces_visual_core_inside_identical_envelope(self):
        base = PreparedSequence(
            inputs={
                "input_ids": torch.tensor([[10, 11, 90, 91]]),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
            },
            prefill_len=2,
            readout_indices=[],
            prompt_text="base",
            generation_text="base+assistant",
            token_roles=["prefill", "prefill", "decode_prefix", "decode_prefix"],
        )
        reference = PreparedSequence(
            inputs={
                "input_ids": torch.tensor([[10, 11, 50, 20, 20, 51, 90, 91]]),
                "attention_mask": torch.ones((1, 8), dtype=torch.long),
            },
            prefill_len=6,
            readout_indices=[3, 4],
            prompt_text="expanded",
            generation_text="expanded+assistant",
            token_roles=["prefill"] * 6 + ["decode_prefix"] * 2,
        )
        spec = _single_edit_spec(
            base.inputs["input_ids"], reference.inputs["input_ids"], 3, 5
        )
        self.assertEqual(spec["prefix_envelope_ids"], [50])
        self.assertEqual(spec["suffix_envelope_ids"], [51])
        text = _insert_matched_text_carrier(
            base,
            reference,
            core_start=3,
            core_end_exclusive=5,
            literal_token_id=13,
            carrier="dot_text",
            family="qwen25vl",
        )
        self.assertEqual(
            text.inputs["input_ids"].tolist(), [[10, 11, 50, 13, 13, 51, 90, 91]]
        )
        self.assertEqual(text.readout_indices, [3, 4])
        self.assertEqual(text.prefill_len, reference.prefill_len)
        self.assertEqual(
            text.prefill_len - text.readout_indices[-1] - 1,
            reference.prefill_len - reference.readout_indices[-1] - 1,
        )

    def test_text_core_handles_one_token_boundary_retokenization(self):
        base = PreparedSequence(
            inputs={
                "input_ids": torch.tensor([[10, 13, 90]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "position_ids": torch.arange(3).unsqueeze(0),
            },
            prefill_len=2,
            readout_indices=[],
            prompt_text="base",
            generation_text="base+assistant",
            token_roles=["prefill", "prefill", "decode_prefix"],
        )
        reference = PreparedSequence(
            inputs={
                "input_ids": torch.tensor([[10, 113, 50, 20, 51, 90]]),
                "attention_mask": torch.ones((1, 6), dtype=torch.long),
                "position_ids": torch.arange(6).unsqueeze(0),
            },
            prefill_len=5,
            readout_indices=[3],
            prompt_text="expanded",
            generation_text="expanded+assistant",
            token_roles=["prefill"] * 5 + ["decode_prefix"],
        )
        text = _insert_matched_text_carrier(
            base,
            reference,
            core_start=3,
            core_end_exclusive=4,
            literal_token_id=13,
            carrier="dot_text",
            family="minicpmo45",
        )
        self.assertEqual(text.inputs["input_ids"].tolist(), [[10, 113, 50, 13, 51, 90]])
        self.assertEqual(text.inputs["position_ids"].tolist(), [[0, 1, 2, 3, 4, 5]])
        self.assertEqual(text.prefill_len, reference.prefill_len)
        self.assertEqual(text.metadata["replaced_source_token_count"], 1)

    def test_generic_text_core_inserts_exact_token_sequence(self):
        base = PreparedSequence(
            inputs={
                "input_ids": torch.tensor([[10, 11, 90]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            },
            prefill_len=2,
            readout_indices=[],
            prompt_text="base",
            generation_text="base+assistant",
            token_roles=["prefill", "prefill", "decode_prefix"],
        )
        reference = PreparedSequence(
            inputs={
                "input_ids": torch.tensor([[10, 11, 50, 20, 20, 20, 51, 90]]),
                "attention_mask": torch.ones((1, 8), dtype=torch.long),
            },
            prefill_len=7,
            readout_indices=[3, 4, 5],
            prompt_text="expanded",
            generation_text="expanded+assistant",
            token_roles=["prefill"] * 7 + ["decode_prefix"],
        )
        text = _insert_matched_text_ids_carrier(
            base,
            reference,
            core_start=3,
            core_end_exclusive=6,
            carrier_token_ids=[71, 72, 73],
            carrier="ordered_lorem",
            family="qwen25vl",
        )
        self.assertEqual(
            text.inputs["input_ids"].tolist(), [[10, 11, 50, 71, 72, 73, 51, 90]]
        )
        self.assertEqual(text.readout_indices, [3, 4, 5])

    def test_noise_is_deterministic_grayscale_and_seed_specific(self):
        first = np.asarray(_natural_noise_image((48, 32), 17))
        repeated = np.asarray(_natural_noise_image((48, 32), 17))
        different = np.asarray(_natural_noise_image((48, 32), 29))
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, different))
        np.testing.assert_array_equal(first[:, :, 0], first[:, :, 1])
        np.testing.assert_array_equal(first[:, :, 0], first[:, :, 2])
        self.assertGreater(float(first[:, :, 0].std()), 20.0)

    def test_lorem_shuffle_preserves_exact_multiset(self):
        class LoremTokenizer:
            all_special_ids = [999]

            def __call__(self, text, add_special_tokens=False):
                del text, add_special_tokens
                return SimpleNamespace(input_ids=[10, 20, 30, 40, 50])

        ordered, _ = _lorem_carrier_token_ids(LoremTokenizer(), 64, "ordered_lorem")
        shuffles = []
        for carrier, seed in (
            ("shuffled_lorem_s0", 17),
            ("shuffled_lorem_s1", 29),
            ("shuffled_lorem_s2", 43),
        ):
            shuffled, metadata = _lorem_carrier_token_ids(LoremTokenizer(), 64, carrier)
            self.assertCountEqual(ordered, shuffled)
            self.assertNotEqual(ordered, shuffled)
            self.assertEqual(metadata["shuffle_seed"], seed)
            shuffles.append(tuple(shuffled))
        self.assertEqual(len(set(shuffles)), 3)

    def test_minicpm_adjacent_text_segments_preserve_joined_prompt(self):
        first = Image.new("RGB", (2, 2), color="white")
        second = Image.new("RGB", (2, 2), color="yellow")
        normalized = _normalize_minicpm_iqi_content(
            [first, "system text", "question text", second]
        )
        self.assertIs(normalized[0], first)
        self.assertEqual(normalized[1], "system text\nquestion text")
        self.assertIs(normalized[2], second)

    def test_structural_match_rejects_equal_core_with_different_answer_position(self):
        def sequence(length, prefill):
            return PreparedSequence(
                inputs={"input_ids": torch.arange(length).unsqueeze(0)},
                prefill_len=prefill,
                readout_indices=[2, 3],
                prompt_text="",
                generation_text="",
                token_roles=["prefill"] * length,
            )

        sequences = {carrier: sequence(10, 8) for carrier in CARRIERS}
        sequences["space_text"] = sequence(8, 6)
        with self.assertRaisesRegex(RuntimeError, "structural match"):
            _validate_matched_readout_counts(sequences)

    def test_structural_match_rejects_internal_readout_misalignment(self):
        def sequence(readout):
            return PreparedSequence(
                inputs={"input_ids": torch.arange(10).unsqueeze(0)},
                prefill_len=8,
                readout_indices=readout,
                prompt_text="",
                generation_text="",
                token_roles=["prefill"] * 10,
            )

        sequences = {carrier: sequence([2, 3]) for carrier in CARRIERS}
        sequences["dot_text"] = sequence([1, 3])
        with self.assertRaisesRegex(RuntimeError, "positions"):
            _validate_matched_readout_counts(sequences)

    def test_tensor_hash_supports_bfloat16(self):
        digest = _tensor_sha256(torch.arange(8, dtype=torch.bfloat16))
        self.assertEqual(len(digest), 64)

    def test_tensor_hash_supports_scalar_long(self):
        value = torch.tensor(-1, dtype=torch.long)
        digest = _tensor_sha256(value)
        self.assertEqual(digest, hashlib.sha256(value.numpy().tobytes()).hexdigest())

    def test_independent_qwen_mrope_reconstructs_all_three_axes(self):
        positions, delta = _independent_qwen_mrope(
            np.asarray([10, 20, 30, 30, 30, 30, 40, 50]),
            [[1, 4, 4]],
            image_token_id=30,
            vision_start_token_id=20,
            spatial_merge_size=2,
        )
        self.assertEqual(positions.shape, (3, 1, 8))
        np.testing.assert_array_equal(
            positions[:, 0, :],
            np.asarray(
                [
                    [0, 1, 2, 2, 2, 2, 4, 5],
                    [0, 1, 2, 2, 3, 3, 4, 5],
                    [0, 1, 2, 3, 2, 3, 4, 5],
                ]
            ),
        )
        np.testing.assert_array_equal(delta, np.asarray([[-2]]))


class ReadoutCarrierScoringTest(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "candidate_token_ids": {"A": 1, "B": 3, "C": 5},
        }

    def test_score_and_raw_reconstruction(self):
        logits = torch.tensor([0.0, 2.0, -1.0, 1.0, 0.5, -2.0])
        score = _score_values(_candidate_values(logits, self.plan), "A")
        self.assertEqual(score["predicted_key"], "A")
        self.assertTrue(score["hit"])
        _validate_raw_score(
            score,
            logits.numpy(),
            self.plan,
            context="unit-test",
        )

    def test_candidate_parity_checks_argmax_and_tolerance(self):
        reference = torch.tensor([0.0, 2.0, -1.0, 1.0, 0.5, -2.0])
        close = reference + 0.01
        result = _candidate_parity(reference, close, self.plan, atol=0.02)
        self.assertTrue(result["passed"])
        changed = reference.clone()
        changed[3] = 3.0
        result = _candidate_parity(reference, changed, self.plan, atol=2.0)
        self.assertFalse(result["argmax_equal"])
        self.assertFalse(result["passed"])

    def test_candidate_parity_rejects_non_candidate_vocab_drift(self):
        reference = torch.tensor([0.0, 2.0, -1.0, 1.0, 0.5, -2.0])
        changed = reference.clone()
        changed[2] += 1.0
        result = _candidate_parity(reference, changed, self.plan, atol=0.01, rtol=0)
        self.assertTrue(result["argmax_equal"])
        self.assertFalse(result["full_vocab_allclose"])
        self.assertFalse(result["passed"])

    def test_corruption_control_requires_aware_effect_and_blocked_invariance(self):
        original = np.zeros((3, 8), dtype=np.float32)
        ineffective = original.copy()
        with self.assertRaisesRegex(RuntimeError, "positive control"):
            _validate_corruption_control(
                original, ineffective, context="unit-ineffective"
            )
        valid = original.copy()
        valid[0, 0] = 1.0
        maxima = _validate_corruption_control(original, valid, context="unit-valid")
        self.assertEqual(maxima["aware"], 1.0)
        leaked = valid.copy()
        leaked[1, 1] = 0.1
        with self.assertRaisesRegex(RuntimeError, "invariance"):
            _validate_corruption_control(original, leaked, context="unit-leak")


if __name__ == "__main__":
    unittest.main()
