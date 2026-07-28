import unittest

import pandas as pd
import torch

from vlmeval.probes.readout_v2 import (
    allowed_masks,
    embedded_choice_labels,
    mask_checks,
    selected_subset_records,
)


class ReadoutV2MaskTest(unittest.TestCase):
    def test_exact_decode_visibility(self):
        masks = allowed_masks(10, 7, {"start": 4, "end": 5})
        checks = mask_checks(masks, 7, {"start": 4, "end": 5})
        self.assertTrue(checks["baseline_exact"])
        self.assertTrue(checks["readout_v2_exact"])
        self.assertEqual(masks[0, 9].nonzero().flatten().tolist(), [7, 8, 9])
        self.assertEqual(masks[1, 9].nonzero().flatten().tolist(), [4, 5, 7, 8, 9])
        self.assertEqual(masks[2, 9].nonzero().flatten().tolist(), list(range(10)))

    def test_prefill_is_ordinary_causal(self):
        masks = allowed_masks(8, 6, {"start": 3, "end": 4})
        expected = torch.tril(torch.ones((6, 8), dtype=torch.bool))
        self.assertTrue(torch.equal(masks[0, :6], expected))
        self.assertTrue(torch.equal(masks[1, :6], expected))

    def test_dynamath_embedded_choice_parser(self):
        question = "Question text\nA: first\nB: second\nC: third\nD: fourth\n"
        self.assertEqual(embedded_choice_labels(question), list("ABCD"))
        inline = "Choose one: (A) first (B) second (C) third (D) fourth"
        self.assertEqual(embedded_choice_labels(inline), list("ABCD"))

    def test_dynamath_keeps_visual_options_with_empty_text(self):
        frame = pd.DataFrame(
            [
                {
                    "index": 1,
                    "answer": "D",
                    "answer_type": "multiple choice",
                    "question": "Choose the visual answer.\nA: first\nB: second\nC: third\nD:",
                }
            ]
        )
        selected, summary = selected_subset_records("DynaMath", frame)
        self.assertEqual([row["sample_index"] for row in selected], ["1"])
        self.assertEqual(summary["selected_rows"], 1)

    def test_mmbench_uses_canonical_rows_without_requiring_complete_circular_group(self):
        frame = pd.DataFrame(
            [
                {"index": 1, "answer": "A", "A": "one", "B": "two", "C": "three", "D": "four"},
                {"index": 2, "answer": "B", "A": "same", "B": "same", "C": "three", "D": "four"},
                {
                    "index": 1_000_001,
                    "answer": "B",
                    "A": "two",
                    "B": "three",
                    "C": "four",
                    "D": "one",
                },
            ]
        )
        selected, summary = selected_subset_records("MMBench_DEV_EN_V11", frame)
        self.assertEqual([row["sample_index"] for row in selected], ["1"])
        self.assertEqual(summary["rejection_counts"]["duplicate_option_text"], 1)
        self.assertEqual(summary["rejection_counts"]["circular_noncanonical"], 1)

    def test_wemath_rejects_semantically_duplicate_options(self):
        frame = pd.DataFrame(
            [
                {
                    "index": 1,
                    "answer": "E",
                    "A": "one",
                    "B": "two",
                    "C": "three",
                    "D": "four",
                    "E": "no correct answer",
                },
                {
                    "index": 2,
                    "answer": "A",
                    "A": "Duplicate",
                    "B": " duplicate ",
                    "C": "three",
                    "D": "four",
                    "E": "no correct answer",
                },
            ]
        )
        selected, summary = selected_subset_records("WeMath", frame)
        self.assertEqual([row["sample_index"] for row in selected], ["1"])
        self.assertEqual(summary["rejection_counts"]["duplicate_option_text"], 1)


if __name__ == "__main__":
    unittest.main()
