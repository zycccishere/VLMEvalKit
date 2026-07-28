import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from vlmeval.dataset.dynamath import _choice_option_examples, _multiple_choice_instruction
from vlmeval.probes.readout_v2 import (
    CONDITIONS,
    aggregate_combined,
    allowed_masks,
    checkpoint_identity,
    current_scoring_contract_sha256,
    embedded_choice_labels,
    expected_provenance,
    independent_expected_masks,
    mask_checks,
    scoring_contract_sha256_from_source,
    selected_subset_records,
    sha256_file,
    sha256_json,
    validate_prediction_payload,
    validate_reuse_artifact_lock,
    verify_checkpoint_identity_quick,
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

    def test_independent_numpy_mask_oracle_matches_explicit_visibility(self):
        expected = independent_expected_masks(10, 7, 4, 5)
        self.assertEqual(np.flatnonzero(expected[0, 9]).tolist(), [7, 8, 9])
        self.assertEqual(np.flatnonzero(expected[1, 9]).tolist(), [4, 5, 7, 8, 9])
        self.assertEqual(np.flatnonzero(expected[2, 9]).tolist(), list(range(10)))

    def test_dynamath_embedded_choice_parser(self):
        question = "Question text\nA: first\nB: second\nC: third\nD: fourth\n"
        self.assertEqual(embedded_choice_labels(question), list("ABCD"))
        inline = "Choose one: (A) first (B) second (C) third (D) fourth"
        self.assertEqual(embedded_choice_labels(inline), list("ABCD"))

    def test_dynamath_prompt_examples_preserve_abcd_and_expand_longer_sets(self):
        four = "Pick one.\nA: 1\nB: 2\nC: 3\nD: 4"
        six = "Pick one.\nA: 1\nB: 2\nC: 3\nD: 4\nE: 5\nF: 6"
        self.assertEqual(_choice_option_examples(four), "'A', 'B', 'C', or 'D'")
        self.assertEqual(
            _multiple_choice_instruction(four, directly_answer=True),
            "Answer with only the corresponding choice option, such as "
            "'A', 'B', 'C', or 'D'.",
        )
        self.assertEqual(
            _choice_option_examples(six),
            "'A', 'B', 'C', 'D', 'E', or 'F'",
        )
        self.assertEqual(
            _multiple_choice_instruction(six, directly_answer=True),
            "Answer with only the corresponding choice option, such as "
            "'A', 'B', 'C', 'D', 'E', or 'F'.",
        )

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

    def test_default_selection_profile_remains_fixed_choice(self):
        frame = pd.DataFrame(
            [
                {
                    "index": 1,
                    "answer": "C",
                    "A": "one",
                    "B": "two",
                    "C": "three",
                },
                {
                    "index": 2,
                    "answer": "E",
                    "A": "one",
                    "B": "two",
                    "C": "three",
                    "D": "four",
                    "E": "five",
                },
            ]
        )
        implicit, _ = selected_subset_records("WeMath", frame)
        explicit, _ = selected_subset_records("WeMath", frame, "fixed_choice")
        self.assertEqual(implicit, explicit)
        self.assertEqual([row["sample_index"] for row in implicit], ["2"])

    def test_dynamath_all_single_choice_accepts_two_through_six_choices(self):
        frame = pd.DataFrame(
            [
                {
                    "index": 1,
                    "answer": "B",
                    "answer_type": "multiple choice",
                    "question": "Pick one.\nA: first\nB: second",
                },
                {
                    "index": 2,
                    "answer": "F",
                    "answer_type": "multiple choice",
                    "question": "Pick one.\nA: 1\nB: 2\nC: 3\nD: 4\nE: 5\nF: 6",
                },
                {
                    "index": 3,
                    "answer": "B",
                    "answer_type": "free response",
                    "question": "Pick one.\nA: first\nB: second",
                },
                {
                    "index": 4,
                    "answer": "D",
                    "answer_type": "multiple choice",
                    "question": "Pick one.\nA: first\nB: second\nD: fourth",
                },
                {
                    "index": 5,
                    "answer": "E",
                    "answer_type": "multiple choice",
                    "question": "Pick one.\nA: first\nB: same\nC: third\nD: fourth\nE: same",
                },
                {
                    "index": 6,
                    "answer": "A",
                    "answer_type": "multiple choice",
                    "question": "Pick one.\nB: second\nA: first",
                },
                {
                    "index": 7,
                    "answer": "A",
                    "answer_type": "multiple choice",
                    "question": "Pick one.\nA: first\nA: repeated\nB: second",
                },
                {
                    "index": 8,
                    "answer": "a",
                    "answer_type": "multiple choice",
                    "question": "Pick one.\nA: first\nB: second",
                },
            ]
        )
        selected, summary = selected_subset_records(
            "DynaMath", frame, "all_single_choice"
        )
        self.assertEqual([row["sample_index"] for row in selected], ["1", "2"])
        self.assertEqual(summary["selected_choice_count_histogram"], {2: 1, 6: 1})
        self.assertEqual(summary["rejection_counts"]["not_multiple_choice"], 1)
        self.assertEqual(summary["rejection_counts"]["nonconsecutive_choice_columns"], 3)
        self.assertEqual(summary["rejection_counts"]["duplicate_option_text"], 1)
        self.assertEqual(summary["rejection_counts"]["answer_not_strict_single_label"], 1)

    def test_wemath_all_single_choice_filters_invalid_rows(self):
        rows = []
        for index, count in enumerate((3, 4, 6, 7), start=1):
            row = {"index": index, "answer": chr(ord("A") + count - 1)}
            row.update(
                {
                    chr(ord("A") + offset): f"option-{index}-{offset}"
                    for offset in range(count)
                }
            )
            rows.append(row)
        rows.extend(
            [
                {"index": 5, "answer": "A", "A": "only"},
                {"index": 6, "answer": "D", "A": "one", "B": "two", "D": "four"},
                {
                    "index": 7,
                    "answer": "A",
                    "A": "same",
                    "B": " SAME ",
                    "C": "three",
                },
                {
                    "index": 8,
                    "answer": "Answer: A",
                    "A": "one",
                    "B": "two",
                },
                {"index": 9, "answer": "A", "A": "None", "B": "two"},
            ]
        )
        selected, summary = selected_subset_records(
            "WeMath", pd.DataFrame(rows), "all_single_choice"
        )
        self.assertEqual(
            [row["sample_index"] for row in selected],
            ["1", "2", "3", "4", "9"],
        )
        self.assertEqual(
            summary["selected_choice_count_histogram"],
            {2: 1, 3: 1, 4: 1, 6: 1, 7: 1},
        )
        self.assertEqual(summary["rejection_counts"]["fewer_than_2_choices"], 1)
        self.assertEqual(summary["rejection_counts"]["nonconsecutive_choice_columns"], 1)
        self.assertEqual(summary["rejection_counts"]["duplicate_option_text"], 1)
        self.assertEqual(summary["rejection_counts"]["answer_not_strict_single_label"], 1)

    def test_seedbench2_plus_all_single_choice_accepts_three_and_four_choices(self):
        frame = pd.DataFrame(
            [
                {"index": 1, "answer": "C", "A": "one", "B": "two", "C": "three"},
                {
                    "index": 2,
                    "answer": "D",
                    "A": "one",
                    "B": "two",
                    "C": "three",
                    "D": "four",
                },
                {
                    "index": 3,
                    "answer": "A",
                    "A": "duplicate",
                    "B": "duplicate",
                    "C": "three",
                    "D": "four",
                },
            ]
        )
        selected, summary = selected_subset_records(
            "SEEDBench2_Plus", frame, "all_single_choice"
        )
        self.assertEqual([row["sample_index"] for row in selected], ["1", "2"])
        self.assertEqual(summary["selected_choice_count_histogram"], {3: 1, 4: 1})
        self.assertEqual(summary["rejection_counts"]["duplicate_option_text"], 1)

    def test_scoring_contract_matches_accepted_readout_v2_commit(self):
        repo_root = Path(__file__).resolve().parents[1]
        accepted_ref = next(
            (
                revision
                for revision in ("4a25a47d", "3410729bb")
                if subprocess.run(
                    ["git", "-C", str(repo_root), "cat-file", "-e", revision],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            ),
            None,
        )
        self.assertIsNotNone(accepted_ref)
        old_source = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "show",
                f"{accepted_ref}:vlmeval/probes/readout_v2.py",
            ],
            text=True,
        )
        self.assertEqual(
            scoring_contract_sha256_from_source(old_source),
            current_scoring_contract_sha256(),
        )

    def test_checkpoint_identity_rejects_wrong_model_path_and_changed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "config.json").write_text('{"model": 1}', encoding="utf-8")
            (first / "model.safetensors").write_bytes(b"weights-one")
            (second / "config.json").write_text('{"model": 1}', encoding="utf-8")
            (second / "model.safetensors").write_bytes(b"weights-one")
            identity = checkpoint_identity(str(first))
            verify_checkpoint_identity_quick(str(first), identity)
            with self.assertRaisesRegex(RuntimeError, "Model path changed"):
                verify_checkpoint_identity_quick(str(second), identity)
            (first / "config.json").write_text('{"model": 2}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "metadata file changed"):
                verify_checkpoint_identity_quick(str(first), identity)

    def test_prediction_payload_rejects_tampered_hit(self):
        manifest_record = {
            "dataset": "WeMath",
            "row_position": 0,
            "sample_index": "1",
            "answer_key": "A",
            "choice_labels": ["A", "B"],
            "shard": 0,
        }
        mask_checks_payload = {
            "prefill_causal_baseline": True,
            "prefill_causal_readout_v2": True,
            "baseline_exact": True,
            "readout_v2_exact": True,
            "full_exact": True,
            "no_future_baseline": True,
            "no_future_readout_v2": True,
            "no_future_full": True,
        }
        conditions = {
            condition: {
                "candidate_logprobs": {"A": 0.0, "B": -1.0},
                "predicted_key": "A",
                "answer_key": "A",
                "hit": condition != "baseline",
            }
            for condition in CONDITIONS
        }
        row = {
            "schema": "topic-image-replay/readout-v2-record/v1",
            "dataset": "WeMath",
            "row_position": 0,
            "sample_index": "1",
            "answer_key": "A",
            "choice_labels": ["A", "B"],
            "conditions": conditions,
            "mask_checks": mask_checks_payload,
            "runtime": {"shard_rank": 0},
            "provenance": {"lock": "value"},
        }
        with self.assertRaisesRegex(RuntimeError, "Hit is inconsistent"):
            validate_prediction_payload(
                row,
                manifest_record,
                {"lock": "value"},
                expected_shard_rank=0,
            )

    def test_reuse_artifact_lock_rejects_prediction_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest = {
                "implementation_sha256": "implementation",
                "records_sha256": "records",
                "repo_snapshot": {"head": "head"},
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            shard_path = root / "shard0.jsonl"
            shard_path.write_text('{"prediction": 1}\n', encoding="utf-8")
            lock_path = root / "lock.json"
            lock = {
                "provenance": {
                    "manifest_file_sha256": sha256_file(manifest_path),
                    "prediction_shard_sha256": {
                        "shard0.jsonl": sha256_file(shard_path)
                    },
                    "implementation_sha256": "implementation",
                    "records_sha256": "records",
                    "repo_head": "head",
                }
            }
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            lock_sha256 = sha256_file(lock_path)
            validate_reuse_artifact_lock(
                manifest_path,
                manifest,
                [shard_path],
                lock_path,
                lock_sha256,
            )
            shard_path.write_text('{"prediction": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "prediction shard hashes"):
                validate_reuse_artifact_lock(
                    manifest_path,
                    manifest,
                    [shard_path],
                    lock_path,
                    lock_sha256,
                )

    def test_combined_aggregate_preserves_reuse_missing_partition(self):
        def manifest_record(sample_index, answer):
            return {
                "dataset": "WeMath",
                "row_position": int(sample_index) - 1,
                "sample_index": str(sample_index),
                "answer_key": answer,
                "choice_labels": ["A", "B"],
                "choice_count": 2,
                "option_text_sha256": f"options-{sample_index}",
                "circular_group_size": None,
                "dataset_position": int(sample_index) - 1,
                "shard": 0,
            }

        def base_manifest(records, implementation):
            return {
                "records": records,
                "records_sha256": sha256_json(records),
                "implementation_sha256": implementation,
                "matrix_config_sha256": "matrix",
                "repo_snapshot": {"head": implementation},
                "conditions": list(CONDITIONS),
                "datasets": [
                    {
                        "dataset": "WeMath",
                        "subset": "all_single_choice",
                        "selected_rows": len(records),
                    }
                ],
            }

        def prediction(manifest, record, hits):
            conditions = {}
            for condition, hit in zip(CONDITIONS, hits):
                answer = record["answer_key"]
                other = next(label for label in record["choice_labels"] if label != answer)
                predicted = answer if hit else other
                conditions[condition] = {
                    "candidate_logprobs": {
                        label: (0.0 if label == predicted else -1.0)
                        for label in record["choice_labels"]
                    },
                    "predicted_key": predicted,
                    "answer_key": answer,
                    "hit": hit,
                }
            return {
                "schema": "topic-image-replay/readout-v2-record/v1",
                "dataset": record["dataset"],
                "row_position": record["row_position"],
                "sample_index": record["sample_index"],
                "answer_key": record["answer_key"],
                "choice_labels": record["choice_labels"],
                "conditions": conditions,
                "mask_checks": {
                    "prefill_causal_baseline": True,
                    "prefill_causal_readout_v2": True,
                    "baseline_exact": True,
                    "readout_v2_exact": True,
                    "full_exact": True,
                    "no_future_baseline": True,
                    "no_future_readout_v2": True,
                    "no_future_full": True,
                },
                "runtime": {"shard_rank": 0},
                "provenance": expected_provenance(manifest),
            }

        reused_record = manifest_record("1", "A")
        missing_record = manifest_record("2", "B")
        all_manifest = base_manifest([reused_record, missing_record], "all-head")
        reuse_manifest = base_manifest([reused_record], "reuse-head")
        missing_manifest = base_manifest([missing_record], "all-head")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                "all": root / "all.json",
                "reuse": root / "reuse.json",
                "missing": root / "missing.json",
            }
            for name, payload in (("all", all_manifest), ("reuse", reuse_manifest)):
                paths[name].write_text(json.dumps(payload), encoding="utf-8")
            reuse_root = root / "reuse_predictions"
            missing_root = root / "missing_predictions"
            reuse_root.mkdir()
            missing_root.mkdir()
            (reuse_root / "shard0.jsonl").write_text(
                json.dumps(prediction(reuse_manifest, reused_record, (True, False, True)))
                + "\n",
                encoding="utf-8",
            )
            missing_prediction_path = missing_root / "shard0.jsonl"
            missing_prediction_path.write_text(
                json.dumps(prediction(missing_manifest, missing_record, (False, True, True)))
                + "\n",
                encoding="utf-8",
            )
            lock_path = root / "accepted-lock.json"
            lock_path.write_text('{"locked": true}\n', encoding="utf-8")
            missing_manifest.update(
                {
                    "parent_all_records_sha256": all_manifest["records_sha256"],
                    "parent_all_manifest_sha256": sha256_file(paths["all"]),
                    "reuse": {
                        "source_manifest_sha256": sha256_file(paths["reuse"]),
                        "source_prediction_files": {
                            "shard0.jsonl": sha256_file(reuse_root / "shard0.jsonl")
                        },
                        "artifact_lock": {
                            "path": str(lock_path),
                            "sha256": sha256_file(lock_path),
                        },
                        "reused_record_count": 1,
                        "reused_keys_sha256": sha256_json([["WeMath", "1"]]),
                    },
                }
            )
            paths["missing"].write_text(json.dumps(missing_manifest), encoding="utf-8")
            result = aggregate_combined(
                args := SimpleNamespace(
                    all_manifest=str(paths["all"]),
                    reuse_manifest=str(paths["reuse"]),
                    missing_manifest=str(paths["missing"]),
                    reuse_input_root=str(reuse_root),
                    reuse_glob="shard*.jsonl",
                    reuse_lock=str(lock_path),
                    missing_input_root=str(missing_root),
                    missing_glob="shard*.jsonl",
                    output_root=str(root / "combined"),
                    require_complete=True,
                )
            )
            semantic_tamper = json.loads(json.dumps(missing_manifest))
            semantic_tamper["records"][0]["option_text_sha256"] = "tampered-options"
            semantic_tamper["records_sha256"] = sha256_json(semantic_tamper["records"])
            paths["missing"].write_text(json.dumps(semantic_tamper), encoding="utf-8")
            missing_prediction_path.write_text(
                json.dumps(
                    prediction(
                        semantic_tamper,
                        semantic_tamper["records"][0],
                        (False, True, True),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "Semantic mismatch"):
                aggregate_combined(args)

            overlap_tamper = json.loads(json.dumps(missing_manifest))
            overlap_tamper["records"] = [reused_record]
            overlap_tamper["records_sha256"] = sha256_json(overlap_tamper["records"])
            paths["missing"].write_text(json.dumps(overlap_tamper), encoding="utf-8")
            missing_prediction_path.write_text(
                json.dumps(
                    prediction(
                        overlap_tamper,
                        overlap_tamper["records"][0],
                        (True, False, True),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "exact complement"):
                aggregate_combined(args)
        self.assertTrue(result["complete"])
        self.assertEqual(result["expected_records"], 2)
        self.assertEqual(result["reused_records"], 1)
        self.assertEqual(result["new_records"], 1)
        self.assertEqual(result["datasets"][0]["baseline_acc"], 0.5)
        self.assertEqual(result["datasets"][0]["readout_v2_acc"], 0.5)
        self.assertEqual(result["datasets"][0]["full_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
