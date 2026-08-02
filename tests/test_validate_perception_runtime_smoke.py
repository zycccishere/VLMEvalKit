import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_perception_runtime_smoke.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("perception_runtime_validator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PerceptionRuntimeSmokeValidatorTest(unittest.TestCase):
    def test_bbox_classifier_separates_normalized_and_out_of_contract(self):
        validator = _load_validator()
        self.assertEqual(
            validator._classify_normalized_bbox("[0.1, 0.2, 0.8, 0.9]"),
            "valid_normalized",
        )
        self.assertEqual(
            validator._classify_normalized_bbox("[100, 200, 800, 900]"),
            "out_of_range_or_reversed",
        )
        self.assertEqual(
            validator._classify_normalized_bbox("[0.8, 0.2, 0.1, 0.9]"),
            "out_of_range_or_reversed",
        )
        self.assertEqual(
            validator._classify_normalized_bbox("bbox: [0.1, 0.2, 0.8, 0.9]"),
            "invalid_syntax",
        )

    def test_refcoco_audit_reads_the_task_prediction_file(self):
        validator = _load_validator()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_root = (
                root / "default" / "image_text" / "baseline" / "model_a"
                / "RefCOCO" / "predictions"
            )
            prediction_root.mkdir(parents=True)
            pd.DataFrame({
                "index": ["RefCOCOg_test_1", "RefCOCOg_test_2"],
                "prediction": ["[0.1, 0.2, 0.8, 0.9]", "[100, 200, 800, 900]"],
            }).to_csv(prediction_root / "predictions.csv", index=False)
            errors = []
            audit = validator._audit_refcoco_predictions(
                root,
                ["RefCOCO:model_a:iq"],
                2,
                errors,
            )

        self.assertEqual(errors, [])
        self.assertEqual(audit["RefCOCO:model_a:iq"]["valid_normalized"], 1)
        self.assertEqual(audit["RefCOCO:model_a:iq"]["out_of_range_or_reversed"], 1)


if __name__ == "__main__":
    unittest.main()
