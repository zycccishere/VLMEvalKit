import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rescore_qwen_refcoco_native.py"
SPEC = importlib.util.spec_from_file_location("qwen_native_rescore", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

VALIDATOR_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_qwen_refcoco_native_rescore.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location("qwen_native_validator", VALIDATOR_SCRIPT)
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
assert VALIDATOR_SPEC.loader is not None
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)


def test_smart_resize_matches_real_qwen_dump():
    assert MODULE.smart_resize(376, 640) == (784, 1316)


def test_dimension_records_use_real_image_geometry(tmp_path):
    path = tmp_path / "dimensions.tsv"
    path.write_text(
        "RefCOCOg_test_1\t/data/one.jpg\t1234\t640x376\n"
        "RefCOCOg_test_2\t/data/two.jpg\t5678\t628x442\n",
        encoding="utf-8",
    )
    assert MODULE.load_dimension_records(path) == {
        "RefCOCOg_test_1": (640, 376),
        "RefCOCOg_test_2": (628, 442),
    }


def test_smart_resize_contract_invariants():
    for height, width in [(376, 640), (640, 480), (1024, 1024), (3000, 4000)]:
        resized_height, resized_width = MODULE.smart_resize(height, width)
        assert resized_height % 28 == 0
        assert resized_width % 28 == 0
        assert MODULE.DEFAULT_MIN_PIXELS <= resized_height * resized_width <= MODULE.DEFAULT_MAX_PIXELS


def test_parse_complete_qwen_formats():
    cases = [
        ("[769, 144, 1050, 552]", [769, 144, 1050, 552]),
        ('{"bbox_2d": [769, 144, 1050, 552], "label": "person"}', [769, 144, 1050, 552]),
        ('[{"bbox_2d": [769, 144, 1050, 552], "label": "person"}]', [769, 144, 1050, 552]),
        ("```json\n[[769, 144, 1050, 552]]\n```", [769, 144, 1050, 552]),
    ]
    for text, expected in cases:
        coords, method = MODULE.parse_qwen_box(text)
        assert method == "json_single_box"
        assert coords.tolist() == expected


def test_parse_explicit_truncated_bbox_2d_only():
    coords, method = MODULE.parse_qwen_box('```json\n[{"bbox_2d": [769, 144, 1050, 552')
    assert method == "bbox_2d_open_array_recovery"
    assert coords.tolist() == [769, 144, 1050, 552]
    assert MODULE.is_complete_box_parse(method) is False


def test_parse_box_complete_outer_json_truncation():
    coords, method = MODULE.parse_qwen_box('[{"bbox_2d": [769, 144, 1050, 552], "label":')
    assert method == "bbox_2d_box_complete_recovery"
    assert coords.tolist() == [769, 144, 1050, 552]
    assert MODULE.is_complete_box_parse(method) is True


def test_parser_rejects_ambiguous_or_unkeyed_output():
    rejected = [
        "There are none.",
        "The box is [769, 144, 1050, 552].",
        "[1, 2, 3, 4, 5, 6]",
        "[[1, 2, 3, 4], [5, 6, 7, 8]]",
        '[{"bbox_2d": [1, 2, 3, 4]}, {"bbox_2d": [5, 6, 7, 8]}]',
        '{"bbox_2d": [1, 2, 3, 4], "other": [5, 6, 7, 8]}',
        "[476,582,522,677], [635,396,688,53",
    ]
    for text in rejected:
        coords, method = MODULE.parse_qwen_box(text)
        assert coords is None
        assert method == "unparsed"


def test_native_absolute_coordinates_recover_real_smoke_iou():
    gt = np.asarray([0.584859, 0.173032, 0.797422, 0.710106])
    scored = MODULE.score_prediction(
        "[769, 144, 1050, 552]",
        gt,
        source_height=376,
        source_width=640,
    )
    assert scored["coordinate_space"] == "processed_image_absolute"
    assert scored["resized_width"] == 1316
    assert scored["resized_height"] == 784
    assert scored["normalized_coords"] == [769 / 1316, 144 / 784, 1050 / 1316, 552 / 784]
    assert scored["iou"] > 0.96
    assert scored["hit"] is True


def test_prompt_compliant_normalized_coordinates_are_preserved():
    gt = np.asarray([0.1, 0.2, 0.8, 0.9])
    scored = MODULE.score_prediction(
        "[0.1, 0.2, 0.8, 0.9]",
        gt,
        source_height=480,
        source_width=640,
    )
    assert scored["coordinate_space"] == "normalized_0_1"
    assert scored["normalized_coords"] == gt.tolist()
    assert scored["iou"] == 1.0


def test_integer_valued_absolute_and_mixed_protocol_are_distinguished():
    gt = np.asarray([0.1, 0.2, 0.8, 0.9])
    absolute = MODULE.score_prediction(
        "[100.0, 200.0, 800.0, 900.0]",
        gt,
        source_height=480,
        source_width=640,
    )
    assert absolute["coordinate_space"] == "processed_image_absolute"
    assert absolute["protocol_valid"] is True

    for prediction in ("[0.28, 0.06, 1.01, 0.91]", "[0.02, 254, 582, 1162]"):
        mixed = MODULE.score_prediction(
            prediction,
            gt,
            source_height=480,
            source_width=640,
        )
        assert mixed["coordinate_space"] == "ambiguous_mixed_protocol"
        assert mixed["parsed"] is True
        assert mixed["protocol_valid"] is False
        assert mixed["normalized_coords"] is None
        assert mixed["iou"] == 0.0


def test_reversed_coordinates_are_not_reordered():
    gt = np.asarray([0.1, 0.2, 0.8, 0.9])
    scored = MODULE.score_prediction(
        "[0.8, 0.2, 0.1, 0.9]",
        gt,
        source_height=480,
        source_width=640,
    )
    assert scored["ordered"] is False
    assert scored["iou"] == 0.0
    assert scored["hit"] is False


def test_out_of_bounds_coordinates_are_not_hits():
    gt = np.asarray([0.0, 0.2, 0.8, 0.9])
    scored = MODULE.score_prediction(
        "[-10, 200, 800, 900]",
        gt,
        source_height=480,
        source_width=640,
    )
    assert scored["coordinate_space"] == "processed_image_absolute"
    assert scored["protocol_valid"] is True
    assert scored["ordered"] is True
    assert scored["in_bounds"] is False
    assert scored["iou"] == 0.0
    assert scored["hit"] is False


def test_validator_rejects_detail_prediction_tamper():
    source_row = {
        "index": "RefCOCOg_test_1",
        "prediction": "[769, 144, 1050, 552]",
        "bbox_x1_norm": 0.584859,
        "bbox_y1_norm": 0.173032,
        "bbox_x2_norm": 0.797422,
        "bbox_y2_norm": 0.710106,
    }
    detail = VALIDATOR.independently_recompute_detail(source_row, 640, 376)
    _, errors = VALIDATOR.validate_detail_against_source(detail, source_row, 640, 376)
    assert errors == []
    detail["prediction"] = "THIS IS NOT A BOX"
    _, errors = VALIDATOR.validate_detail_against_source(detail, source_row, 640, 376)
    assert "prediction mismatch" in errors
