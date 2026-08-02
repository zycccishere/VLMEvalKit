#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd


MODE_TO_CONDITION = {
    'image_text': 'iq',
    'image_text_image_text': 'iqiq',
}
MODELS = [
    'gemma3_4b',
    'gemma3_12b',
    'gemma3_27b',
    'minicpm_o_45_no_reasoning',
    'minicpm_v_45_no_reasoning',
]


def _load_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path, usecols=['index', 'prediction'])
    if path.suffix.lower() == '.tsv':
        return pd.read_csv(path, sep='\t', usecols=['index', 'prediction'])
    return pd.read_csv(path, usecols=['index', 'prediction'])


def _gemma_prediction_box(value: object):
    text = str(value).strip()
    match = re.search(r'```json\s+(.*?)\s+```', text, flags=re.DOTALL | re.IGNORECASE)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, list) or len(payload) != 1:
        return None
    payload = payload[0]
    if not isinstance(payload, dict) or 'box_2d' not in payload:
        return None
    coords = payload['box_2d']
    if not isinstance(coords, list) or len(coords) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in coords):
        return None
    if not all(0 <= item <= 1000 for item in coords):
        return None
    y1, x1, y2, x2 = coords
    if y2 < y1 or x2 < x1:
        return None
    return [x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0]


def _minicpm_prediction_box(value: object):
    match = re.search(
        r'<box>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</box>',
        str(value),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    coords = [int(match.group(index)) for index in range(1, 5)]
    if not all(0 <= item <= 1000 for item in coords):
        return None
    if coords[2] < coords[0] or coords[3] < coords[1]:
        return None
    return [item / 1000.0 for item in coords]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    if path.suffix.lower() == '.tsv':
        return pd.read_csv(path, sep='\t')
    return pd.read_csv(path)


def _single_table(root: Path, pattern: str, label: str, errors: list[str]):
    candidates = [path for path in root.glob(pattern) if path.is_file()]
    if len(candidates) != 1:
        errors.append(f'{root}: expected one {label}, found {len(candidates)}')
        return None
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-records', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    trace_counts: Counter[str] = Counter()
    trace_records: dict[tuple[str, str, str], dict] = {}
    for path in sorted(args.root.rglob('replay_raw.jsonl')):
        for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
            record = json.loads(line)
            if record.get('stage') != 'final_model_input':
                continue
            identity = record.get('task_identity') or {}
            model = str(identity.get('model_key', ''))
            condition = str(identity.get('condition', '')).lower()
            if model not in MODELS or condition not in {'iq', 'iqiq'}:
                continue
            key = f'{model}:{condition}'
            trace_counts[key] += 1
            prefix = f'{path}:{line_number}:{key}'
            canonical_index = str(identity.get('canonical_index', ''))
            trace_key = (model, condition, canonical_index)
            if not canonical_index:
                errors.append(f'{prefix}: missing canonical_index')
            elif trace_key in trace_records:
                errors.append(f'{prefix}: duplicate canonical_index={canonical_index}')
            else:
                trace_records[trace_key] = record
            expected_visuals = 1 if condition == 'iq' else 2
            if record.get('visual_input_count') != expected_visuals:
                errors.append(
                    f'{prefix}: visual_input_count={record.get("visual_input_count")}, '
                    f'expected {expected_visuals}'
                )
            logical_content = record.get('content_sequence') or []
            expected_types = ['image', 'text'] * expected_visuals
            observed_types = [item.get('type') for item in logical_content]
            if observed_types != expected_types:
                errors.append(f'{prefix}: content types={observed_types}, expected {expected_types}')
            text_hashes = [item.get('text_sha256') for item in logical_content if item.get('type') == 'text']
            if len(text_hashes) != expected_visuals or len(set(text_hashes)) != 1:
                errors.append(f'{prefix}: replay text hashes are not identical: {text_hashes}')
            visuals = record.get('visual_inputs') or []
            visual_hashes = [item.get('sha256') for item in visuals]
            source_hashes = [item.get('source_sha256') for item in visuals]
            if len(visual_hashes) != expected_visuals or len(set(visual_hashes)) != 1:
                errors.append(f'{prefix}: visual hashes are not identical: {visual_hashes}')
            if len(source_hashes) != expected_visuals or len(set(source_hashes)) != 1:
                errors.append(f'{prefix}: source image hashes are not identical: {source_hashes}')
            rendered = json.dumps(record.get('text_chat_representation'), ensure_ascii=False)
            expected_prompts = expected_visuals
            if model.startswith('gemma3'):
                for required in ('output only ```json',):
                    if rendered.count(required) != expected_prompts:
                        errors.append(f'{prefix}: Gemma prompt token {required!r} count mismatch')
                if '<ref>' in rendered:
                    errors.append(f'{prefix}: Gemma prompt unexpectedly contains <ref> tags')
            else:
                signature = 'Please provide the bounding box coordinate of the region this sentence describes:'
                if rendered.count(signature) != expected_prompts:
                    errors.append(f'{prefix}: MiniCPM native prompt count is not {expected_prompts}')
                if rendered.count('<ref>') != expected_prompts or rendered.count('</ref>') != expected_prompts:
                    errors.append(f'{prefix}: MiniCPM <ref> prompt count mismatch')
            generation = record.get('generation_config') or {}
            if generation.get('max_tokens') != 64:
                errors.append(f'{prefix}: max_tokens={generation.get("max_tokens")}, expected 64')
            try:
                temperature = float(generation.get('temperature'))
            except (TypeError, ValueError):
                temperature = math.nan
            if not math.isclose(temperature, 0.0, abs_tol=1e-9):
                errors.append(f'{prefix}: temperature={generation.get("temperature")}, expected 0')

    for model in MODELS:
        indices = {
            index for candidate_model, condition, index in trace_records
            if candidate_model == model and condition == 'iq'
        }
        for index in indices:
            iq = trace_records.get((model, 'iq', index))
            iqiq = trace_records.get((model, 'iqiq', index))
            if iq is None or iqiq is None:
                errors.append(f'{model}:{index}: IQ/IQIQ cross-pair is incomplete')
                continue
            iq_texts = [item.get('text_sha256') for item in iq.get('content_sequence', []) if item.get('type') == 'text']
            iqiq_texts = [item.get('text_sha256') for item in iqiq.get('content_sequence', []) if item.get('type') == 'text']
            if not iq_texts or iqiq_texts != iq_texts * 2:
                errors.append(f'{model}:{index}: IQ/IQIQ text hashes differ')
            iq_visual = [item.get('source_sha256') for item in iq.get('visual_inputs', [])]
            iqiq_visual = [item.get('source_sha256') for item in iqiq.get('visual_inputs', [])]
            if not iq_visual or iqiq_visual != iq_visual * 2:
                errors.append(f'{model}:{index}: IQ/IQIQ source image hashes differ')

    prediction_audit: dict[str, dict] = {}
    for model in MODELS:
        for mode, condition in MODE_TO_CONDITION.items():
            key = f'{model}:{condition}'
            expected_trace = args.expected_records
            if trace_counts[key] != expected_trace:
                errors.append(f'{key}: trace records={trace_counts[key]}, expected {expected_trace}')
            prediction_root = (
                args.root / 'default' / mode / 'baseline' / model / 'RefCOCO' / 'predictions'
            )
            candidates = [
                path for path in prediction_root.glob('*')
                if path.suffix.lower() in {'.xlsx', '.xls', '.csv', '.tsv'}
            ]
            if len(candidates) != 1:
                errors.append(f'{key}: expected one prediction file, found {len(candidates)}')
                continue
            frame = _load_predictions(candidates[0])
            if len(frame) != args.expected_records:
                errors.append(f'{key}: predictions={len(frame)}, expected {args.expected_records}')
            if frame['index'].astype(str).duplicated().any():
                errors.append(f'{key}: duplicate prediction indices')
            extractor = _gemma_prediction_box if model.startswith('gemma3') else _minicpm_prediction_box
            boxes = [extractor(value) for value in frame['prediction']]
            valid = [box is not None for box in boxes]
            if not any(valid):
                errors.append(f'{key}: no prediction follows the native protocol')

            eval_root = args.root / 'default' / mode / 'baseline' / model / 'RefCOCO' / 'eval'
            detail_path = _single_table(eval_root, '*_detail.*', 'detail table', errors)
            score_path = _single_table(eval_root, '*_acc.*', 'score table', errors)
            if detail_path is not None:
                detail = _read_table(detail_path)
                expected_mode = (
                    'gemma_0_1000_yxyx' if model.startswith('gemma3')
                    else 'minicpm_v46_0_1000_xyxy'
                )
                if len(detail) != args.expected_records:
                    errors.append(f'{key}: detail rows={len(detail)}, expected {args.expected_records}')
                if set(detail.get('coordinate_mode', [])) != {expected_mode}:
                    errors.append(f'{key}: detail coordinate_mode is not {expected_mode}')
                detail_by_index = detail.set_index(detail['index'].astype(str))
                for (_, row), expected_box in zip(frame.iterrows(), boxes):
                    if expected_box is None:
                        continue
                    detail_row = detail_by_index.loc[str(row['index'])]
                    observed = re.findall(r'[-+]?\d*\.?\d+', str(detail_row['pred_bbox']))
                    observed = [float(item) for item in observed[:4]]
                    if len(observed) != 4 or any(
                        not math.isclose(actual, expected, abs_tol=1e-6)
                        for actual, expected in zip(observed, expected_box)
                    ):
                        errors.append(
                            f'{key}:{row["index"]}: detail bbox={observed}, expected {expected_box}'
                        )
            if score_path is not None:
                score = _read_table(score_path)
                selected = score[score['Split'].astype(str) == 'RefCOCOg_test']
                if len(selected) != 1 or int(selected.iloc[0]['Samples']) != args.expected_records:
                    errors.append(f'{key}: score denominator is not {args.expected_records}')
            prediction_audit[key] = {
                'file': str(candidates[0]),
                'predictions': [str(value) for value in frame['prediction']],
                'valid_native_protocol': sum(valid),
                'records': len(frame),
                'detail_file': str(detail_path) if detail_path else None,
                'score_file': str(score_path) if score_path else None,
            }

    payload = {
        'ok': not errors,
        'root': str(args.root),
        'expected_records': args.expected_records,
        'trace_counts': dict(sorted(trace_counts.items())),
        'prediction_audit': prediction_audit,
        'errors': errors,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
