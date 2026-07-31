#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from vlmeval.dataset import build_dataset
from vlmeval.dataset.countqa import parse_countqa_integer


def _single_text(message):
    texts = [part['value'] for part in message if part.get('type') == 'text']
    if len(texts) != 1:
        raise AssertionError(f'Expected one text part, found {len(texts)}.')
    return texts[0]


def _message_types(message):
    return [part.get('type') for part in message]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    countqa = build_dataset('CountQA')
    spatialmqa = build_dataset('SpatialMQA')
    refcoco = build_dataset('RefCOCO')

    count_row = countqa.data.iloc[0]
    count_message = countqa.build_prompt(count_row)
    count_text = _single_text(count_message)

    spatial_row = spatialmqa.data.iloc[0]
    spatial_message = spatialmqa.build_prompt(spatial_row)
    spatial_text = _single_text(spatial_message)

    ref_indices = {'RefCOCOg_test_42959', 'RefCOCOg_test_42960'}
    ref_rows = refcoco.data[refcoco.data['index'].isin(ref_indices)]
    refcocog_test_rows = int((refcoco.data['split'] == 'RefCOCOg_test').sum())
    if len(ref_rows) != len(ref_indices):
        raise AssertionError(f'RefCOCOg smoke rows missing: {sorted(ref_indices)}')
    ref_row = ref_rows.iloc[0]
    ref_message = refcoco.build_prompt(ref_row)
    ref_text = _single_text(ref_message)
    pred_norm = refcoco._parse_prediction(str(ref_row['answer']))
    pred_abs = refcoco._to_absolute(
        pred_norm,
        float(ref_row['width']),
        float(ref_row['height']),
    )
    ref_iou = float(refcoco._compute_iou(pred_abs, refcoco._extract_gt_bbox(ref_row)))

    count_parser_sentinels = {
        '2': parse_countqa_integer('2'),
        'There are 2 objects.': parse_countqa_integer('There are 2 objects.'),
        'two': parse_countqa_integer('two'),
        '<think>1</think><answer>3</answer>': parse_countqa_integer(
            '<think>1</think><answer>3</answer>'
        ),
    }
    checks = {
        'countqa_rows': len(countqa.data) == 1528,
        'countqa_class': type(countqa).__name__ == 'CountQADataset',
        'countqa_message_shape': _message_types(count_message) == ['image', 'text'],
        'countqa_number_prompt': count_text.endswith(' Please answer with only a number.'),
        'countqa_parser_sentinels': list(count_parser_sentinels.values()) == [2, 2, 2, 3],
        'spatialmqa_rows': len(spatialmqa.data) == 1076,
        'spatialmqa_class': type(spatialmqa).__name__ == 'SpatialMQADataset',
        'spatialmqa_message_shape': _message_types(spatial_message) == ['image', 'text'],
        'spatialmqa_relation_prompt': 'answer the correct spatial relation' in spatial_text,
        'spatialmqa_relation_options': 'Options: ' in spatial_text and '; ' in spatial_text,
        'spatialmqa_no_text_image_marker': '<image>' not in spatial_text,
        'refcoco_class': type(refcoco).__name__ == 'RefCOCODataset',
        'refcocog_rows_found': len(ref_rows) == 2,
        'refcocog_test_rows': refcocog_test_rows == 9602,
        'refcocog_split': set(ref_rows['split']) == {'RefCOCOg_test'},
        'refcoco_bbox_prompt': 'bounding box coordinate' in ref_text.lower(),
        'refcoco_gt_roundtrip': abs(ref_iou - 1.0) < 1e-12,
    }
    failed = {key: value for key, value in checks.items() if not value}
    if failed:
        raise AssertionError(failed)

    payload = {
        'checks': checks,
        'datasets': {
            'CountQA': {
                'class': type(countqa).__name__,
                'rows': len(countqa.data),
                'index': str(count_row['index']),
                'answer': str(count_row['answer']),
                'message': count_message,
                'parser_sentinels': count_parser_sentinels,
            },
            'SpatialMQA': {
                'class': type(spatialmqa).__name__,
                'rows': len(spatialmqa.data),
                'index': str(spatial_row['index']),
                'answer_letter': str(spatial_row['answer']),
                'answer_relation': str(spatial_row['relation']),
                'message': spatial_message,
            },
            'RefCOCOg_test': {
                'class': type(refcoco).__name__,
                'aggregate_rows': len(refcoco.data),
                'test_rows': refcocog_test_rows,
                'selected_indices': sorted(ref_rows['index'].astype(str).tolist()),
                'selected_splits': sorted(ref_rows['split'].astype(str).unique().tolist()),
                'answer': str(ref_row['answer']),
                'message': ref_message,
                'gt_roundtrip_iou': ref_iou,
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'checks': checks}, sort_keys=True))


if __name__ == '__main__':
    main()
