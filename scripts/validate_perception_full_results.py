#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


MODE_TO_CONDITION = {
    'image_text': 'IQ',
    'image_text_image_text': 'IQIQ',
}
EXPECTED_ROWS = {
    'CountQA': 1528,
    'SpatialMQA': 1076,
    'RefCOCO': 9602,
}


def _load_manifest(path, expected_rows, matrix, task_identity, artifact_type):
    if not path.is_file():
        raise AssertionError(f'Missing manifest: {path}')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if int(payload.get('schema_version', -1)) != 2:
        raise AssertionError(f'Full run requires schema_version=2: {path}')
    if payload.get('status') != 'complete':
        raise AssertionError(f'Incomplete manifest: {path}')
    if int(payload.get('expected_rows', -1)) != expected_rows:
        raise AssertionError(
            f'Wrong expected_rows in {path}: {payload.get("expected_rows")} != {expected_rows}'
        )
    if payload.get('matrix') != matrix:
        raise AssertionError(f'Wrong matrix identity in {path}: {payload.get("matrix")!r}')
    if payload.get('artifact_type') != artifact_type:
        raise AssertionError(
            f'Wrong artifact type in {path}: {payload.get("artifact_type")!r}'
        )
    task = payload.get('task')
    if not isinstance(task, dict):
        raise AssertionError(f'Missing task identity in {path}')
    for key, expected in task_identity.items():
        if task.get(key) != expected:
            raise AssertionError(
                f'Wrong task identity {key} in {path}: {task.get(key)!r} != {expected!r}'
            )
    model = payload.get('model')
    if not isinstance(model, dict) or model.get('key') != task_identity['model_key']:
        raise AssertionError(f'Wrong model identity in {path}: {model!r}')
    return payload


def _one_file(paths, label):
    paths = [path for path in paths if path.is_file()]
    if len(paths) != 1:
        raise AssertionError(f'Expected one {label}, found {len(paths)}: {paths}')
    return paths[0]


def _prediction_count(path):
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        frame = pd.read_excel(path, usecols=['index', 'prediction'])
    elif path.suffix.lower() == '.tsv':
        frame = pd.read_csv(path, sep='\t', usecols=['index', 'prediction'])
    else:
        frame = pd.read_csv(path, usecols=['index', 'prediction'])
    if frame['index'].astype(str).duplicated().any():
        raise AssertionError(f'Duplicate prediction indices in {path}')
    return len(frame)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_between(value, low, high, label):
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise AssertionError(f'{label}={value} is outside [{low}, {high}]')
    return value


def _metric_row(dataset, score_file, expected_rows):
    score = pd.read_csv(score_file)
    if dataset == 'CountQA':
        if len(score) != 1:
            raise AssertionError(f'CountQA score must have one row: {score_file}')
        row = score.iloc[0]
        if int(row['annotations']) != expected_rows or int(row['predictions']) != expected_rows:
            raise AssertionError(f'CountQA denominator mismatch: {score_file}')
        if int(row['missing']) != 0:
            raise AssertionError(f'CountQA has missing predictions: {score_file}')
        value = _finite_between(row['Overall'], 0.0, 1.0, 'CountQA Overall')
        return value * 100.0, int(row['correct']), {
            'missing': int(row['missing']),
            'invalid_format': int(row['invalid_format']),
        }
    if dataset == 'SpatialMQA':
        if len(score) != 1:
            raise AssertionError(f'SpatialMQA score must have one row: {score_file}')
        row = score.iloc[0]
        if int(row['total']) != expected_rows or int(row['predictions']) != expected_rows:
            raise AssertionError(f'SpatialMQA denominator mismatch: {score_file}')
        if int(row['missing']) != 0:
            raise AssertionError(f'SpatialMQA has missing predictions: {score_file}')
        value = _finite_between(row['Overall'], 0.0, 1.0, 'SpatialMQA Overall')
        return value * 100.0, int(row['correct']), {'missing': int(row['missing'])}

    selected = score[score['Split'].astype(str) == 'RefCOCOg_test']
    if len(selected) != 1:
        raise AssertionError(f'RefCOCO score lacks one RefCOCOg_test row: {score_file}')
    row = selected.iloc[0]
    if int(row['Samples']) != expected_rows:
        raise AssertionError(f'RefCOCOg test denominator mismatch: {score_file}')
    value = _finite_between(row['Precision@1'], 0.0, 100.0, 'RefCOCOg Precision@1')
    compliance = _finite_between(
        row['Format Compliance'], 0.0, 100.0, 'RefCOCOg Format Compliance'
    )
    average_iou = _finite_between(row['Average IoU'], 0.0, 1.0, 'RefCOCOg Average IoU')
    return value, None, {
        'format_compliance': compliance,
        'average_iou': average_iou,
    }


def _require_within(path, parent, label):
    path = path.resolve()
    parent = parent.resolve()
    if path != parent and parent not in path.parents:
        raise AssertionError(f'{label} escapes {parent}: {path}')
    return path


def _validate_allowlist_manifest(root, manifest_path):
    manifest_path = _require_within(manifest_path, root / '_inputs', 'allowlist manifest')
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    if payload.get('dataset') != 'RefCOCO' or payload.get('split_value') != 'RefCOCOg_test':
        raise AssertionError(f'Wrong RefCOCOg allowlist identity: {manifest_path}')
    if int(payload.get('selected_count', -1)) != EXPECTED_ROWS['RefCOCO']:
        raise AssertionError(f'Wrong RefCOCOg allowlist count: {manifest_path}')
    allowlist = _require_within(Path(payload['allowlist_path']), root / '_inputs', 'allowlist')
    raw = allowlist.read_bytes()
    if hashlib.sha256(raw).hexdigest() != payload.get('allowlist_sha256'):
        raise AssertionError(f'RefCOCOg allowlist checksum mismatch: {allowlist}')
    indices = [line.strip() for line in raw.decode('utf-8').splitlines() if line.strip()]
    if len(indices) != EXPECTED_ROWS['RefCOCO'] or len(set(indices)) != len(indices):
        raise AssertionError(f'RefCOCOg allowlist cardinality mismatch: {allowlist}')
    if any(not index.startswith('RefCOCOg_test_') for index in indices):
        raise AssertionError(f'RefCOCOg allowlist contains another split: {allowlist}')
    return payload


def _validate_run_markers(root, run_uuid, nodes=2):
    if nodes < 1:
        raise AssertionError(f'Node count must be positive: {nodes}')
    marker_paths = [
        root / '_control' / 'run_uuid',
        root / '_inputs' / 'PREFLIGHT_READY',
    ]
    marker_paths.extend(root / f'NODE_{rank}_RUNNER_DONE' for rank in range(nodes))
    for marker in marker_paths:
        if not marker.is_file():
            raise AssertionError(f'Missing run marker: {marker}')
        value = marker.read_text(encoding='utf-8').strip()
        if value != run_uuid:
            raise AssertionError(f'Run marker UUID mismatch in {marker}: {value!r}')


def validate(root, models, matrix, allowlist_manifest):
    allowlist_payload = _validate_allowlist_manifest(root, allowlist_manifest)
    records = []
    for mode, condition in MODE_TO_CONDITION.items():
        for model in models:
            for dataset, expected_rows in EXPECTED_ROWS.items():
                task_root = root / 'default' / mode / 'baseline' / model / dataset
                task_identity = {
                    'model_key': model,
                    'policy_key': 'default',
                    'mode': mode,
                    'transform': 'baseline',
                    'dataset': dataset,
                }
                prediction_manifest = _load_manifest(
                    task_root / 'predictions' / 'manifest.json',
                    expected_rows,
                    matrix,
                    task_identity,
                    'prediction',
                )
                eval_manifest = _load_manifest(
                    task_root / 'eval' / 'manifest.json',
                    expected_rows,
                    matrix,
                    task_identity,
                    'eval',
                )
                if int(eval_manifest.get('returncode', -1)) != 0:
                    raise AssertionError(f'Eval return code is nonzero: {task_root}')

                prediction_file = _require_within(
                    Path(prediction_manifest['prediction_file']),
                    task_root / 'predictions',
                    'prediction file',
                )
                eval_prediction_file = Path(eval_manifest.get('prediction_file', '')).resolve()
                if eval_prediction_file != prediction_file:
                    raise AssertionError(f'Eval/prediction manifest mismatch: {task_root}')
                if _prediction_count(prediction_file) != expected_rows:
                    raise AssertionError(f'Prediction denominator mismatch: {prediction_file}')
                prediction_hash = _sha256(prediction_file)
                if prediction_manifest.get('prediction_sha256') != prediction_hash:
                    raise AssertionError(f'Prediction manifest checksum mismatch: {prediction_file}')
                if eval_manifest.get('prediction_sha256') != prediction_hash:
                    raise AssertionError(f'Eval prediction checksum mismatch: {prediction_file}')
                score_file = _require_within(_one_file(
                    [Path(value) for value in eval_manifest.get('score_files', [])], 'score file'
                ), task_root / 'eval', 'score file')
                score_hashes = eval_manifest.get('score_file_sha256')
                if not isinstance(score_hashes, dict):
                    raise AssertionError(f'Missing score checksums: {task_root}')
                resolved_score_hashes = {
                    str(Path(path).resolve()): digest for path, digest in score_hashes.items()
                }
                if set(resolved_score_hashes) != {str(score_file)}:
                    raise AssertionError(f'Score checksum path mismatch: {task_root}')
                if resolved_score_hashes[str(score_file)] != _sha256(score_file):
                    raise AssertionError(f'Score checksum mismatch: {score_file}')
                accuracy, correct, diagnostics = _metric_row(dataset, score_file, expected_rows)
                records.append({
                    'model': model,
                    'condition': condition,
                    'dataset': 'RefCOCOg_test' if dataset == 'RefCOCO' else dataset,
                    'accuracy': accuracy,
                    'correct': correct,
                    'samples': expected_rows,
                    **diagnostics,
                    'prediction_file': str(prediction_file),
                    'score_file': str(score_file),
                })
    if len(records) != len(MODE_TO_CONDITION) * len(models) * len(EXPECTED_ROWS):
        raise AssertionError(f'Wrong result cell count: {len(records)}')
    return records, allowlist_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--run-uuid', required=True)
    parser.add_argument('--nodes', type=int, default=2)
    parser.add_argument('--allowlist-manifest', required=True)
    parser.add_argument('--json-output', required=True)
    parser.add_argument('--csv-output', required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    _validate_run_markers(root, args.run_uuid, args.nodes)
    records, allowlist_payload = validate(
        root,
        args.models,
        args.matrix,
        Path(args.allowlist_manifest),
    )
    payload = {
        'ok': True,
        'root': str(root),
        'matrix': args.matrix,
        'run_uuid': args.run_uuid,
        'nodes': args.nodes,
        'cell_count': len(records),
        'expected_rows': EXPECTED_ROWS,
        'refcocog_allowlist': allowlist_payload,
        'records': records,
    }
    json_output = Path(args.json_output)
    csv_output = Path(args.csv_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    pd.DataFrame(records).to_csv(csv_output, index=False)
    print(json.dumps({'ok': True, 'cell_count': len(records), 'root': str(root)}, sort_keys=True))


if __name__ == '__main__':
    main()
