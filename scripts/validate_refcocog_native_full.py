#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


MODE_TO_CONDITION = {
    'image_text': 'IQ',
    'image_text_image_text': 'IQIQ',
}
ALL_MODELS = [
    'gemma3_4b',
    'gemma3_12b',
    'gemma3_27b',
    'minicpm_o_45_no_reasoning',
    'minicpm_v_45_no_reasoning',
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    if path.suffix.lower() == '.tsv':
        return pd.read_csv(path, sep='\t')
    return pd.read_csv(path)


def _one(paths, label):
    paths = [path for path in paths if path.is_file()]
    if len(paths) != 1:
        raise AssertionError(f'Expected one {label}, found {len(paths)}: {paths}')
    return paths[0]


def _manifest(path: Path, matrix: str, model: str, mode: str, expected_rows: int, artifact: str):
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema_version') != 2 or payload.get('status') != 'complete':
        raise AssertionError(f'Incomplete schema-v2 manifest: {path}')
    if payload.get('matrix') != matrix or payload.get('artifact_type') != artifact:
        raise AssertionError(f'Manifest identity mismatch: {path}')
    if int(payload.get('expected_rows', -1)) != expected_rows:
        raise AssertionError(f'Manifest denominator mismatch: {path}')
    task = payload.get('task') or {}
    expected_task = {
        'model_key': model,
        'policy_key': 'default',
        'mode': mode,
        'transform': 'baseline',
        'dataset': 'RefCOCO',
    }
    for key, expected in expected_task.items():
        if task.get(key) != expected:
            raise AssertionError(f'Manifest task {key} mismatch in {path}')
    return payload


def validate(root: Path, models: list[str], matrix: str, allowlist: Path, expected_rows: int):
    expected_indices = [line.strip() for line in allowlist.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(expected_indices) != expected_rows or len(set(expected_indices)) != expected_rows:
        raise AssertionError('Allowlist cardinality or uniqueness mismatch')
    expected_index_set = set(expected_indices)
    records = []

    for model in models:
        expected_coordinate_mode = (
            'gemma_0_1000_yxyx' if model.startswith('gemma3')
            else 'minicpm_v46_0_1000_xyxy'
        )
        for mode, condition in MODE_TO_CONDITION.items():
            task_root = root / 'default' / mode / 'baseline' / model / 'RefCOCO'
            prediction_manifest = _manifest(
                task_root / 'predictions' / 'manifest.json',
                matrix,
                model,
                mode,
                expected_rows,
                'prediction',
            )
            eval_manifest = _manifest(
                task_root / 'eval' / 'manifest.json',
                matrix,
                model,
                mode,
                expected_rows,
                'eval',
            )
            if int(eval_manifest.get('returncode', -1)) != 0:
                raise AssertionError(f'Nonzero eval return code: {task_root}')

            prediction_path = Path(prediction_manifest['prediction_file']).resolve()
            if not prediction_path.is_file() or task_root.resolve() not in prediction_path.parents:
                raise AssertionError(f'Invalid prediction path: {prediction_path}')
            prediction = _read_table(prediction_path)
            indices = prediction['index'].astype(str).tolist()
            if len(indices) != expected_rows or len(set(indices)) != expected_rows:
                raise AssertionError(f'Prediction cardinality mismatch: {prediction_path}')
            if set(indices) != expected_index_set:
                raise AssertionError(f'Prediction indices differ from allowlist: {prediction_path}')
            prediction_hash = _sha256(prediction_path)
            if prediction_manifest.get('prediction_sha256') != prediction_hash:
                raise AssertionError(f'Prediction manifest hash mismatch: {prediction_path}')
            if eval_manifest.get('prediction_sha256') != prediction_hash:
                raise AssertionError(f'Eval prediction hash mismatch: {prediction_path}')

            eval_root = task_root / 'eval'
            detail_path = _one(eval_root.glob('*_detail.*'), 'detail table')
            detail = _read_table(detail_path)
            detail_indices = detail['index'].astype(str).tolist()
            if len(detail_indices) != expected_rows or len(set(detail_indices)) != expected_rows:
                raise AssertionError(f'Detail cardinality mismatch: {detail_path}')
            if set(detail_indices) != expected_index_set:
                raise AssertionError(f'Detail indices differ from allowlist: {detail_path}')
            if set(detail['coordinate_mode'].astype(str)) != {expected_coordinate_mode}:
                raise AssertionError(f'Coordinate mode mismatch: {detail_path}')

            score_paths = [Path(path).resolve() for path in eval_manifest.get('score_files', [])]
            score_path = _one(score_paths, 'score table')
            if eval_root.resolve() not in score_path.parents:
                raise AssertionError(f'Score path escapes eval root: {score_path}')
            score_hashes = {
                str(Path(path).resolve()): digest
                for path, digest in (eval_manifest.get('score_file_sha256') or {}).items()
            }
            if score_hashes.get(str(score_path)) != _sha256(score_path):
                raise AssertionError(f'Score hash mismatch: {score_path}')
            score = _read_table(score_path)
            selected = score[score['Split'].astype(str) == 'RefCOCOg_test']
            if len(selected) != 1 or int(selected.iloc[0]['Samples']) != expected_rows:
                raise AssertionError(f'Score denominator mismatch: {score_path}')
            row = selected.iloc[0]
            records.append({
                'model': model,
                'condition': condition,
                'precision_at_1': float(row['Precision@1']),
                'average_iou': float(row['Average IoU']),
                'format_compliance': float(row['Format Compliance']),
                'samples': expected_rows,
                'coordinate_mode': expected_coordinate_mode,
                'prediction_file': str(prediction_path),
                'detail_file': str(detail_path),
                'score_file': str(score_path),
            })
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--allowlist', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS, required=True)
    parser.add_argument('--expected-rows', type=int, default=9602)
    parser.add_argument('--json-output', type=Path, required=True)
    parser.add_argument('--csv-output', type=Path, required=True)
    args = parser.parse_args()

    records = validate(
        args.root.resolve(),
        args.models,
        args.matrix,
        args.allowlist.resolve(),
        args.expected_rows,
    )
    payload = {
        'ok': True,
        'matrix': args.matrix,
        'root': str(args.root.resolve()),
        'models': args.models,
        'expected_rows': args.expected_rows,
        'cell_count': len(records),
        'records': records,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    pd.DataFrame(records).to_csv(args.csv_output, index=False)
    print(json.dumps({'ok': True, 'cell_count': len(records)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
