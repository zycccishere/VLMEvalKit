#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

from vlmeval.dataset import build_dataset


def _atomic_write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    tmp.write_text(content, encoding='utf-8')
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(
        description='Build an index allowlist from the data returned by a VLMEvalKit adapter.'
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split-column', default='split')
    parser.add_argument('--split-value', required=True)
    parser.add_argument('--expected-count', required=True, type=int)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest-output', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()

    dataset = build_dataset(args.dataset)
    if dataset is None or not hasattr(dataset, 'data'):
        raise RuntimeError(f'Unable to build dataset adapter: {args.dataset}')
    data = dataset.data
    required = {'index', args.split_column}
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f'{args.dataset} adapter is missing columns: {missing}')

    selected = data[data[args.split_column].astype(str) == args.split_value]
    indices = selected['index'].astype(str).tolist()
    if len(indices) != args.expected_count:
        raise AssertionError(
            f'{args.dataset}:{args.split_value} has {len(indices)} rows; '
            f'expected {args.expected_count}.'
        )
    if any(not value.strip() for value in indices):
        raise AssertionError('Selected split contains an empty index.')
    if len(set(indices)) != len(indices):
        raise AssertionError('Selected split contains duplicate indices.')

    output = Path(args.output)
    payload = ''.join(f'{value}\n' for value in indices)

    manifest = {
        'dataset': args.dataset,
        'split_column': args.split_column,
        'split_value': args.split_value,
        'adapter_class': type(dataset).__name__,
        'aggregate_count': int(len(data)),
        'selected_count': len(indices),
        'first_index': indices[0],
        'last_index': indices[-1],
        'allowlist_path': str(output.resolve()),
        'allowlist_sha256': hashlib.sha256(payload.encode('utf-8')).hexdigest(),
    }
    manifest_output = Path(args.manifest_output)
    if args.check_only:
        if output.read_text(encoding='utf-8') != payload:
            raise AssertionError(f'Existing allowlist does not match adapter split: {output}')
        existing_manifest = json.loads(manifest_output.read_text(encoding='utf-8'))
        if existing_manifest != manifest:
            raise AssertionError(f'Existing allowlist manifest does not match: {manifest_output}')
    else:
        _atomic_write_text(output, payload)
        _atomic_write_text(
            manifest_output, json.dumps(manifest, indent=2, sort_keys=True) + '\n'
        )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == '__main__':
    main()
