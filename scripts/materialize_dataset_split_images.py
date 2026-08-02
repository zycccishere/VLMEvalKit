#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from vlmeval.dataset import build_dataset


def _atomic_write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    tmp.write_text(content, encoding='utf-8')
    os.replace(tmp, path)


def _existing_image_paths(dataset, row):
    if 'image_path' in row and isinstance(row['image_path'], str):
        path = Path(row['image_path'])
        if not path.is_absolute():
            path = Path(dataset.img_root) / path
        return [path]
    if 'image' in row and isinstance(row['image'], str):
        return [Path(dataset.img_root) / f'{row["index"]}.jpg']
    raise AssertionError(
        f'Check-only path derivation is unsupported for index {row["index"]}.'
    )


def _verify_image(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise AssertionError(f'Missing or empty materialized image: {path}')
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
    if width <= 0 or height <= 0:
        raise AssertionError(f'Invalid image dimensions for {path}: {(width, height)}')
    return path.stat().st_size, width, height


def main():
    parser = argparse.ArgumentParser(
        description='Materialize one dataset split through its real dump_image path.'
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split-column', default='split')
    parser.add_argument('--split-value', required=True)
    parser.add_argument('--expected-count', required=True, type=int)
    parser.add_argument('--manifest-output', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()

    dataset = build_dataset(args.dataset)
    if dataset is None or not hasattr(dataset, 'data') or not hasattr(dataset, 'dump_image'):
        raise RuntimeError(f'Dataset does not expose the required image adapter: {args.dataset}')
    data = dataset.data
    required = {'index', args.split_column}
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f'{args.dataset} adapter is missing columns: {missing}')
    selected = data[data[args.split_column].astype(str) == args.split_value]
    if len(selected) != args.expected_count:
        raise AssertionError(
            f'{args.dataset}:{args.split_value} has {len(selected)} rows; '
            f'expected {args.expected_count}.'
        )

    records = []
    for offset, (_, row) in enumerate(selected.iterrows(), start=1):
        paths = _existing_image_paths(dataset, row) if args.check_only else dataset.dump_image(row)
        if len(paths) != 1:
            raise AssertionError(
                f'Expected one image for index {row["index"]}, found {len(paths)}.'
            )
        path = Path(paths[0]).resolve()
        size, width, height = _verify_image(path)
        records.append(f'{row["index"]}\t{path}\t{size}\t{width}x{height}\n')
        if offset % 500 == 0 or offset == len(selected):
            print(f'[MATERIALIZE] {offset}/{len(selected)}', flush=True)

    record_payload = ''.join(records)
    manifest = {
        'dataset': args.dataset,
        'split_column': args.split_column,
        'split_value': args.split_value,
        'adapter_class': type(dataset).__name__,
        'selected_count': len(selected),
        'image_count': len(records),
        'image_root': str(Path(dataset.img_root).resolve()),
        'records_sha256': hashlib.sha256(record_payload.encode('utf-8')).hexdigest(),
    }
    output = Path(args.manifest_output)
    if args.check_only:
        existing_manifest = json.loads(output.read_text(encoding='utf-8'))
        if existing_manifest != manifest:
            raise AssertionError(f'Existing materialization manifest does not match: {output}')
    else:
        _atomic_write_text(output, json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, sort_keys=True))


if __name__ == '__main__':
    main()
