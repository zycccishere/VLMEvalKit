#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault('VLMEVAL_LAZY_INIT', '1')
import vlmeval  # noqa: E402


dataset_package = types.ModuleType('vlmeval.dataset')
dataset_package.__path__ = [str(Path(vlmeval.__file__).resolve().parent / 'dataset')]
sys.modules.setdefault('vlmeval.dataset', dataset_package)
SpatialMQA = importlib.import_module('vlmeval.dataset.spatialmqa').SpatialMQA


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare the pinned SpatialMQA test split once under LMUData.'
    )
    parser.add_argument(
        '--lmu-data-root',
        default=os.environ.get('LMUData'),
        help='LMUData root. Defaults to $LMUData or ~/LMUData.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Publish a fresh validated revision even if the current manifest is valid.',
    )
    parser.add_argument('--max-workers', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    lmu_data_root = args.lmu_data_root or str(Path.home() / 'LMUData')
    manifest = SpatialMQA.prepare(
        lmu_data_root=lmu_data_root,
        force=args.force,
        max_workers=args.max_workers,
    )
    summary = {
        'prepared_root': str(Path(lmu_data_root).resolve() / SpatialMQA.PREPARED_DIRNAME),
        'source_revision': manifest['source']['revision'],
        'row_count': manifest['row_count'],
        'unique_image_count': manifest['unique_image_count'],
        'data_file': manifest['data_file'],
        'checks': manifest['checks'],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
