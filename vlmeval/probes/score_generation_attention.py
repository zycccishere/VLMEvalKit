"""Merge fixed LogicVista shards and call the existing dataset evaluator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vlmeval.probes.readout_v2 import sha256_file, write_json

CONDITIONS = ['slp', 'no_write', 'vtt']


def merge_records(groups, expected_ids):
    records = {}
    for group in groups:
        for record in group:
            key = str(record['index'])
            if key in records:
                raise ValueError(f'Duplicate original index: {key}')
            records[key] = record
    if set(records) != set(expected_ids):
        raise ValueError('Merged coverage differs from frozen LogicVista indices')
    return [records[key] for key in expected_ids]


def run(root, nproc):
    from vlmeval.dataset.image_vqa import LogicVista
    from vlmeval.smp import dump
    if os.environ.get('LOCAL_LLM') or os.environ.get('VLMEVAL_OPENAI_TEMPERATURE'):
        raise ValueError('Use the existing gpt-4o-mini judge recipe without model/temperature overrides')
    dataset = LogicVista('LogicVista')
    ids = dataset.data['index'].astype(str).tolist()
    assert len(ids) == 447 and len(set(ids)) == 447
    shards = [root/f'shard-{i}' for i in range(4)]
    manifests = [json.loads((p/'manifest.json').read_text()) for p in shards]
    for path, manifest in zip(shards, manifests):
        complete = json.loads((path/'generation-complete.json').read_text())
        assert complete['conditions'] == CONDITIONS
        assert complete['samples'] == manifest['args']['limit']
        for field in ['dataset_sha256', 'versions', 'source_sha256', 'flex_backend_sha256',
                      'model_metadata_sha256']:
            assert manifest[field] == manifests[0][field], field
        sampling = {k:v for k,v in manifest['sampling'].items() if k != 'sample_seeds'}
        assert sampling == {k:v for k,v in manifests[0]['sampling'].items() if k != 'sample_seeds'}
        for field in ['model', 'tp', 'batch_size', 'max_model_len', 'max_batched_tokens',
                      'max_tokens', 'seed', 'gpu_memory_utilization']:
            assert manifest['args'][field] == manifests[0]['args'][field], field
    assert manifests[0]['dataset_sha256'] == sha256_file(Path(os.environ['LMUData'])/'LogicVista.tsv')
    evaluation = {}
    for condition in CONDITIONS:
        groups = []
        for path, manifest in zip(shards, manifests):
            records = json.loads((path/f'{condition}.json').read_text())['records']
            assert [str(r['index']) for r in records] == manifest['sample_indices']
            groups.append(records)
        records = merge_records(groups, ids)
        frame = dataset.data.drop(columns=['image', 'image_path'], errors='ignore').copy()
        assert frame['answer'].astype(str).tolist() == [r['answer'] for r in records]
        frame['prediction'] = [r['prediction'] for r in records]
        dest = root/'merged'/condition
        dest.mkdir(parents=True, exist_ok=True)
        prediction_file = dest/'LogicVista.xlsx'
        dump(frame, str(prediction_file))
        write_json(dest/'generation-audit.json', dict(
            samples=len(records), truncated=sum(r['finish_reason']=='length' for r in records),
            generated_tokens=sum(len(r['token_ids']) for r in records),
            prediction_sha256=sha256_file(prediction_file),
            shard_sources=[str(p/f'{condition}.json') for p in shards]))
        score = dataset.evaluate(str(prediction_file), model='gpt-4o-mini', nproc=nproc)
        if score is None:
            raise RuntimeError(f'Native judge did not produce scores for {condition}')
        evaluation[condition] = score.to_dict(orient='records')
        write_json(dest/'evaluation-complete.json', dict(
            judge='gpt-4o-mini-2024-07-18', samples=447, score=evaluation[condition],
            scorer_sha256=sha256_file(Path(__file__).resolve().parents[1]/'dataset/image_vqa.py'),
            parser_sha256=sha256_file(Path(__file__).resolve().parents[1]/'dataset/utils/logicvista.py')))
    write_json(root/'evaluation-complete.json', evaluation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--nproc', type=int, default=512)
    args = parser.parse_args()
    run(args.root, args.nproc)
