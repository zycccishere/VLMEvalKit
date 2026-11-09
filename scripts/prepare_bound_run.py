#!/usr/bin/env python3
"""Bind an ordinary native task CSV to existing viewer cells; no GPU work."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def load_runner():
    path = Path(__file__).resolve().parents[1] / 'vlmeval/cli/run_benchmark.py'
    spec = importlib.util.spec_from_file_location('bound_native_runner', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--matrix', type=Path, required=True)
    ap.add_argument('--models', type=Path, required=True)
    ap.add_argument('--tasks', type=Path, required=True, help='native CSV with one extra cell_id column')
    ap.add_argument('--registry', type=Path, required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--stack', required=True)
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    if not a.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.run_id):
        ap.error('run-id must be a simple unique directory name')
    native = load_runner()
    protocol = json.loads(a.plan.read_text())['final_protocol']
    cells = {}
    for s in protocol['matrix_specs']:
        for ds in s['datasets']:
            for cond in s['conditions']:
                key = '|'.join(['最终协议', s['model'], cond, ds])
                cells.setdefault(key, []).append(s['id'])
    matrix = yaml.safe_load(a.matrix.read_text())
    models = yaml.safe_load(a.models.read_text())
    repo = Path(matrix['repo_root']).resolve()
    assert repo.is_dir() and '{' not in str(repo), 'matrix.repo_root must name the pinned execution checkout'
    sha = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'], text=True).strip(), 'execution checkout must be clean'
    root = Path(matrix['results_root'])
    assert root.is_absolute() and not (root.exists() and any(root.iterdir())), 'use a fresh results_root; never overwrite a previous run'
    rows = list(csv.DictReader(a.tasks.open()))
    assert rows, 'empty task manifest'
    bindings, ids = {}, set()
    for row in rows:
        cell = row['cell_id']
        assert cell in cells and cell not in ids, ('unknown/duplicate cell',cell)
        _, model_name, condition, ds = cell.split('|')
        assert ds == row['dataset']
        allowed = protocol['model_entries'][model_name]
        if ' · ' in condition:
            assert model_name == 'Qwen3.8-27B'
            allowed = protocol['effort_entries'][condition.split(' · ')[0]]
        assert row['model'] in allowed, ('model entry does not match table cell',cell,row['model'])
        mode = {'VT':'image_text','VTT':'image_text_text','SLP':'image_text_image_text','VVTT':'image_image_text_text',
                '1 blank':'image_text_image_text','2 blank':'image_text_image_image_text',
                '3 blank':'image_text_image_image_image_text','dots':'image_text_filler_text',
                'spaces':'image_text_filler_text','noise':'image_text_image_text',
                'SLP-½':'image_text_image_text','SLP-¼':'image_text_image_text'}
        cond = condition.split(' · ')[-1]
        assert row['mode'] == mode[cond], ('mode does not match table condition',cell)
        transform = 'blank' if 'blank' in cond else 'v2_tokens_half' if cond=='SLP-½' else 'v2_tokens_quarter' if cond=='SLP-¼' else 'noise_image_seed17' if cond=='noise' else 'baseline'
        assert row['transform'] == transform
        model = models['models'][row['model']]
        if cond in ['dots','spaces']:
            profile = models['env_profiles'][model['env_profile']]
            effective = dict(profile.get('base_env',{}), **model.get('task_env',{}))
            assert effective.get('REPLAY_TEXT_FILLER_TYPE') == cond, 'filler type must be explicit; split dots/spaces into separate native batches'
        tag = '__'.join([row[k] for k in ['model','policy','mode','transform','dataset']])
        assert tag not in bindings
        bindings[tag] = dict(cell_id=cell, protocol_id=protocol['id'], tables=cells[cell], model=model_name, condition=condition)
        ids.add(cell)
    for registration in a.registry.glob('*/registration.json'):
        existing = json.loads(registration.read_text())
        assert not (ids & set(existing['sources'])), f'cells already registered in {registration}; resume their saved launcher instead'
    matrix['evaluation'] = dict(matrix.get('evaluation',{}), launch_mode='bg', judge='gpt-4o-mini', nproc=512)
    matrix['scheduler'] = 'gpu_pool'
    matrix['experiment'] = dict(run_id=a.run_id, bindings=bindings, execution_sha=sha)
    args = argparse.Namespace(matrix_config=a.matrix, model_config=a.models,
        task_manifest=a.tasks, nodes=1, node_rank=0, gpu_ids='', models='',policies='',
        modes='',transforms='',datasets='',scheduler='',manifest_is_node_shard=False,resume_infer=None,plan_only=True)
    runner = native.BenchmarkRunner(Path(__file__).resolve().parents[1], args)
    assert {t.tag for t in runner.tasks} == set(bindings), 'native matrix selectors did not generate exactly the bound rows'
    print(json.dumps(dict(run_id=a.run_id, task_count=len(rows), judge=matrix['evaluation'], cells=sorted(ids)), ensure_ascii=False, indent=2))
    if a.check_only:
        return
    out = a.registry / a.run_id
    out.mkdir(parents=True, exist_ok=False)
    for name, value in [('matrix.yaml',matrix),('models.yaml',models)]:
        (out/name).write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False))
    with (out/'tasks.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    sources = {}
    for t in runner.tasks:
        b = bindings[t.tag]
        sources[b['cell_id']] = dict(run=a.run_id, path=str(runner.task_root(t)), stack=a.stack,
            model_entry=t.model_key, inventory_id='|'.join([a.run_id,b['model'],b['condition'],t.dataset]),
            binding=b, runtime=dict(judge='gpt-4o-mini',nproc=512,launch_mode='bg'))
    registration = dict(run_id=a.run_id, protocol_id=protocol['id'], sources=sources,
        execution_repo=str(repo), execution_sha=sha,
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.plan,a.matrix,a.models,a.tasks]},
        prepared_files={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['matrix.yaml','models.yaml','tasks.csv']},
        state='registered_not_started')
    # This file is the publication marker; incomplete preparations aren't seen.
    (out/'registration.json').write_text(json.dumps(registration,ensure_ascii=False,indent=2)+'\n')
    entry=Path(__file__).resolve().parent/'run_benchmark.sh'
    print('Launch after loading the existing judge environment:')
    print(shlex.join(['bash',str(entry),'--matrix-config',str(out/'matrix.yaml'),
                     '--model-config',str(out/'models.yaml'),'--task-manifest',str(out/'tasks.csv')]))


if __name__ == '__main__':
    main()
