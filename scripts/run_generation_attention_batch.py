"""Fixed four-shard driver; inference stays in native vLLM subprocesses."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def attach_reused_shards(root, records):
    """Reference verified completed shards without copying or rewriting results."""
    for record in records:
        source = Path(record['path']).resolve()
        for name, digest in record['sha256'].items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Reused artifact changed: {source/name}')
        complete = json.loads((source/'generation-complete.json').read_text())
        if complete['samples'] != record['limit'] or complete['conditions'] != ['slp','no_write','vtt']:
            raise RuntimeError('Reused shard has a different completion contract')
        (root/f"shard-{record['id']}").symlink_to(source, target_is_directory=True)


def run(config_path):
    config = json.loads(config_path.read_text())
    root, repo = Path(config['output']), Path(config['repo'])
    if socket.gethostname() != config['hostname']:
        raise RuntimeError('Wrong execution host')
    actual = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
    if actual != config['commit'] or subprocess.check_output(
            ['git','status','--porcelain'], cwd=repo, text=True).strip():
        raise RuntimeError('Execution checkout differs from frozen clean commit')
    gpu_csv = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                       '--format=csv,noheader,nounits'], text=True)
    free = {int(line.split(',')[0]):int(line.split(',')[1]) for line in gpu_csv.splitlines()}
    selected_gpus = [i for shard in config['shards'] for i in shard['gpus']]
    if len(set(selected_gpus)) != len(selected_gpus):
        raise RuntimeError('Inference shards must use disjoint GPUs')
    if any(free.get(i, 0) < 70000 for i in selected_gpus):
        raise RuntimeError('The selected GPUs no longer have the required free memory')
    root.mkdir(parents=True, exist_ok=True)
    with (root/'started.json').open('x') as handle:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname(),
                       started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()), handle)
    reused = config.get('reused_shards', [])
    if {x['id'] for x in reused} & {x['id'] for x in config['shards']}:
        raise RuntimeError('A shard cannot be both reused and launched')
    attach_reused_shards(root, reused)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('QWEN2VL_VLLM_') or key in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_PORT','MASTER_ADDR']:
            env.pop(key)
    env.update(config['environment'])
    status = dict(phase='inference_running', host=socket.gethostname(), pid=os.getpid(),
                  config=str(config_path), output=str(root), shards=[], reused_shards=reused)
    processes = []
    for shard in config['shards']:
        i = shard['id']
        command = [config['python'], '-u', '-m', 'vlmeval.probes.run_generation_attention',
                   '--output', str(root/f'shard-{i}'), '--offset', str(shard['offset']),
                   '--limit', str(shard['limit']), *config['common_args']]
        child_env = dict(env, CUDA_VISIBLE_DEVICES=','.join(map(str, shard['gpus'])))
        cache_env = shard.get('environment', {})
        if set(cache_env) - {'TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','VLLM_CACHE_ROOT'}:
            raise RuntimeError('Per-shard overrides are limited to compilation cache paths')
        child_env.update(cache_env)
        log_path = root/f'shard-{i}.log'
        with log_path.open('x') as log:
            process = subprocess.Popen(command, cwd=repo, env=child_env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        status['shards'].append(dict(**shard, pid=process.pid, log=str(log_path),
                                     command=command, cache_environment=cache_env,
                                     returncode=None))
        save(root/'status.json', status)
    while True:
        changed = False
        for record, process in zip(status['shards'], processes):
            code = process.poll()
            if code is not None and record['returncode'] is None:
                record['returncode'] = code
                changed = True
        if changed:
            save(root/'status.json', status)
        if all(record['returncode'] is not None for record in status['shards']):
            break
        time.sleep(5)
    if any(record['returncode'] != 0 for record in status['shards']):
        status['phase'] = 'inference_failed'
        save(root/'status.json', status)
        return 1
    status['phase'] = 'merging_and_scoring'
    save(root/'status.json', status)
    # Same credential/model handling as the existing run_bound.sh recipe.
    judge_env = dict(env, CUDA_VISIBLE_DEVICES='', JUDGE_LOADER=config['judge_loader'])
    command = ['bash', '-c', 'set -euo pipefail; source "$JUDGE_LOADER"; '
               'unset LOCAL_LLM VLMEVAL_OPENAI_TEMPERATURE; exec "$@"', 'judge',
               config['python'], '-u', '-m', 'vlmeval.probes.score_generation_attention',
               '--root', str(root), '--nproc', '512']
    with (root/'score.log').open('a') as log:
        process = subprocess.Popen(command, cwd=repo, env=judge_env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        status['scoring_pid'] = process.pid
        save(root/'status.json', status)
        code = process.wait()
    status.update(phase='scored' if code == 0 else 'scoring_failed', scoring_returncode=code)
    save(root/'status.json', status)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    sys.exit(run(args.config))
