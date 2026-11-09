#!/usr/bin/env python3
"""Editable host queue for registered cells; executes the existing bound launcher."""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

from prepare_bound_run import load_runner
from atomic_checkpoints import atomic_writer

native = load_runner()
CONTROL_SHA = hashlib.sha256(b''.join(p.read_bytes() for p in
    (Path(__file__), Path(__file__).with_name('prepare_bound_run.py'),
     Path(__file__).with_name('atomic_checkpoints.py'), Path(native.__file__)))).hexdigest()
TERMINAL = {'complete', 'failed', 'cancelled', 'needs_review'}


def nodes(config):
    return config.get('nodes') or {'local': {'gpu_ids': config['gpu_ids'], 'ssh': []}}


def main_node(config):
    return next(name for name, node in nodes(config).items() if not node.get('ssh'))


def remote_command(node, args):
    command = [node.get('python', sys.executable), node.get('script', str(Path(__file__).resolve())), *args]
    return [*node['ssh'], shlex.join(command)]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def identity(pid):
    raw = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start=raw[19], boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def alive(who):
    if not who:
        return False
    try:
        return identity(who['pid']) == who and Path(f"/proc/{who['pid']}/stat").read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def locked(path):
    if not path.exists():
        return False
    with path.open('r') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def edit(self):
        with (self.root/'state.lock').open('a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            state = read(self.root/'state.json') or {'next_id': 1, 'jobs': [], 'events': []}
            yield state
            atomic_writer(lambda value, path: Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n'))(state, self.root/'state.json')

    def snapshot(self):
        return read(self.root/'state.json')


def transition(state, job, status, **details):
    job.update(state=status, updated_at=time.time(), **details)
    state['events'].append(dict(id=job['id'], attempt=job['attempt'], state=status,
                                at=job['updated_at'], details=details))


def validate(spec):
    directory = Path(spec['registration'])
    reg = read(directory/'registration.json')
    if hashlib.sha256((directory/'registration.json').read_bytes()).hexdigest() != spec['registration_sha']:
        raise ValueError('registration changed since admission')
    for name, digest in reg['prepared_files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'prepared configuration changed: {name}')


def validate_checkpoints(spec):
    for checkpoint in Path(spec['path']).rglob('*.pkl'):
        if checkpoint.name.startswith('.'):
            continue
        try:
            with checkpoint.open('rb') as f:
                pickle.load(f)
        except Exception as exc:
            raise ValueError(f'checkpoint unreadable; preserve and repair instead of discarding: {checkpoint}') from exc


def registered_specs(config, run_id, cells, all_cells=False):
    directory = Path(config['registry'])/run_id
    if directory.parent != Path(config['registry']) or not directory.is_dir():
        raise ValueError('unknown registered run')
    reg = read(directory/'registration.json')
    with (directory/'tasks.csv').open() as f:
        rows = list(csv.DictReader(f))
    selected = set(reg['sources']) if all_cells else set(cells)
    if not selected or not selected <= set(reg['sources']):
        raise ValueError('select explicit registered cell IDs, or --all')
    args = SimpleNamespace(matrix_config=directory/'matrix.yaml', model_config=directory/'models.yaml',
        task_manifest=directory/'tasks.csv', nodes=1, node_rank=0, gpu_ids='', models='', policies='',
        modes='', transforms='', datasets='', scheduler='', manifest_is_node_shard=False,
        resume_infer=None, plan_only=True)
    runner = native.BenchmarkRunner(Path(__file__).resolve().parents[1], args)
    if runner.evaluation_cfg.get('launch_mode') != 'bg' or runner.evaluation_cfg.get('judge') != 'gpt-4o-mini' or runner.evaluation_cfg.get('nproc') != 512:
        raise ValueError('queue requires the bound bg/gpt-4o-mini/512 contract')
    tasks = {t.tag: t for t in runner.tasks}
    specs = []
    for row in rows:
        if row['cell_id'] not in selected:
            continue
        tag = '__'.join(row[k] for k in ('model', 'policy', 'mode', 'transform', 'dataset'))
        task = tasks[tag]
        if runner.experiment_cfg['bindings'][tag]['cell_id'] != row['cell_id']:
            raise ValueError('native task binding does not match selected cell')
        spec = dict(run_id=run_id, cell_id=row['cell_id'], row=row,
                    registration=str(directory), registration_sha=hashlib.sha256((directory/'registration.json').read_bytes()).hexdigest(),
                    path=reg['sources'][row['cell_id']]['path'], gpus=runner.models[task.model_key].gpus_per_job,
                    weight=task.weight, model_key=run_id+'/'+task.model_key)
        validate(spec)
        if str(runner.task_root(task)) != spec['path'] or not 0 < spec['gpus'] <= max(len(n['gpu_ids']) for n in nodes(config).values()):
            raise ValueError('invalid cell root or GPU width for this host')
        specs.append(spec)
    return specs


def external_owners():
    """Observe legacy controllers too, including their not-yet-started tasks."""
    result = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = [v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
            if 'vlmeval.cli.run_benchmark' not in args or '--eval-task-tag' in args or '--task-manifest' not in args:
                continue
            manifest = args[max(i for i, x in enumerate(args) if x == '--task-manifest')+1]
            who = identity(int(proc.name))
            with Path(manifest).open() as f:
                for row in csv.DictReader(f):
                    if row.get('cell_id'):
                        result[row['cell_id']] = who
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def artifact_state(spec):
    root = Path(spec['path'])
    manifest = read(root/'eval/manifest.json')
    if manifest.get('status') == 'complete':
        if manifest.get('binding', {}).get('cell_id') != spec['cell_id'] or manifest.get('run_id') != spec['run_id']:
            raise ValueError('score manifest binding mismatch')
        return 'complete'
    if locked(root/'eval/.worker.lock'):
        return 'scoring'
    if locked(root/'.host-queue.lock'):
        return 'observing'
    for phase in ('predictions', 'eval'):
        s = read(root/phase/'status.json')
        if s.get('status') in ('running', 'queued', 'waiting_gpu') and s.get('host', socket.gethostname()) == socket.gethostname():
            pid = s.get('pid', s.get('supervisor_pid'))
            if pid:
                try:
                    who = identity(pid)
                    if alive(who) and (not s.get('pid_start_ticks') or s['pid_start_ticks'] == who['start']):
                        return 'scoring' if phase == 'eval' else 'observing'
                except (FileNotFoundError, ProcessLookupError):
                    pass
    return 'idle'


def probe(store, node_id):
    state = store.snapshot()
    node = nodes(state['config'])[node_id]
    return dict(node=node_id, host=socket.gethostname(), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                instance=state['config'].get('instance_id', str(store.root)),
                source_sha=CONTROL_SHA,
                shared_token=(store.root/'shared-token').read_text() if (store.root/'shared-token').exists() else None,
                daemon_locked=locked(store.root/'daemon.lock'),
                active_locks=[j['id'] for j in state['jobs'] if locked(Path(j['spec']['path'])/'eval/.worker.lock') or locked(Path(j['spec']['path'])/'.host-queue.lock')],
                free=query_gpus(node['gpu_ids']), owners=external_owners(),
                versions={str(j['id']): [j['attempt'], j['updated_at']] for j in state['jobs']},
                workers={str(j['id']): alive(j.get('identity')) for j in state['jobs'] if j.get('node', 'local') == node_id},
                artifacts={str(j['id']): artifact_state(j['spec']) for j in state['jobs']})


def collect_nodes(store):
    state = store.snapshot()
    def collect(item):
        name, node = item
        try:
            if node.get('ssh'):
                p = subprocess.run(remote_command(node, ['--root', str(store.root), 'probe', '--node', name]),
                                   capture_output=True, text=True, check=True, timeout=20)
                result = json.loads(p.stdout)
            else:
                result = probe(store, name)
            if result['source_sha'] != CONTROL_SHA:
                raise ValueError('node control source differs from main')
            if result['instance'] != state['config'].get('instance_id', str(store.root)):
                raise ValueError('node instance mismatch')
            if result['node'] != name:
                raise ValueError('node name mismatch')
            expected = state.get('node_identities', {}).get(name)
            if expected and expected != [result['host'], result['boot']]:
                raise ValueError('node identity changed; do not reuse its reservations')
            return name, result
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return name, {'error': str(exc)}
    with ThreadPoolExecutor(max_workers=len(nodes(state['config']))) as pool:
        return dict(pool.map(collect, nodes(state['config']).items()))


def enqueue(store, specs, priority=0, observe=False):
    owners = external_owners()
    with store.edit() as state:
        if state.get('closed_for_admission'):
            raise ValueError('instance is closing; submit a new instance for new work')
        ids = []
        for spec in specs:
            prior = next((j for j in state['jobs'] if j['spec']['cell_id'] == spec['cell_id']), None)
            if prior:
                if prior['spec'] != spec:
                    raise ValueError('cell already admitted with a different configuration')
                ids.append(prior['id'])
                continue
            status = artifact_state(spec)
            if spec['cell_id'] in owners and status != 'complete':
                status = 'observing'
            if status == 'idle':
                status = 'failed' if observe else 'queued'
            job = dict(id=state['next_id'], spec=spec, priority=priority, attempt=0, gpu_ids=[], identity=None,
                       node=main_node(state['config']))
            state['next_id'] += 1
            state['jobs'].append(job)
            transition(state, job, status, reason='admitted; existing work observed without resubmission')
            ids.append(job['id'])
    return ids


def change(store, job_id, command, priority=0):
    observations = collect_nodes(store)
    owners = {cell: name for name, value in observations.items() for cell in value.get('owners', {})}
    with store.edit() as state:
        job = next(j for j in state['jobs'] if j['id'] == job_id)
        observation = observations[job.get('node', main_node(state['config']))]
        if observation.get('error'):
            raise ValueError('node unreachable; cannot safely change ownership')
        active = observation.get('workers', {}).get(str(job_id), False) or observation.get('artifacts', {}).get(str(job_id)) in ('observing', 'scoring')
        if command == 'remove':
            if job['state'] not in ('queued', 'failed', 'needs_review', 'cancelled'):
                raise ValueError('already started; removing a queued item does not stop active inference/evaluation')
            if active or job['spec']['cell_id'] in owners:
                raise ValueError('active process still owns this cell; removal will not stop it')
            transition(state, job, 'cancelled', reason='removed by user')
        elif command == 'priority':
            if job['state'] != 'queued':
                raise ValueError('only queued items can be reprioritized')
            transition(state, job, 'queued', priority=priority)
        elif command == 'retry':
            if state.get('closed_for_admission'):
                raise ValueError('instance is closing')
            if job['state'] not in ('failed', 'needs_review', 'cancelled'):
                raise ValueError('retry requires a failed/cancelled item; complete results need a new registered run')
            if active or job['spec']['cell_id'] in owners or observation.get('artifacts', {}).get(str(job_id)) != 'idle':
                raise ValueError('existing process, lock, or complete result still owns this cell')
            validate(job['spec'])
            validate_checkpoints(job['spec'])
            transition(state, job, 'queued', identity=None, gpu_ids=[], reason='explicit retry; reuse native predictions')


def query_gpus(gpu_ids):
    proc = subprocess.run(['nvidia-smi', '-i', ','.join(gpu_ids), '--query-gpu=index,memory.used',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    values = dict((p[0].strip(), int(p[1])) for line in proc.stdout.splitlines() if (p := line.split(',')))
    if set(values) != set(gpu_ids):
        raise ValueError('incomplete GPU query; refuse dispatch')
    return [gpu for gpu in gpu_ids if 0 <= values[gpu] <= 1024]


def update_estimates(state):
    """Reuse the viewer's source-backed remaining inference-time estimates."""
    filename = state['config'].get('estimate_file')
    try:
        payload = read(Path(filename)) if filename else {}
        estimates = {r['id']: r for r in payload.get('cells', [])}
    except (OSError, ValueError, KeyError, TypeError):
        estimates = {}
    for job in state['jobs']:
        if job['state'] != 'queued':
            continue
        e = estimates.get(job['spec']['cell_id'], {})
        seconds = e.get('remaining_s', e.get('full_s'))
        source_matches = (e.get('entry') == job['spec']['row']['model']
                          and e.get('gpus') == job['spec']['gpus']
                          and e.get('runtime_source')
                          and Path(e['runtime_source']).resolve() == (Path(job['spec']['registration'])/'models.yaml').resolve())
        valid = source_matches and isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds >= 0
        job['estimated_seconds'] = float(seconds) if valid else None
        job['estimate_source'] = dict(file=filename, updated_at=payload.get('updated_at'),
                                      kind=e.get('kind'), basis=e.get('progress_basis')) if valid else None


def choose(jobs, free_count):
    candidates = [j for j in jobs if j['state'] == 'queued' and j['spec']['gpus'] <= free_count]
    if not candidates:
        return None
    priority = max(j['priority'] for j in candidates)
    candidates = [j for j in candidates if j['priority'] == priority]
    duration = lambda j: j.get('estimated_seconds') if j.get('estimated_seconds') is not None else math.inf
    shortest = min(map(duration, candidates))
    candidates = [j for j in candidates if duration(j) == shortest]
    models = {j['spec']['model_key']: SimpleNamespace(gpus_per_job=j['spec']['gpus']) for j in jobs}
    order = list(models)
    tasks = [SimpleNamespace(model_key=j['spec']['model_key'], weight=j['spec']['weight'], index=j['id']) for j in candidates]
    view = SimpleNamespace(models=models, model_order=order)
    running = Counter(j['spec']['model_key'] for j in jobs if j['state'] in ('launching', 'running'))
    index = native.BenchmarkRunner._select_gpu_pool_task(view, tasks, free_count, running)
    return candidates[index]


def reconcile(state, owners, observations=None):
    for job in state['jobs']:
        status = job['state']
        if status in TERMINAL:
            continue
        node = job.get('node', main_node(state['config']) if 'config' in state else 'local')
        observation = observations.get(node, {}) if observations is not None else None
        if observation is not None and observation.get('error'):
            continue  # uncertain remote state never releases or resubmits a task
        if observation is not None and observation.get('versions', {}).get(str(job['id'])) != [job['attempt'], job['updated_at']]:
            continue  # RPC snapshot predates a concurrent admission/worker update
        worker_live = observation.get('workers', {}).get(str(job['id']), False) if observation is not None else alive(job.get('identity'))
        if status == 'running' and worker_live:
            continue
        if status == 'launching':
            if time.time() - job['updated_at'] > 30:
                transition(state, job, 'needs_review', reason='launch ACK missing; never auto-resubmit')
            continue
        actual = observation.get('artifacts', {}).get(str(job['id']), 'idle') if observation is not None else artifact_state(job['spec'])
        if actual == 'idle' and job['spec']['cell_id'] in owners:
            actual = 'observing'
            if observations is not None:
                job['node'] = owners[job['spec']['cell_id']]
        if actual == 'idle':
            if status != 'queued':
                transition(state, job, 'failed', gpu_ids=[], reason='owner exited without complete score; explicit retry required')
        elif actual != status:
            transition(state, job, actual, gpu_ids=[])


def tick(store, children):
    for pid, proc in list(children.items()):
        if proc.poll() is not None:
            del children[pid]
    observations = collect_nodes(store)
    owners = {cell: name for name, value in observations.items() for cell in value.get('owners', {})}
    with store.edit() as state:
        reconcile(state, owners, observations)
        update_estimates(state)
        state['scheduling_policy'] = 'shortest_remaining_inference_first'
        state['nodes'] = observations
        state['gpu_error'] = '; '.join(f"{n}: {p['error']}" for n, p in observations.items() if p.get('error')) or None
        state['heartbeat'] = time.time()
        state['free_gpus'] = []
        for name, observation in sorted(observations.items(), key=lambda item: -len(item[1].get('free', []))):
            reserved = {g for j in state['jobs'] if j.get('node', 'local') == name and j['state'] in ('launching', 'running', 'needs_review') for g in j.get('gpu_ids', [])}
            free = [g for g in observation.get('free', []) if g not in reserved]
            while not state.get('paused') and (job := choose(state['jobs'], len(free))) is not None:
                assigned = free[:job['spec']['gpus']]
                free = free[len(assigned):]
                job['attempt'] += 1
                transition(state, job, 'launching', gpu_ids=assigned, identity=None, node=name,
                           estimated_seconds=job.get('estimated_seconds'), estimate_source=job.get('estimate_source'))
            state['free_gpus'].extend(f'{name}:{g}' for g in free)
    # Claim is durable before Popen. A crash here becomes needs_review, not retry.
    for job in store.snapshot()['jobs']:
        if job['state'] != 'launching' or job.get('identity'):
            continue
        attempt = store.root/'attempts'/f"{job['id']}-{job['attempt']}"
        # Exclusive directory also fences a second spawn of a claimed attempt.
        try:
            attempt.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        with (attempt/'worker.log').open('a') as log:
            node = nodes(store.snapshot()['config'])[job['node']]
            args = ['--root', str(store.root), 'worker', '--id', str(job['id']), '--attempt', str(job['attempt'])]
            cmd = remote_command(node, [*args, '--detach']) if node.get('ssh') else [sys.executable, str(Path(__file__).resolve()), *args]
            proc = subprocess.Popen(cmd,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children[proc.pid] = proc


def worker(store, job_id, attempt):
    with store.edit() as state:
        job = next(j for j in state['jobs'] if j['id'] == job_id)
        if job['state'] != 'launching' or job['attempt'] != attempt:
            return
        transition(state, job, 'running', identity=identity(os.getpid()))
        config = state['config']
    spec = job['spec']
    root = Path(spec['path'])
    root.mkdir(parents=True, exist_ok=True)
    status, reason = 'failed', ''
    try:
        if store.snapshot().get('control_source_sha', CONTROL_SHA) != CONTROL_SHA:
            raise ValueError('controller source changed; do not mix execution-control versions')
        validate(spec)
        # Includes legacy controller pending lists, so a new queue cannot steal
        # a task already scheduled by an old fixed pool.
        actual = artifact_state(spec)
        if actual != 'idle' or spec['cell_id'] in external_owners():
            status = actual if actual != 'idle' else 'observing'
            reason = 'existing artifact/process owns cell; no new native launch'
        else:
            with (root/'.host-queue.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                directory = store.root/'attempts'/f'{job_id}-{attempt}'
                manifest = directory/'task.csv'
                with manifest.open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=list(spec['row']))
                    writer.writeheader(); writer.writerow(spec['row'])
                cmd = ['bash', config['launcher'], spec['run_id'], '--task-manifest', str(manifest),
                       '--gpu-ids', ','.join(job['gpu_ids']), '--nodes', '1', '--node-rank', '0',
                       '--gpu-ready-timeout', '600', '--atomic-checkpoints']
                with (directory/'native.log').open('a') as log:
                    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                            pass_fds=(lock.fileno(),))
                    rc = proc.wait()
                actual = artifact_state(spec)
                status = actual if rc == 0 and actual in ('complete', 'scoring') else 'failed'
                reason = f'native launcher rc={rc}; artifact={actual}'
    except Exception as exc:
        reason = f'{type(exc).__name__}: {exc}'
    with store.edit() as state:
        latest = next(j for j in state['jobs'] if j['id'] == job_id)
        if latest['attempt'] == attempt:
            transition(state, latest, status, reason=reason, gpu_ids=[])


def resume_instance(store, nodes_file=None, instance_id=None, previous_status=None, previous_task=None):
    """Reattach durable work only after the old controller and workers are gone."""
    with (store.root/'daemon.lock').open('a+') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = store.snapshot()
        old_config = saved['config']
        proof = None
        if nodes_file:
            proof = read(previous_status) if previous_status else {}
            reason = str(proof.get('status', {}).get('reason', '')).lower()
            terminal = {'completed','failed','stopped','killed','cancelled','canceled','succeeded','terminated','error','evicted','oomkilled'}
            env = {v['key']: v['value'] for v in proof.get('spec', {}).get('podCommon', {}).get('environments', [])}
            if (not previous_task or proof.get('meta', {}).get('name') != previous_task or reason not in terminal
                    or old_config.get('instance_id') != env.get('RUN_ID', '')+'/'+env.get('CTL_EXP_RUN_TS', '')
                    or not instance_id):
                raise ValueError('require matching terminal job-pack get JSON for the previous submission')
        else:
            observations = collect_nodes(store)
            own_cells = {j['spec']['cell_id'] for j in saved['jobs']}
            if any(p.get('error') or own_cells.intersection(p.get('owners', {})) or any(p.get('workers', {}).values()) for p in observations.values()):
                raise ValueError('previous node/workers remain live or unknown; cannot resume')
        for job in saved['jobs']:
            root = Path(job['spec']['path'])
            if locked(root/'.host-queue.lock') or locked(root/'eval/.worker.lock'):
                raise ValueError('a native inference or evaluator still owns a cell')
            validate(job['spec'])
            if artifact_state(job['spec']) != 'complete':
                validate_checkpoints(job['spec'])
        terminal = store.root/'terminal.json'
        if terminal.exists():
            terminal.rename(store.root/f'terminal-before-resume-{time.time_ns()}.json')
        with store.edit() as state:
            state.setdefault('resume_history', []).append(dict(config=state['config'].copy(),
                daemon=state.get('daemon'), at=time.time(), previous_task=previous_task,
                terminal_proof=dict(path=str(previous_status), sha256=hashlib.sha256(previous_status.read_bytes()).hexdigest(),
                                    reason=proof['status']['reason']) if proof else None))
            if nodes_file:
                proposed = read(nodes_file)
                if not proposed or sum(not n.get('ssh') for n in proposed.values()) != 1:
                    raise ValueError('replacement topology requires exactly one main node')
                if any(not n.get('gpu_ids') or len(set(n['gpu_ids'])) != len(n['gpu_ids']) or not all(g.isdigit() for g in n['gpu_ids']) or not isinstance(n.get('ssh', []), list) for n in proposed.values()):
                    raise ValueError('invalid replacement topology')
                state['config'].update(nodes=proposed, instance_id=instance_id)
            state.pop('node_identities', None)
            state.pop('daemon', None)
            state.pop('heartbeat', None)
            state.pop('nodes', None)
            state['paused'] = False
            state['closed_for_admission'] = False
            for job in state['jobs']:
                if artifact_state(job['spec']) == 'complete':
                    transition(state, job, 'complete', identity=None, gpu_ids=[])
                elif job['state'] != 'cancelled' and (job['state'] != 'failed'
                        or job.get('reason', '').startswith('owner exited without complete score')
                        or any(f'rc={rc};' in job.get('reason', '') for rc in (-9, -15, 137, 143))):
                    transition(state, job, 'queued', identity=None, gpu_ids=[], node=main_node(state['config']),
                               reason='explicit resume of stopped instance; preserve native prediction/judge caches')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--via', help='send this command to the existing main controller over SSH')
    ap.add_argument('--ssh-port', type=int, default=2222)
    ap.add_argument('--remote-script', default=str(Path(__file__).resolve()))
    ap.add_argument('--remote-python', default=sys.executable)
    sub = ap.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init'); init.add_argument('--registry', type=Path, required=True)
    init.add_argument('--launcher', type=Path, required=True); init.add_argument('--gpu-ids', default='0,1,2,3,4,5,6,7')
    init.add_argument('--nodes-file', type=Path); init.add_argument('--instance-id')
    add = sub.add_parser('add'); add.add_argument('--run-id', required=True); add.add_argument('--cell', action='append', default=[])
    add.add_argument('--all', action='store_true'); add.add_argument('--observe', action='store_true'); add.add_argument('--priority', type=int, default=0)
    listing = sub.add_parser('list'); listing.add_argument('--json', action='store_true')
    sub.add_parser('serve')
    sub.add_parser('close')
    sub.add_parser('pause'); sub.add_parser('unpause')
    resuming = sub.add_parser('resume')
    resuming.add_argument('--nodes-file', type=Path); resuming.add_argument('--instance-id')
    resuming.add_argument('--previous-status', type=Path); resuming.add_argument('--previous-task')
    estimating = sub.add_parser('estimates'); estimating.add_argument('--file', type=Path, required=True)
    probing = sub.add_parser('probe'); probing.add_argument('--node', required=True)
    for name in ('remove', 'retry', 'priority', 'worker'):
        p = sub.add_parser(name); p.add_argument('--id', type=int, required=True)
        if name == 'priority': p.add_argument('--value', type=int, required=True)
        if name == 'worker':
            p.add_argument('--attempt', type=int, required=True); p.add_argument('--detach', action='store_true')
    a = ap.parse_args()
    if a.via:
        forwarded = []
        remaining = iter(sys.argv[1:])
        for arg in remaining:
            if arg in ('--via', '--ssh-port', '--remote-script', '--remote-python'):
                next(remaining)
            elif not arg.startswith(('--via=', '--ssh-port=', '--remote-script=', '--remote-python=')):
                forwarded.append(arg)
        return subprocess.call(remote_command({'ssh':['ssh','-p',str(a.ssh_port),'-o','BatchMode=yes',a.via],
                                                'python':a.remote_python, 'script':a.remote_script}, forwarded))
    store = Store(a.root.resolve())
    if a.command == 'init':
        with store.edit() as state:
            config = dict(registry=str(a.registry.resolve()), launcher=str(a.launcher.resolve()), gpu_ids=a.gpu_ids.split(','))
            if a.nodes_file:
                config['nodes'] = read(a.nodes_file)
                config['instance_id'] = a.instance_id or str(store.root)
                if not config['nodes'] or sum(not n.get('ssh') for n in config['nodes'].values()) != 1:
                    raise ValueError('nodes must include the main local node')
                for node in config['nodes'].values():
                    if not isinstance(node.get('ssh', []), list) or not node.get('gpu_ids') or len(set(node['gpu_ids'])) != len(node['gpu_ids']) or not all(g.isdigit() for g in node['gpu_ids']):
                        raise ValueError('nodes require argv SSH commands and distinct numeric GPU IDs')
            if (not a.registry.is_dir() or not a.launcher.is_file()
                    or not all(g.isdigit() for g in config['gpu_ids'])
                    or len(set(config['gpu_ids'])) != len(config['gpu_ids'])):
                raise ValueError('require existing registry/launcher and distinct numeric GPU IDs')
            if 'config' in state and state['config'] != config:
                raise ValueError('queue already initialized with different host settings')
            state['config'] = config
    elif a.command == 'probe':
        print(json.dumps(probe(store, a.node)))
    elif a.command == 'close':
        with store.edit() as state: state['closed_for_admission'] = True
    elif a.command in ('pause','unpause'):
        with store.edit() as state: state['paused'] = a.command == 'pause'
    elif a.command == 'resume':
        resume_instance(store, a.nodes_file, a.instance_id, a.previous_status, a.previous_task)
    elif a.command == 'estimates':
        payload = read(a.file)
        if not isinstance(payload.get('cells'), list):
            raise ValueError('expected the viewer runtime-estimates.json export')
        with store.edit() as state:
            state['config']['estimate_file'] = str(a.file.resolve())
            update_estimates(state)
    elif a.command == 'add':
        specs = registered_specs(store.snapshot()['config'], a.run_id, a.cell, a.all)
        print(json.dumps(enqueue(store, specs, a.priority, a.observe)))
    elif a.command in ('remove', 'retry', 'priority'):
        change(store, a.id, a.command, getattr(a, 'value', 0))
    elif a.command == 'list':
        state = store.snapshot()
        if a.json:
            print(json.dumps(state, ensure_ascii=False, indent=2))
        else:
            print('ID\tSTATE\tPRIORITY\tGPUS\tCELL')
            for job in state.get('jobs', []):
                print(f"{job['id']}\t{job['state']}\t{job['priority']}\t{','.join(job['gpu_ids']) or '-'}\t{job['spec']['cell_id']}")
    elif a.command == 'worker':
        if a.detach:
            marker = store.root/'attempts'/f'{a.id}-{a.attempt}'/'remote-started'
            with marker.open('x') as f: f.write(str(time.time()))
            with marker.with_suffix('.log').open('a') as log:
                subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--root', str(store.root),
                                  'worker', '--id', str(a.id), '--attempt', str(a.attempt)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        else:
            worker(store, a.id, a.attempt)
    elif a.command == 'serve':
        with (store.root/'daemon.lock').open('a+') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            token = store.root/'shared-token'
            if not token.exists(): token.write_text(os.urandom(32).hex())
            observations = collect_nodes(store)
            for name, result in observations.items():
                if result.get('error') or result['shared_token'] != token.read_text() or not result['daemon_locked']:
                    raise ValueError(f'node {name} shared storage/lock/source/SSH preflight failed: {result}')
            if len({(p['host'], p['boot']) for p in observations.values()}) != len(observations):
                raise ValueError('multiple node endpoints resolve to the same host identity')
            with store.edit() as state:
                state['node_identities'] = {n: [p['host'], p['boot']] for n, p in observations.items()}
                state['control_source_sha'] = CONTROL_SHA
            with store.edit() as state: state['daemon'] = identity(os.getpid())
            children = {}
            while True:
                tick(store, children)
                state = store.snapshot()
                quiescent = all(not p.get('error') and not p.get('active_locks') and not any(p.get('workers', {}).values()) for p in state['nodes'].values())
                if state.get('closed_for_admission') and quiescent and all(j['state'] in TERMINAL for j in state['jobs']):
                    result = dict(closed=True, failed=any(j['state'] in ('failed', 'needs_review') for j in state['jobs']))
                    temporary = store.root/'terminal.next.json'
                    temporary.write_text(json.dumps(result)+'\n')
                    os.replace(temporary, store.root/'terminal.json')
                    return int(result['failed'])
                time.sleep(2)


if __name__ == '__main__':
    raise SystemExit(main())
