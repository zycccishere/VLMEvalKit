"""CPU tests: editable host queue -> real native runner -> detached evaluator."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import shlex
from unittest.mock import patch

import yaml
import bound_queue as q
from test_bound_runner import FAKE


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='host-queue-')
        self.root = Path(self.tmp.name)
        self.repo = self.root/'execution'
        (self.repo/'vlmeval/dataset').mkdir(parents=True)
        (self.repo/'vlmeval/__init__.py').write_text('')
        (self.repo/'vlmeval/dataset/__init__.py').write_text('import pandas as pd\nclass D:\n data=pd.DataFrame({"index":[0]})\ndef build_dataset(name):return D()\n')
        (self.repo/'run.py').write_text(FAKE)
        self.bin = self.root/'bin'; self.bin.mkdir()
        smi=self.bin/'nvidia-smi'
        smi.write_text('#!/usr/bin/env python3\nimport sys\nids=sys.argv[sys.argv.index("-i")+1].split(",")\nprint("\\n".join(g+", 0" if "--query-gpu=index,memory.used" in sys.argv else "0" for g in ids))\n')
        smi.chmod(0o755)
        self.env = patch.dict(os.environ, {'PATH':str(self.bin)+':'+os.environ['PATH'], 'LMUData':str(self.root)})
        self.env.start()
        self.registry=self.root/'registry'; self.registry.mkdir()
        self.launcher=self.root/'launch.sh'
        # Exercise native CLI and per-cell CSV, not a replacement executor.
        self.launcher.write_text('#!/bin/bash\nset -eu\nrun="$1"\nshift\nexec '+sys.executable+' '+str(q.native.__file__)+' --matrix-config '+str(self.registry)+'/$run/matrix.yaml --model-config '+str(self.registry)+'/$run/models.yaml "$@"\n')
        self.store=q.Store(self.root/'queue')
        with self.store.edit() as s:s['config']=dict(registry=str(self.registry),launcher=str(self.launcher),gpu_ids=['0','1'])
        self.children={}

    def tearDown(self):
        (self.repo/'release').touch()
        for p in self.children.values():
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        self.env.stop();self.tmp.cleanup()

    def register(self, run, dataset):
        d=self.registry/run;d.mkdir()
        cell=f'final|m|VT|{dataset}';tag=f'm__default__image_text__baseline__{dataset}'
        matrix=dict(name=run,repo_root=str(self.repo),results_root=str(self.root/run),
            node_gpu_ids='0,1',scheduler='gpu_pool',datasets=[dataset],policies={'default':{'replay_prompt_template_name':'identity'}},
            replay_modes=['image_text'],image_transforms=['baseline'],models=['m'],
            evaluation={'launch_mode':'bg','judge':'gpt-4o-mini','nproc':512},
            answer_format={'enable':False},worker_monitor={'enable':False},
            experiment={'run_id':run,'bindings':{tag:{'cell_id':cell}}})
        models={'env_profiles':{'cpu':{'python':sys.executable,'pythonpath':[str(self.repo)]}},
                'models':{'m':{'display_name':'Toy','registry_name':'Toy','env_profile':'cpu','model_path':'/unused',
                              'runtime':{'gpus_per_job':1,'infer_batch_size':1,'max_num_seqs':1,'tp_size':1,'estimated_dataset_cost':1}}}}
        for name,data in [('matrix.yaml',matrix),('models.yaml',models)]:(d/name).write_text(yaml.safe_dump(data))
        row=dict(model='m',policy='default',mode='image_text',transform='baseline',dataset=dataset,cell_id=cell)
        with (d/'tasks.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
        source=self.root/run/'default/image_text/baseline/m'/dataset
        reg=dict(sources={cell:{'path':str(source)}},prepared_files={n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in ['matrix.yaml','models.yaml','tasks.csv']})
        (d/'registration.json').write_text(json.dumps(reg))
        return q.registered_specs(self.store.snapshot()['config'],run,[cell])[0]

    def until(self, check):
        end=time.time()+20
        while time.time()<end:
            q.tick(self.store,self.children)
            if check():return
            time.sleep(.1)
        self.fail(json.dumps(self.store.snapshot(),indent=2))

    def test_dynamic_add_remove_cross_run_gpu_reuse_and_background_eval(self):
        a=self.register('left','A');b=self.register('right','B');c=self.register('cancel','C')
        self.assertEqual(q.enqueue(self.store,[a]),[1])
        q.tick(self.store,self.children)
        self.until(lambda:self.store.snapshot()['jobs'][0]['state']=='scoring')
        self.assertFalse((self.repo/'release').exists())
        q.enqueue(self.store,[b,c]);q.change(self.store,3,'remove')
        self.until(lambda:self.store.snapshot()['jobs'][1]['state']=='scoring')
        state=self.store.snapshot();self.assertEqual(state['jobs'][2]['state'],'cancelled')
        self.assertFalse(Path(c['path']).exists())
        # Another run uses GPU0 immediately, while A's detached judge remains alive.
        dispatches=[e for e in state['events'] if e['state']=='launching']
        self.assertEqual([e['details']['gpu_ids'] for e in dispatches],[['0'],['0']])
        self.assertEqual(q.enqueue(self.store,[a,b]),[1,2])
        (self.repo/'release').touch()
        self.until(lambda:all(j['state']=='complete' for j in self.store.snapshot()['jobs'][:2]))
        for ds in ['A','B']:self.assertEqual(len((self.repo/(ds+'.starts')).read_text().splitlines()),1)
        for job in self.store.snapshot()['jobs'][:2]:
            for phase in ['predictions','eval']:
                self.assertEqual(q.read(Path(job['spec']['path'])/phase/'manifest.json')['checkpoint_io'],'atomic-fsync-replace-v1')

    def test_busy_cards_priority_and_no_duplicate_after_scheduler_restart(self):
        a=self.register('left','A');b=self.register('right','B');q.enqueue(self.store,[a,b])
        q.change(self.store,2,'priority',10)
        with patch.object(q,'query_gpus',return_value=[]):q.tick(self.store,self.children)
        self.assertTrue(all(j['state']=='queued' for j in self.store.snapshot()['jobs']))
        with patch.object(q,'query_gpus',return_value=['1']):q.tick(self.store,self.children)
        state=self.store.snapshot();self.assertEqual(state['jobs'][1]['state'],'launching')
        self.until(lambda:all(j['state']=='scoring' for j in self.store.snapshot()['jobs']))
        fresh=q.Store(self.store.root);q.tick(fresh,{})
        self.assertEqual(len([e for e in fresh.snapshot()['events'] if e['state']=='launching']),2)
        with self.assertRaises(ValueError):q.change(fresh,2,'remove')
        (self.repo/'release').touch()
        self.until(lambda:all(j['state']=='complete' for j in self.store.snapshot()['jobs']))

    def test_failed_judge_retry_preserves_predictions_and_config_drift_is_rejected(self):
        spec=self.register('failure','Fail');q.enqueue(self.store,[spec]);(self.repo/'release').touch()
        self.until(lambda:self.store.snapshot()['jobs'][0]['state']=='failed')
        pred=Path(spec['path'])/'predictions/Toy_Fail.tsv';before=pred.read_bytes()
        q.change(self.store,1,'retry');self.until(lambda:self.store.snapshot()['jobs'][0]['state']=='failed')
        self.assertEqual(pred.read_bytes(),before)
        self.assertEqual(self.store.snapshot()['jobs'][0]['attempt'],2)
        path=Path(spec['registration'])/'models.yaml';path.write_text(path.read_text()+'\n#changed\n')
        with self.assertRaises(ValueError):q.change(self.store,1,'retry')

    def test_native_packing_and_missing_launch_ack(self):
        jobs=[dict(id=i,attempt=1,state='queued',priority=0,spec=dict(model_key=str(i),gpus=width,weight=1)) for i,width in [(1,4),(2,2),(3,1)]]
        self.assertEqual(q.choose(jobs,3)['id'],2)
        self.assertEqual(q.choose(jobs,1)['id'],3)
        self.assertIsNone(q.choose(jobs,0))
        jobs[0].update(state='launching',updated_at=time.time()-60)
        state=dict(jobs=[jobs[0]],events=[]);q.reconcile(state,{})
        self.assertEqual(state['jobs'][0]['state'],'needs_review')

    def test_shortest_estimated_runtime_with_capacity_priority_and_unknowns(self):
        jobs=[dict(id=i, state='queued', priority=0, estimated_seconds=seconds,
                   spec=dict(model_key=str(i),gpus=gpus,weight=1))
              for i,seconds,gpus in [(1,600,4),(2,60,1),(3,None,1),(4,10,8)]]
        self.assertEqual(q.choose(jobs,4)['id'],2)
        self.assertEqual(q.choose(jobs,8)['id'],4)
        jobs[1]['state']='running'
        self.assertEqual(q.choose(jobs,4)['id'],1)
        jobs[2]['priority']=10
        self.assertEqual(q.choose(jobs,4)['id'],3)

    def test_estimates_reuse_remaining_time_and_reject_wrong_model_configuration(self):
        spec=self.register('left','A');q.enqueue(self.store,[spec])
        p=self.root/'runtime-estimates.json'
        estimate=dict(id=spec['cell_id'],entry='m',gpus=1,runtime_source=str(Path(spec['registration'])/'models.yaml'),
                      remaining_s=20,full_s=600,kind='estimated')
        p.write_text(json.dumps({'cells':[estimate]}))
        with self.store.edit() as state:
            state['config']['estimate_file']=str(p);q.update_estimates(state)
            self.assertEqual(state['jobs'][0]['estimated_seconds'],20)
            estimate['runtime_source']='/different/models.yaml';p.write_text(json.dumps({'cells':[estimate]}));q.update_estimates(state)
            self.assertIsNone(state['jobs'][0]['estimated_seconds'])

    def test_resume_preserves_inference_and_judge_checkpoints(self):
        import pickle
        spec=self.register('left','A');q.enqueue(self.store,[spec]);root=Path(spec['path']);root.mkdir(parents=True)
        checkpoints=[root/'prediction-partial.pkl',root/'judge-partial.pkl']
        for p in checkpoints:p.write_bytes(pickle.dumps({0:'committed'}))
        before=[p.read_bytes() for p in checkpoints]
        with self.store.edit() as state:
            state['jobs'][0].update(state='scoring',attempt=2)
            state['paused']=True
        q.resume_instance(self.store)
        s=self.store.snapshot();self.assertEqual(s['jobs'][0]['state'],'queued');self.assertEqual(s['jobs'][0]['attempt'],2)
        self.assertEqual([p.read_bytes() for p in checkpoints],before)
        checkpoints[0].write_bytes(b'truncated')
        with self.assertRaises(ValueError):q.resume_instance(self.store)

    def test_platform_resume_requires_previous_terminal_identity(self):
        spec=self.register('left','A');q.enqueue(self.store,[spec])
        with self.store.edit() as state:state['config']['instance_id']='run/old'
        nodes=self.root/'nodes.json';nodes.write_text(json.dumps({'new-main':{'gpu_ids':['0','1'],'ssh':[]}}))
        receipt=self.root/'platform.json';value={'meta':{'name':'task-old'},'status':{'reason':'Running'},
                'spec':{'podCommon':{'environments':[{'key':'RUN_ID','value':'run'},{'key':'CTL_EXP_RUN_TS','value':'old'}]}}}
        receipt.write_text(json.dumps(value))
        with self.assertRaises(ValueError):q.resume_instance(self.store,nodes,'run/new',receipt,'task-old')
        value['status']['reason']='Stopped';receipt.write_text(json.dumps(value))
        q.resume_instance(self.store,nodes,'run/new',receipt,'task-old')
        s=self.store.snapshot();self.assertEqual(s['config']['instance_id'],'run/new');self.assertEqual(len(s['resume_history']),1)

    def test_worker_reports_via_files_and_never_touches_ledger(self):
        spec=self.register('left','A');q.enqueue(self.store,[spec])
        with self.store.edit() as state:state['jobs'][0].update(state='launching',attempt=1,gpu_ids=['0'],node='local')
        (self.store.root/'attempts/1-1').mkdir(parents=True)
        before=(self.store.root/'state.json').read_bytes()
        q.worker(self.store,1,1)
        self.assertEqual((self.store.root/'state.json').read_bytes(),before)  # single-writer: ledger untouched
        directory=self.store.root/'attempts/1-1'
        self.assertTrue((directory/'started.json').exists() and (directory/'outcome.json').exists())
        with self.store.edit() as state:q.apply_attempt_records(state,self.store.root)
        job=self.store.snapshot()['jobs'][0]
        self.assertIn(job['state'],('running','scoring','complete'))

    def test_edit_refuses_to_recreate_ledger_on_initialized_root(self):
        # Regression: a worker whose volume view lacked state.json once recreated
        # an empty ledger, destroying config and crashing the main controller.
        (self.store.root/'shared-token').write_text('bootstrap-token')
        (self.store.root/'state.json').unlink()
        with self.assertRaises(ValueError):self.store.edit().__enter__()
        (self.store.root/'state.json').write_text('')
        with self.assertRaises(ValueError):self.store.edit().__enter__()

    def test_apply_attempt_records_guards_stale_and_bogus_outcomes(self):
        spec=self.register('left','A');q.enqueue(self.store,[spec])
        directory=self.store.root/'attempts/1-2';directory.mkdir(parents=True)
        q.atomic_json(directory/'outcome.json',dict(status='complete',reason='stale attempt',attempt=1))
        with self.store.edit() as state:
            state['jobs'][0].update(state='running',gpu_ids=['0'],node='local',identity=q.identity(os.getpid()))
            q.apply_attempt_records(state,self.store.root)
            self.assertEqual(state['jobs'][0]['state'],'running')  # attempt mismatch ignored
        q.atomic_json(directory/'outcome.json',dict(status='skipped',reason='gate failed',attempt=2))
        with self.store.edit() as state:state['jobs'][0].update(state='running',attempt=2)
        self.until(lambda:self.store.snapshot()['jobs'][0]['state']=='needs_review')
        reason=self.store.snapshot()['jobs'][0].get('reason','')
        self.assertIn('gate failed',reason)
        directory2=self.store.root/'attempts/1-3';directory2.mkdir(parents=True)
        q.atomic_json(directory2/'outcome.json',dict(status='weird-status',attempt=3))
        with self.store.edit() as state:
            state['jobs'][0].update(state='launching',attempt=3,gpu_ids=['0'],node='local')
            q.apply_attempt_records(state,self.store.root)
            self.assertEqual(state['jobs'][0]['state'],'needs_review')

    def test_transient_owner_sighting_requeues_never_dispatched_cell(self):
        # Regression (platform 20261109T140450Z): a seconds-long plan/verify
        # child carrying the registration-wide manifest made external_owners
        # claim queued cells; when it vanished, reconcile failed them. A
        # never-dispatched (attempt==0) observing cell must return to queued.
        spec=self.register('left','A');q.enqueue(self.store,[spec])
        def snap(job):return {'local':{'error':None,'versions':{str(job['id']):[job['attempt'],job['updated_at']]},
                                   'workers':{},'artifacts':{str(job['id']):'idle'}}}
        with self.store.edit() as state:
            job=state['jobs'][0]
            q.reconcile(state,{spec['cell_id']:'local'},snap(job))
            self.assertEqual(job['state'],'observing')
            q.reconcile(state,{},snap(job))  # owner vanished, artifact idle
            self.assertEqual(job['state'],'queued')
            self.assertEqual(job['attempt'],0)

    def test_concurrent_admission_and_legacy_pending_owner(self):
        spec=self.register('left','A')
        cmd=[sys.executable,str(Path(q.__file__).resolve()),'--root',str(self.store.root),
             'add','--run-id','left','--all']
        procs=[subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
        for proc in procs:
            out,err=proc.communicate(timeout=10)
            self.assertEqual(proc.returncode,0,err);self.assertEqual(json.loads(out),[1])
        self.assertEqual(len(self.store.snapshot()['jobs']),1)
        with patch.object(q,'external_owners',return_value={spec['cell_id']:q.identity(os.getpid())}):
            q.tick(self.store,self.children)
            self.assertEqual(self.store.snapshot()['jobs'][0]['state'],'observing')
            self.assertFalse(self.children)
            with self.assertRaises(ValueError):q.change(self.store,1,'retry')

    def test_remote_protocol_detaches_and_main_can_receive_commands(self):
        spec=self.register('remote-run','A')
        # Simulate the SSH transport only; exercise real command quoting,
        # separate processes, shared ledger/locks and the native evaluator.
        ssh=self.bin/'ssh'
        ssh.write_text('#!/usr/bin/env python3\nimport subprocess,sys\nraise SystemExit(subprocess.call(sys.argv[-1],shell=True))\n')
        ssh.chmod(0o755)
        with self.store.edit() as state:
            state['config']['nodes']={'main':{'gpu_ids':['0'],'ssh':[]},
                'worker':{'gpu_ids':['1'],'ssh':[str(ssh),'fixture-worker']}}
            state['config']['instance_id']='fixture/submission1'
        q.enqueue(self.store,[spec])
        original=q.query_gpus
        with patch.object(q,'query_gpus',side_effect=lambda ids:[] if ids==['0'] else original(ids)):
            q.tick(self.store,self.children)
        self.until(lambda:self.store.snapshot()['jobs'][0]['state']=='scoring')
        job=self.store.snapshot()['jobs'][0];self.assertEqual(job['node'],'worker')
        receipt=self.store.root/'attempts/1-1/remote-started'
        self.assertTrue(receipt.exists())
        self.assertTrue(all(p.poll() is not None for p in self.children.values()))
        cmd=[sys.executable,str(Path(q.__file__).resolve()),'--root',str(self.store.root),
             '--via','fixture-main','list','--json']
        remote=json.loads(subprocess.check_output(cmd,text=True))
        self.assertEqual(remote['config']['instance_id'],'fixture/submission1')
        (self.repo/'release').touch()
        self.until(lambda:self.store.snapshot()['jobs'][0]['state']=='complete')

    def test_unreachable_node_retains_claim_and_close_blocks_admission(self):
        spec=self.register('left','A');q.enqueue(self.store,[spec])
        with self.store.edit() as state:
            job=state['jobs'][0];job.update(state='running',node='local',gpu_ids=['0'],identity=q.identity(os.getpid()))
            q.reconcile(state,{}, {'local':{'error':'SSH unavailable'}})
            self.assertEqual(job['state'],'running');self.assertEqual(job['gpu_ids'],['0'])
            state['closed_for_admission']=True
        with self.assertRaises(ValueError):q.enqueue(self.store,[spec])
        with patch.object(q,'collect_nodes',return_value={'local':{'error':'SSH unavailable'}}):
            with self.assertRaises(ValueError):q.change(self.store,1,'retry')

    def test_remote_command_preserves_arguments(self):
        args=['--root','/a path', 'add', '--cell', '最终协议|模型|spaces|$(not-a-command);x']
        command=q.remote_command({'ssh':['ssh','main']},args)
        self.assertEqual(shlex.split(command[-1])[2:],args)

    def test_stale_rpc_cannot_release_a_new_worker_reservation(self):
        spec=self.register('left','A');q.enqueue(self.store,[spec])
        with self.store.edit() as state:
            job=state['jobs'][0]
            stale={'local':{'versions':{'1':[job['attempt'],job['updated_at']]},'workers':{},'artifacts':{'1':'idle'}}}
            q.transition(state,job,'running',identity=q.identity(os.getpid()),gpu_ids=['0'])
            q.reconcile(state,{},stale)
            self.assertEqual(job['state'],'running');self.assertEqual(job['gpu_ids'],['0'])

    def test_instance_stays_open_when_empty_then_drains_on_close(self):
        cmd=[sys.executable,str(Path(q.__file__).resolve()),'--root',str(self.store.root)]
        proc=subprocess.Popen([*cmd,'serve'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            end=time.time()+10
            while time.time()<end and not self.store.snapshot().get('heartbeat'):
                time.sleep(.05)
            self.assertIn('local',self.store.snapshot()['node_identities'])
            self.assertIsNone(proc.poll())
            subprocess.run([*cmd,'close'],check=True)
            out,err=proc.communicate(timeout=10)
            self.assertEqual(proc.returncode,0,err)
            self.assertEqual(q.read(self.store.root/'terminal.json'),{'closed':True,'failed':False})
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=5)


if __name__=='__main__':unittest.main()
