"""CPU behavioral tests of native scheduling/child processes; no model/API calls."""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import yaml

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_test', P/'vlmeval/cli/run_benchmark.py')
native = importlib.util.module_from_spec(spec); sys.modules[spec.name] = native; spec.loader.exec_module(native)

FAKE = '''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data');p.add_argument('--model');p.add_argument('--mode');p.add_argument('--pred-output-dir');p.add_argument('--eval-dir');p.add_argument('--api-nproc');p.add_argument('--judge');a,_=p.parse_known_args()
root=Path(__file__).parent
if a.mode=='infer':
    d=Path(a.pred_output_dir);d.mkdir(parents=True,exist_ok=True)
    (d/(a.model+'_'+a.data+'.tsv')).write_text('index\\tprediction\\n0\\tA\\n')
else:
    assert a.api_nproc=='512' and a.judge=='gpt-4o-mini'
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    assert 'LOCAL_LLM' not in os.environ and 'VLMEVAL_OPENAI_TEMPERATURE' not in os.environ
    with (root/(a.data+'.starts')).open('a') as f:f.write(str(os.getpid())+'\\n')
    (root/(a.data+'.env.json')).write_text(json.dumps({k:v for k,v in os.environ.items() if k.startswith('VLMEVAL_DYNAMIC_CONCURRENCY_')}))
    end=time.time()+15
    while not (root/'release').exists() and time.time()<end:time.sleep(.02)
    if a.data=='Fail':raise SystemExit(7)
    d=Path(a.eval_dir);d.mkdir(parents=True,exist_ok=True)
    if a.data!='NoScore':(d/(a.model+'_'+a.data+'_acc.csv')).write_text('split,Overall\\nall,1.0\\n')
'''


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='bound-runner-')
        self.root = Path(self.tmp.name)
        self.repo = self.root/'execution';(self.repo/'vlmeval/dataset').mkdir(parents=True)
        (self.repo/'vlmeval/__init__.py').write_text('')
        (self.repo/'vlmeval/dataset/__init__.py').write_text('import pandas as pd\nclass D:\n    data=pd.DataFrame({"index":[0]})\ndef build_dataset(name):return D()\n')
        (self.repo/'run.py').write_text(FAKE)
        os.environ.setdefault('LMUData',str(self.root))

    def tearDown(self):
        (self.repo/'release').touch()
        time.sleep(.2)
        self.tmp.cleanup()

    def runner(self, datasets):
        bindings={f'm__default__image_text__baseline__{ds}':{'cell_id':f'final|m|VT|{ds}'} for ds in datasets}
        matrix=dict(name='test',repo_root=str(self.repo),results_root=str(self.root/'results'),
            node_gpu_ids='0',scheduler='gpu_pool',datasets=datasets,policies={'default':{'replay_prompt_template_name':'identity'}},
            replay_modes=['image_text'],image_transforms=['baseline'],models=['m'],
            evaluation={'launch_mode':'bg','judge':'gpt-4o-mini','nproc':512},
            answer_format={'enable':False},worker_monitor={'enable':False},
            experiment={'run_id':'test-run','bindings':bindings})
        models={'env_profiles':{'cpu':{'python':sys.executable,'pythonpath':[str(self.repo)],'base_env':{'LOCAL_LLM':'must-be-removed','VLMEVAL_OPENAI_TEMPERATURE':'omit'}}},
            'models':{'m':{'display_name':'Toy','registry_name':'Toy','env_profile':'cpu','model_path':'/not-used',
                          'runtime':{'gpus_per_job':1,'infer_batch_size':1,'max_num_seqs':1,'tp_size':1,'estimated_dataset_cost':1}}}}
        for name,d in [('matrix.yaml',matrix),('models.yaml',models)]: (self.root/name).write_text(yaml.safe_dump(d))
        a=argparse.Namespace(matrix_config=self.root/'matrix.yaml',model_config=self.root/'models.yaml',task_manifest=None,
            nodes=1,node_rank=0,gpu_ids='',models='',policies='',modes='',transforms='',datasets='',scheduler='',
            manifest_is_node_shard=False,resume_infer=None,plan_only=False)
        return native.BenchmarkRunner(P,a)

    def until(self, check):
        end=time.time()+12
        while time.time()<end:
            if check():return
            time.sleep(.03)
        self.fail('timed out waiting for subprocess evidence')

    def test_gpu_released_while_eval_running_and_duplicate_is_skipped(self):
        r=self.runner(['A','B']);self.assertEqual(r.run(),0)
        for t in r.tasks:self.assertTrue(r.infer_complete(t,r.models['m'],1))
        self.until(lambda:all((self.repo/(ds+'.starts')).exists() for ds in ['A','B']))
        self.assertFalse((self.repo/'release').exists())
        for t in r.tasks:r.run_single_task(r.models['m'],t,['0'])
        for ds in ['A','B']:self.assertEqual(len((self.repo/(ds+'.starts')).read_text().splitlines()),1)
        envs=[json.loads((self.repo/(ds+'.env.json')).read_text()) for ds in ['A','B']]
        self.assertNotEqual(envs[0]['VLMEVAL_DYNAMIC_CONCURRENCY_FILE'],envs[1]['VLMEVAL_DYNAMIC_CONCURRENCY_FILE'])
        (self.repo/'release').touch()
        self.until(lambda:all(r.eval_complete(t,r.models['m'],1) for t in r.tasks))
        for t in r.tasks:self.assertEqual(r._read_json(r.eval_manifest_path(t))['binding']['cell_id'],f'final|m|VT|{t.dataset}')

    def test_failed_eval_and_exit_zero_without_scores_are_not_complete(self):
        r=self.runner(['Fail','NoScore']);(self.repo/'release').touch();self.assertEqual(r.run(),0)
        self.until(lambda:all((r.eval_output_dir(t)/'status.json').exists() and r._read_json(r.eval_output_dir(t)/'status.json').get('status')=='failed' for t in r.tasks))
        for t in r.tasks:
            self.assertFalse(r.eval_complete(t,r.models['m'],1))
            self.assertTrue(r.infer_complete(t,r.models['m'],1))

    def test_gpu_guard_waits_for_all_assigned_cards(self):
        r=self.runner(['A']);r.gpu_ready_timeout=10
        responses=[native.subprocess.CompletedProcess([],0,'20000\n100\n'),
                   native.subprocess.CompletedProcess([],0,'100\n100\n')]
        with patch.object(native.subprocess,'run',side_effect=responses) as query, \
             patch.object(native.time,'sleep') as sleep:
            self.assertTrue(r.wait_for_gpu_ready('6,7'))
            self.assertEqual(query.call_count,2);sleep.assert_called_once()
            self.assertEqual(query.call_args.args[0][2],'6,7')

    def test_gpu_guard_timeout_never_launches_inference(self):
        r=self.runner(['A']);r.gpu_ready_timeout=1
        with patch.object(native.subprocess,'run',return_value=native.subprocess.CompletedProcess([],0,'70000\n')), \
             patch.object(native.time,'monotonic',side_effect=[0,0,2]), \
             patch.object(r,'run_subprocess') as launch:
            self.assertEqual(r.run_infer(r.tasks[0],r.models['m'],{'CUDA_VISIBLE_DEVICES':'7'},1),75)
            launch.assert_not_called()
        status=r._read_json(r.prediction_dir(r.tasks[0])/'status.json')
        self.assertEqual(status['reason'],'gpu_readiness_timeout')

    def test_gpu_guard_disabled_does_not_query(self):
        r=self.runner(['A'])
        with patch.object(native.subprocess,'run') as query:
            self.assertTrue(r.wait_for_gpu_ready('7'));query.assert_not_called()


if __name__=='__main__':unittest.main()
