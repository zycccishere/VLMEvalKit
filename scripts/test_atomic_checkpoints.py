"""Kill-during-write tests for prediction and judge-cache persistence."""
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
import unittest
from atomic_checkpoints import atomic_writer

class AtomicTests(unittest.TestCase):
    def test_inference_and_judge_cache_survive_kill_during_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for phase in ['inference','judge']:
                with self.subTest(phase=phase):
                    target=root/(phase+'.pkl');target.write_bytes(pickle.dumps({0:'saved0',1:'saved1'}))
                    marker=root/(phase+'.writing')
                    code='''import sys,time,pickle
from pathlib import Path
from atomic_checkpoints import atomic_writer
checkpoint,marker=sys.argv[1:]
def slow(data,path):
    with open(path,'wb') as f:
        f.write(b'partial pickle');f.flush()
        Path(marker).touch()
        time.sleep(60)
atomic_writer(slow)({2:'new'},checkpoint)
'''
                    p=subprocess.Popen([sys.executable,'-c',code,str(target),str(marker)],cwd=Path(__file__).parent)
                    end=time.time()+5
                    while not marker.exists() and time.time()<end:time.sleep(.02)
                    self.assertTrue(marker.exists());p.kill();p.wait(timeout=5)
                    self.assertEqual(pickle.loads(target.read_bytes()),{0:'saved0',1:'saved1'})
                    # The native resume contract skips keys already in its cache.
                    cache=pickle.loads(target.read_bytes());called=[]
                    for index in range(4):
                        if index not in cache:
                            called.append(index);cache[index]='new'+str(index)
                            atomic_writer(lambda data,path:Path(path).write_bytes(pickle.dumps(data)))(cache,target)
                    self.assertEqual(called,[2,3]);self.assertEqual(len(pickle.loads(target.read_bytes())),4)

    def test_import_hook_wraps_native_dump_before_alias_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);package=root/'vlmeval/smp';package.mkdir(parents=True)
            (root/'vlmeval/__init__.py').write_text('')
            (package/'__init__.py').write_text('from .file import dump\n')
            (package/'file.py').write_text("import pickle\ndef dump(data,path,**kwargs):\n pickle.dump(data,open(path,'wb'))\n")
            (root/'run.py').write_text("from vlmeval.smp import dump\nassert dump.__module__ == '__main__'\ndump({1:'ok'},'result.pkl')\n")
            proc=subprocess.run([sys.executable,str(Path(__file__).with_name('atomic_checkpoints.py')),str(root/'run.py')],cwd=root,capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertIn('atomic-fsync-replace-v1',proc.stdout)
            self.assertEqual(pickle.loads((root/'result.pkl').read_bytes()),{1:'ok'})

    def test_serialization_failure_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'cache.pkl';target.write_bytes(pickle.dumps({'old':1}))
            def broken(data,path):
                Path(path).write_bytes(b'broken');raise ValueError('writer failed')
            with self.assertRaises(ValueError):atomic_writer(broken)({},target)
            self.assertEqual(pickle.loads(target.read_bytes()),{'old':1})

if __name__=='__main__':unittest.main()
