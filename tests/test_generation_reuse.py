import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('attention_batch',
    Path(__file__).resolve().parents[1]/'scripts/run_generation_attention_batch.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class ReuseTests(unittest.TestCase):
    def test_verified_completion_is_referenced_and_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'original';source.mkdir()
            complete = source/'generation-complete.json'
            complete.write_text(json.dumps(dict(samples=112,conditions=['slp','no_write','vtt'])))
            record = dict(id=0,path=str(source),limit=112,sha256={
                complete.name:hashlib.sha256(complete.read_bytes()).hexdigest()})
            destination=root/'recovery';destination.mkdir()
            driver.attach_reused_shards(destination,[record])
            self.assertEqual((destination/'shard-0').resolve(),source)
            complete.write_text('{}')
            another=root/'bad';another.mkdir()
            with self.assertRaises(RuntimeError):
                driver.attach_reused_shards(another,[record])
            self.assertFalse((another/'shard-0').exists())


if __name__=='__main__':
    unittest.main()
