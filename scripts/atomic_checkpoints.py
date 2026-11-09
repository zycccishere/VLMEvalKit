"""Run the unchanged native entrypoint with transactional smp.dump writes.

Only the native file writer is wrapped, at its normal import time. Model,
dataset, judge, batching, and resume selection remain in the execution checkout.
"""
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import runpy
import stat
import sys
import tempfile


def atomic_writer(original):
    def dump(data, filename, **kwargs):
        target = Path(filename)
        fd, name = tempfile.mkstemp(prefix='.'+target.stem+'.checkpoint-', suffix=target.suffix, dir=target.parent)
        os.close(fd)
        temp = Path(name)
        fallback = temp.with_suffix('.pkl')
        try:
            original(data, str(temp), **kwargs)
            if fallback != temp and fallback.exists():
                raise RuntimeError('native serialization changed format; preserve the previous checkpoint')
            if target.exists():
                os.chmod(temp, stat.S_IMODE(target.stat().st_mode))
            with temp.open('rb') as f:
                os.fsync(f.fileno())
            os.replace(temp, target)
            directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temp.unlink(missing_ok=True)
            if fallback != temp:
                fallback.unlink(missing_ok=True)
    return dump


class NativeDumpHook(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != 'vlmeval.smp.file':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError('native smp.file writer is required for atomic checkpoints')
        loader = spec.loader

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return loader.create_module(spec)

            def exec_module(self, module):
                loader.exec_module(module)
                module.dump = atomic_writer(module.dump)
                print('[CHECKPOINT_IO] atomic-fsync-replace-v1', flush=True)

        spec.loader = Loader()
        return spec


def main():
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path[0] = str(script.parent)
    hook = NativeDumpHook()
    sys.meta_path.insert(0, hook)
    try:
        runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.meta_path.remove(hook)


if __name__ == '__main__':
    main()
