"""Full-sequence replay with an exact separator (multi-image E8 Replay@8)."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

# Load replay_policy by file path (stdlib-only module; avoids the wrapper/vllm import chain).
_policy_path = Path(__file__).resolve().parents[1] / 'vlmeval/vlm/replay_policy.py'
_spec = importlib.util.spec_from_file_location('replay_policy_separator_test', _policy_path)
rp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rp
_spec.loader.exec_module(rp)

SEP = 'lets repeat the question again:'


def interleaved(key):
    # T V T V T, as ERQABench builds from visual_indices.
    return [{'type': 'text', key: 'A'}, {'type': 'image', 'value': '/x/1.jpg'}, {'type': 'text', key: 'B'},
            {'type': 'image', 'value': '/x/2.jpg'}, {'type': 'text', key: 'C'}]


class SeparatorTests(unittest.TestCase):
    def test_exact_separator_between_original_and_full_copy(self):
        for key in ('value', 'text'):
            src = interleaved(key)
            out = rp.apply_replay(src, 'image_text_image_text', sequence_separator=SEP)
            self.assertEqual(out[:5], src)
            self.assertEqual(out[5], {'type': 'text', key: SEP})
            self.assertEqual(out[6:], src)
            self.assertEqual(''.join('V' if x['type'] == 'image' else 'T' for x in out), 'TVTVT' + 'T' + 'TVTVT')

    def test_empty_separator_keeps_existing_full_sequence_replay(self):
        src = interleaved('value')
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('REPLAY_SEQUENCE_SEPARATOR', None)
            self.assertEqual(rp.apply_replay(src, 'image_text_image_text'), src + src)

    def test_env_default_is_honoured_and_recorded(self):
        src = interleaved('text')
        with mock.patch.dict(os.environ, {'REPLAY_SEQUENCE_SEPARATOR': SEP, 'REPLAY_MODE': 'image_text_image_text'}):
            self.assertEqual(rp.apply_replay(src, 'image_text_image_text')[5]['text'], SEP)
            self.assertEqual(rp.read_replay_config_from_env()['sequence_separator'], SEP)

    def test_separator_is_never_silently_dropped(self):
        src = interleaved('value')
        self.assertEqual(rp.apply_replay(src, 'image_text', sequence_separator=SEP), src)  # Original@8: no replay
        for mode in ('image_text_text', 'image_image_text_text', 'image_text_image_image_text'):
            with self.assertRaises(ValueError):
                rp.apply_replay(src, mode, sequence_separator=SEP)


if __name__ == '__main__':
    unittest.main()
