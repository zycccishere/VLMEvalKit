"""VVTT ⟨V,V,T,T⟩ item-order control: construction and no silent fallback."""
import importlib.util
import sys
import unittest
from pathlib import Path

# Load replay_policy by file path: importing through the vlmeval.vlm package
# pulls every wrapper (and vllm), which is slow and version-fragile; the
# policy module itself is stdlib-only.
_policy_path = Path(__file__).resolve().parents[1] / 'vlmeval/vlm/replay_policy.py'
_spec = importlib.util.spec_from_file_location('replay_policy_under_test', _policy_path)
rp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rp
_spec.loader.exec_module(rp)

CANONICAL_REPLAY_MODES = rp.CANONICAL_REPLAY_MODES
REPLAY_IMAGE_IMAGE_TEXT_TEXT = rp.REPLAY_IMAGE_IMAGE_TEXT_TEXT
apply_replay = rp.apply_replay
canonicalize_replay_mode = rp.canonicalize_replay_mode
count_modalities = rp.count_modalities


def content():
    return [
        {"type": "image", "path": "/x/a.png"},
        {"type": "text", "text": "Q"},
    ]


class VVTTModeTests(unittest.TestCase):
    def test_mode_is_canonical_and_never_degrades(self):
        self.assertIn(REPLAY_IMAGE_IMAGE_TEXT_TEXT, CANONICAL_REPLAY_MODES)
        self.assertEqual(canonicalize_replay_mode("image_image_text_text"),
                         REPLAY_IMAGE_IMAGE_TEXT_TEXT)
        # A typo must NOT silently run as image_text; it degrades to the
        # no-op only for genuinely unknown strings, so assert the real one
        # round-trips and produces a non-noop result below.
        self.assertNotEqual(canonicalize_replay_mode("image_image_text_textx"),
                            REPLAY_IMAGE_IMAGE_TEXT_TEXT)

    def test_order_is_vvtt_with_same_multiset_as_slp(self):
        out = apply_replay(content(), REPLAY_IMAGE_IMAGE_TEXT_TEXT)
        self.assertEqual([i["type"] for i in out],
                         ["image", "image", "text", "text"])
        # Same item multiset as SLP ⟨V,T,V,T⟩; engine token equality
        # requires the separate native payload audit.
        slp = apply_replay(content(), "image_text_image_text")
        self.assertEqual(count_modalities(out), count_modalities(slp))
        self.assertEqual(out[0]["path"], out[1]["path"])
        self.assertEqual(out[2]["text"], "Q")
        self.assertEqual(out[3]["text"], "Q")

    def test_repeat_times_two_keeps_prefix_order(self):
        out = apply_replay(content(), REPLAY_IMAGE_IMAGE_TEXT_TEXT, repeat_times=2)
        self.assertEqual([i["type"] for i in out], ["image"] * 4 + ["text"] * 4)


if __name__ == "__main__":
    unittest.main()
