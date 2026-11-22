"""Tests for the dub evaluation helpers (no model needed)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from eval_dub import normalize, parse_srt  # noqa: E402


class ParseSrt(unittest.TestCase):
    def test_round_trips_pipeline_output(self):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from lang_utils import segments_to_srt

        srt = segments_to_srt([
            {"start": 19.629, "end": 25.01, "translated_text": "Pick any decent girl."},
            {"start": 32.151, "end": 34.131, "translated_text": "Excuse me, is anyone there?"},
        ], text_key="translated_text")
        self.assertEqual(parse_srt(srt), [
            (19.629, 25.01, "Pick any decent girl."),
            (32.151, 34.131, "Excuse me, is anyone there?"),
        ])


class Normalize(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(normalize("Excuse me, is anyone THERE?"), "excuse me is anyone there")


if __name__ == "__main__":
    unittest.main()
