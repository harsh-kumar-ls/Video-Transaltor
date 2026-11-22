"""Unit tests for lang_utils (stdlib only: python -m unittest discover tests)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lang_utils import (  # noqa: E402
    NLLB_LANG_CODES,
    SOURCE_LANGUAGE_LABELS,
    nllb_code,
    normalize_language_code,
    resolve_source_language,
    segments_to_srt,
)


class NormalizeLanguageCode(unittest.TestCase):
    def test_strips_region_and_case(self):
        self.assertEqual(normalize_language_code("hi-IN"), "hi")
        self.assertEqual(normalize_language_code("HI"), "hi")
        self.assertEqual(normalize_language_code("en_US"), "en")

    def test_auto_and_empty_mean_no_override(self):
        for value in (None, "", "  ", "auto", "AUTO"):
            self.assertIsNone(normalize_language_code(value), value)


class ResolveSourceLanguage(unittest.TestCase):
    def test_override_beats_detection(self):
        # Whisper detected Urdu on Hindi audio; the user forced Hindi.
        self.assertEqual(resolve_source_language("hi", "ur"), "hi")

    def test_detection_used_when_auto(self):
        self.assertEqual(resolve_source_language("auto", "hi"), "hi")
        self.assertEqual(resolve_source_language(None, "fr-FR"), "fr")

    def test_defaults_to_english_when_nothing_known(self):
        self.assertEqual(resolve_source_language(None, None), "en")

    def test_rejects_unsupported_override(self):
        with self.assertRaises(ValueError):
            resolve_source_language("xx", "hi")


class NllbCode(unittest.TestCase):
    def test_hindi_uses_devanagari(self):
        self.assertEqual(nllb_code("hi"), "hin_Deva")

    def test_indian_languages_have_native_scripts(self):
        expected = {
            "bn": "ben_Beng", "mr": "mar_Deva", "ta": "tam_Taml", "te": "tel_Telu",
            "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym", "pa": "pan_Guru",
            "ur": "urd_Arab",
        }
        for code, nllb in expected.items():
            self.assertEqual(nllb_code(code), nllb, code)

    def test_unknown_language_fails_loudly(self):
        # The old code silently produced "hi_Latn"-style codes that NLLB does not know.
        with self.assertRaises(ValueError):
            nllb_code("xx")

    def test_every_source_option_is_translatable(self):
        for code in SOURCE_LANGUAGE_LABELS:
            self.assertIn(code, NLLB_LANG_CODES, code)


class SegmentsToSrt(unittest.TestCase):
    def test_formats_cues(self):
        srt = segments_to_srt([
            {"start": 0.0, "end": 2.5, "translated_text": "Where were you last night?"},
            {"start": 3661.042, "end": 3663.9, "translated_text": "At the office."},
        ], text_key="translated_text")
        self.assertEqual(
            srt,
            "1\n00:00:00,000 --> 00:00:02,500\nWhere were you last night?\n\n"
            "2\n01:01:01,042 --> 01:01:03,900\nAt the office.\n",
        )

    def test_keeps_devanagari_text(self):
        srt = segments_to_srt([{"start": 1, "end": 2, "text": "तुम कल रात कहाँ थे?"}])
        self.assertIn("तुम कल रात कहाँ थे?", srt)

    def test_skips_empty_and_renumbers(self):
        srt = segments_to_srt([
            {"start": 0, "end": 1, "text": "  "},
            {"start": 1, "end": 2, "text": "Hello"},
        ])
        self.assertTrue(srt.startswith("1\n00:00:01,000"))

    def test_clamps_end_before_start(self):
        srt = segments_to_srt([{"start": 5, "end": 4, "text": "x"}])
        self.assertIn("00:00:05,000 --> 00:00:05,000", srt)


if __name__ == "__main__":
    unittest.main()
