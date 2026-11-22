"""Unit tests for lang_utils (stdlib only: python -m unittest discover tests)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lang_utils import (  # noqa: E402
    NLLB_LANG_CODES,
    SOURCE_LANGUAGE_LABELS,
    collapse_repeats,
    nllb_code,
    nllb_max_new_tokens,
    normalize_language_code,
    resolve_source_language,
    segments_to_srt,
    split_long_segments,
    tidy_translation,
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


class TidyTranslation(unittest.TestCase):
    # Inputs below are real NLLB outputs from the Kaggle run on a Hindi drama clip.
    def test_reattaches_contractions_and_punctuation(self):
        self.assertEqual(
            tidy_translation("I hate all the things I 've given up for you . Finally , you 'll see"),
            "I hate all the things I've given up for you. Finally, you'll see",
        )
        self.assertEqual(
            tidy_translation("If we don 't get the money by tomorrow, it will all be over."),
            "If we don't get the money by tomorrow, it will all be over.",
        )

    def test_cuts_decode_loop(self):
        looped = "I don't care for an unknown man in the trunk " + "of the trunk " * 40
        self.assertEqual(
            tidy_translation(looped), "I don't care for an unknown man in the trunk of the trunk"
        )

    def test_leaves_clean_text_alone(self):
        clean = "Pick any decent girl, ask her to marry you, and if she says no, kill her?"
        self.assertEqual(tidy_translation(clean), clean)


class CollapseRepeats(unittest.TestCase):
    def test_keeps_two_copies(self):
        self.assertEqual(collapse_repeats("too many, too many will"), "too many, too many will")

    def test_collapses_three_or_more(self):
        self.assertEqual(collapse_repeats("no no no no"), "no")
        self.assertEqual(collapse_repeats("go away go away go away now"), "go away now")


def _words(*timed):
    return [{"word": w, "start": s, "end": e} for w, s, e in timed]


class SplitLongSegments(unittest.TestCase):
    # Shaped like the Hindi fine-tuned Whisper output on the test clip: no punctuation,
    # three lines from different speakers in one 30 s segment.
    LONG = {
        "start": 19.6, "end": 49.1,
        "text": "किसी भी शरीफ लड़की को उठाओ एक्सक्यूज़ मी कोई है तुम यहां क्या कर",
        "words": _words(("किसी", 19.6, 20.0), ("भी", 20.0, 20.3), ("शरीफ", 20.4, 21.0),
                        ("लड़की", 21.1, 21.6), ("को", 21.6, 21.8), ("उठाओ", 21.9, 25.0),
                        ("एक्सक्यूज़", 32.1, 32.6), ("मी", 32.6, 32.9), ("कोई", 33.0, 33.3),
                        ("है", 33.3, 34.1), ("तुम", 34.2, 34.5), ("यहां", 34.5, 35.0),
                        ("क्या", 35.1, 35.5), ("कर", 35.5, 38.4)),
    }

    def test_splits_at_longest_pause(self):
        parts = split_long_segments([self.LONG], max_seconds=8.0)
        self.assertEqual([p["text"] for p in parts], [
            "किसी भी शरीफ लड़की को उठाओ",
            "एक्सक्यूज़ मी कोई है तुम यहां क्या कर",
        ])
        self.assertEqual((parts[0]["start"], parts[0]["end"]), (19.6, 25.0))
        self.assertEqual((parts[1]["start"], parts[1]["end"]), (32.1, 38.4))
        self.assertEqual(len(parts[1]["words"]), 8)

    def test_keeps_short_segments_unchanged(self):
        seg = {"start": 0.0, "end": 5.0, "text": "Excuse me, कोई है?", "words": _words(("x", 0, 5))}
        self.assertEqual(split_long_segments([seg]), [seg])

    def test_needs_a_real_pause(self):
        # 10 s of continuous speech (gaps of 0.1 s) is left as one line rather than cut mid-phrase.
        words = _words(*[(f"w{i}", i * 1.0, i * 1.0 + 0.9) for i in range(10)])
        seg = {"start": 0.0, "end": 9.9, "text": "", "words": words}
        self.assertEqual(len(split_long_segments([seg], max_seconds=8.0)), 1)

    def test_without_word_timestamps_returns_segment(self):
        seg = {"start": 0.0, "end": 30.0, "text": "unaligned"}
        self.assertEqual(split_long_segments([seg]), [seg])

    def test_untimed_word_stays_with_previous(self):
        words = _words(("एक", 0.0, 1.0), ("ज़", None, None), ("दो", 1.1, 2.0), ("तीन", 9.0, 10.0))
        parts = split_long_segments([{"start": 0.0, "end": 10.0, "text": "", "words": words}],
                                    max_seconds=8.0)
        self.assertEqual([p["text"] for p in parts], ["एक ज़ दो", "तीन"])


class NllbMaxNewTokens(unittest.TestCase):
    def test_scales_with_input_and_is_bounded(self):
        self.assertEqual(nllb_max_new_tokens(10), 36)
        self.assertEqual(nllb_max_new_tokens(0), 16)
        self.assertEqual(nllb_max_new_tokens(1000), 512)


if __name__ == "__main__":
    unittest.main()
