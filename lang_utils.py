"""
Language helpers for the dubbing pipeline: source-language handling, NLLB codes,
translation clean-up and subtitle export.

Added in this fork for Hindi (and other Indian-language) source video. Kept free of
heavy imports (torch, whisperx) so it can be unit-tested on its own.
"""
import re
from typing import Dict, Iterable, List, Optional

# ISO 639-1 code -> NLLB-200 code. NLLB needs the script suffix: Hindi is written in
# Devanagari, so "hi" must map to "hin_Deva". The old fallback f"{code}_Latn" produced
# "hi_Latn", which is not an NLLB language code, so Hindi input got an unknown-language token.
NLLB_LANG_CODES: Dict[str, str] = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "de": "deu_Latn",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "ru": "rus_Cyrl",
    # Indian languages
    "hi": "hin_Deva",
    "ur": "urd_Arab",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
}

# Languages the user can force as the source. Whisper's own language detection can
# labels Hindi speech as Urdu (same spoken language, different script) and then writes the
# transcript in Arabic script, so forcing "hi" gives a Devanagari transcript that NLLB and
# the Hindi alignment model both expect.
SOURCE_LANGUAGE_LABELS: Dict[str, str] = {
    "hi": "Hindi",
    "en": "English",
    "bn": "Bengali",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "es": "Spanish",
    "fr": "French",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "pt": "Portuguese",
}

AUTO = "auto"


def normalize_language_code(code: Optional[str]) -> Optional[str]:
    """'hi-IN' -> 'hi', 'HI' -> 'hi', '' / None / 'auto' -> None."""
    if code is None:
        return None
    code = str(code).strip().lower().replace("_", "-")
    if not code or code == AUTO:
        return None
    return code.split("-")[0]


def resolve_source_language(override: Optional[str], detected: Optional[str]) -> str:
    """
    Pick the source language for translation.

    An explicit override (CLI flag, config.yaml or UI) always wins over Whisper's
    detection. Raises ValueError for an override the pipeline cannot translate.
    """
    forced = normalize_language_code(override)
    if forced is not None:
        if forced not in NLLB_LANG_CODES:
            raise ValueError(
                f"Unsupported source language '{override}'. "
                f"Supported: {sorted(NLLB_LANG_CODES)}"
            )
        return forced
    return normalize_language_code(detected) or "en"


def nllb_code(lang: str) -> str:
    """NLLB code for a short language code; fails loudly instead of guessing a script."""
    code = normalize_language_code(lang)
    if code not in NLLB_LANG_CODES:
        raise ValueError(
            f"No NLLB-200 code configured for language '{lang}'. "
            f"Add it to NLLB_LANG_CODES in lang_utils.py."
        )
    return NLLB_LANG_CODES[code]


def _srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def segments_to_srt(segments: Iterable[dict], text_key: str = "text") -> str:
    """
    Build an .srt file body from segments with 'start'/'end' in seconds.

    Segments with empty text are skipped; an end time before the start is clamped to the
    start. Cue numbers stay consecutive.
    """
    blocks: List[str] = []
    for seg in segments:
        text = str(seg.get(text_key) or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0) or 0.0)
        end = max(start, float(seg.get("end", start) or start))
        blocks.append(
            f"{len(blocks) + 1}\n{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n{text}\n"
        )
    return "\n".join(blocks)


# --- Translation clean-up -------------------------------------------------------------
# Found in an end-to-end run on a Hindi drama clip (Kaggle, 2026-11-22):
#  * NLLB output had detached punctuation and contractions ("I 've", "don 't", "tell .").
#    Those go to TTS and subtitles as-is.
#  * On a misheard, music-heavy segment NLLB looped ("of the trunk of the trunk of the ...")
#    until max_new_tokens, and the loop was then squeezed into a 15 s slot.

_CONTRACTION = re.compile(r"(\w)\s+'(s|t|m|d|ll|re|ve)\b", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:%])")


def collapse_repeats(text: str, min_copies: int = 3, max_ngram: int = 8) -> str:
    """
    Collapse a phrase repeated back-to-back `min_copies` or more times into one copy.

    "of the trunk of the trunk of the trunk" -> "of the trunk". Two copies are left alone
    ("too many, too many" can be real dialogue).
    """
    words = text.split()
    changed = True
    while changed:
        changed = False
        for n in range(1, max_ngram + 1):
            i = 0
            while i + n * min_copies <= len(words):
                gram = words[i:i + n]
                copies = 1
                while words[i + copies * n:i + (copies + 1) * n] == gram:
                    copies += 1
                if copies >= min_copies:
                    words = words[:i + n] + words[i + copies * n:]
                    changed = True
                else:
                    i += 1
    return " ".join(words)


def tidy_translation(text: str) -> str:
    """Re-attach punctuation and contractions, collapse runaway repeats, squash spaces."""
    text = _CONTRACTION.sub(lambda m: f"{m.group(1)}'{m.group(2)}", text or "")
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = collapse_repeats(text)
    return re.sub(r"\s{2,}", " ", text).strip()


def nllb_max_new_tokens(input_tokens: int, cap: int = 512) -> int:
    """Output budget relative to the input, so a looping decode stops early."""
    return max(16, min(cap, 2 * int(input_tokens) + 16))
