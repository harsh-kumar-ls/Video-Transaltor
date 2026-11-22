"""
Score a dubbed video: is the synthesized speech saying what the subtitles say it should?

Transcribes the dubbed audio with faster-whisper (word timestamps), assigns each heard word
to the subtitle cue whose time window contains it, and reports word error rate (WER) per cue
and overall (word-weighted). A high WER means the dub is unintelligible, cut off, or
squeezed too fast into its slot. It does not judge whether the translation itself is correct.

Usage:
    python tools/eval_dub.py <dubbed.mp4> <translated.srt> [--lang en] [--json out.json]

Needs: pip install faster-whisper jiwer
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_srt(text):
    """[(start_s, end_s, text)] from an .srt body."""
    def ts(s):
        h, m, rest = s.strip().replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)

    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [l for l in block.splitlines() if l.strip()]
        timing = next((l for l in lines if "-->" in l), None)
        if not timing:
            continue
        start, end = timing.split("-->")
        body = " ".join(lines[lines.index(timing) + 1:])
        cues.append((ts(start), ts(end), body))
    return cues


def load_audio_16k(path):
    """Decode any media file to 16 kHz mono float32 with ffmpeg (avoids PyAV version clashes)."""
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def normalize(text):
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", text.lower()).split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("srt")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--model", default="deepdml/faster-whisper-large-v3-turbo-ct2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import jiwer
    if args.device == "cuda":
        import torch  # noqa: F401  loads torch's bundled CUDA libs (libcublas) for CTranslate2
    from faster_whisper import WhisperModel

    cues = parse_srt(open(args.srt, encoding="utf-8").read())
    model = WhisperModel(args.model, device=args.device,
                         compute_type="float16" if args.device == "cuda" else "int8")
    segments, _ = model.transcribe(load_audio_16k(args.video), language=args.lang, word_timestamps=True)
    words = [w for seg in segments for w in (seg.words or [])]

    heard = [[] for _ in cues]
    for w in words:
        mid = (w.start + w.end) / 2
        for i, (start, end, _) in enumerate(cues):
            if start - 0.25 <= mid <= end + 0.25:
                heard[i].append(w.word)
                break

    rows, ref_words, errors = [], 0, 0.0
    for (start, end, ref), got in zip(cues, heard):
        ref_n, got_n = normalize(ref), normalize(" ".join(got))
        n = len(ref_n.split())
        wer = jiwer.wer(ref_n, got_n) if n else 0.0
        ref_words += n
        errors += wer * n
        rows.append({"start": round(start, 2), "end": round(end, 2), "words": n,
                     "wer": round(wer, 3), "intended": ref, "heard": " ".join(got).strip()})

    overall = errors / ref_words if ref_words else 0.0
    for r in rows:
        print(f"{r['start']:6.1f}-{r['end']:6.1f}s  WER {100 * r['wer']:5.1f}%  "
              f"({r['words']} words)  heard: {r['heard'][:80]}")
    print(f"OVERALL WER {100 * overall:.1f}% over {ref_words} intended words, {len(rows)} cues")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"overall_wer": overall, "intended_words": ref_words, "cues": rows},
                      f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
