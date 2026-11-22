"""
Compare speech-recognition models on one clip, run the way the pipeline runs them
(WhisperX: pyannote VAD + batched faster-whisper, source language forced), then translate
each transcript with the pipeline's NLLB settings so you can see what each dub would say.

Speech recognition is the weak link on music-heavy Hindi drama: a misheard line is
translated into fluent nonsense, and the dub WER from tools/eval_dub.py cannot see it.
There is no reference transcript here, so read the Hindi side by side (or have a Hindi
speaker mark which is right) before switching models.

Usage (from the repo root, GPU recommended):
    python tools/compare_asr.py clip.wav \
        --model turbo=deepdml/faster-whisper-large-v3-turbo-ct2 \
        --model large-v3=large-v3 \
        --model vasista22=vasista22/whisper-hindi-large-v2 \
        --vocals demucs_vocals.wav --lang hi --translate --out asr_compare

A --model spec is anything asr_utils.resolve_whisper_model accepts (faster-whisper name,
CTranslate2 dir/repo, or a transformers Whisper repo, converted on first use).
Writes <out>.json and <out>.md.
"""
import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_named(values, flag):
    named = []
    for v in values:
        name, sep, spec = v.partition("=")
        if not sep or not name or not spec:
            raise SystemExit(f"{flag} expects name=value, got {v!r}")
        named.append((name, spec))
    return named


def transcribe(model_path, audio, lang, device, batch_size):
    import torch
    import whisperx

    model = whisperx.load_model(model_path, device, compute_type="float16" if device == "cuda" else "int8",
                                language=lang, asr_options={"word_timestamps": True})
    try:
        result = model.transcribe(audio, batch_size=batch_size, language=lang)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return [{"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2), "text": s["text"].strip()}
            for s in result["segments"]]


def to_markdown(runs, translate):
    lines = []
    for run in runs:
        lines.append(f"## {run['model']} on {run['audio']}  ({run['seconds']} s)\n")
        lines.append("| time | Hindi | English |" if translate else "| time | Hindi |")
        lines.append("|---|---|---|" if translate else "|---|---|")
        for seg in run["segments"]:
            row = f"| {seg['start']:.1f}-{seg['end']:.1f} | {seg['text']} |"
            if translate:
                row += f" {seg.get('translated_text', '')} |"
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="mixed audio (or video) of the clip")
    ap.add_argument("--vocals", default=None, help="optional Demucs vocal stem; every model also runs on it")
    ap.add_argument("--model", action="append", required=True, help="name=spec, repeatable")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--tgt", default="en")
    ap.add_argument("--translate", action="store_true", help="also translate each transcript with NLLB")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="asr_compare")
    args = ap.parse_args()

    if args.device == "cuda":
        import torch  # noqa: F401  load torch's CUDA libs before CTranslate2
    import whisperx

    from asr_utils import resolve_whisper_model

    inputs = [("mix", args.audio)] + ([("vocals", args.vocals)] if args.vocals else [])
    audios = {label: whisperx.load_audio(path) for label, path in inputs}

    runs = []
    for name, spec in parse_named(args.model, "--model"):
        path = resolve_whisper_model(spec)
        for label, _ in inputs:
            t0 = time.time()
            segs = transcribe(path, audios[label], args.lang, args.device, args.batch_size)
            runs.append({"model": name, "spec": spec, "audio": label,
                         "seconds": round(time.time() - t0, 1), "segments": segs})
            print(f"[{name} / {label}] {len(segs)} segments in {runs[-1]['seconds']} s")
            for s in segs:
                print(f"   {s['start']:6.1f}-{s['end']:6.1f}  {s['text']}")

    if args.translate:
        from translator import (APP_CONFIG, batch_translate_segments, load_translation_model,
                                merge_short_segments)

        model, tokenizer = load_translation_model(APP_CONFIG.get("hardware_tier") or "gpu_medium")
        for run in runs:
            # Same steps as the pipeline: merge very short segments, then batch-translate.
            run["segments"] = batch_translate_segments(merge_short_segments(run["segments"]),
                                                       args.lang, args.tgt, model, tokenizer)

    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=1)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(to_markdown(runs, args.translate))
    print(f"wrote {args.out}.json and {args.out}.md")


if __name__ == "__main__":
    main()
