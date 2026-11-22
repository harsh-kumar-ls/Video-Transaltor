"""
Speech-recognition model resolution for the dubbing pipeline.

`whisper_model` in config.yaml (or a tier default) can be any of:
  * a faster-whisper built-in name:        "large-v3", "large-v3-turbo", "medium", ...
  * a local CTranslate2 model directory:   "/models/whisper-hindi-ct2"
  * a CTranslate2 repo on Hugging Face:    "deepdml/faster-whisper-large-v3-turbo-ct2"
  * a transformers Whisper repo:           "vasista22/whisper-hindi-large-v2"
    (fine-tuned models are usually published this way; they are converted to
    CTranslate2 float16 once and cached, because WhisperX runs on faster-whisper)

Heavy imports (huggingface_hub, ctranslate2, faster_whisper) happen inside functions so
the classification logic can be unit-tested on its own.
"""
import logging
import os
from typing import Iterable, Optional

BUILTIN = "builtin"
LOCAL = "local"
CT2_REPO = "ct2_repo"
TRANSFORMERS_REPO = "transformers_repo"

_TRANSFORMERS_WEIGHTS = ("pytorch_model.bin", "model.safetensors", "model.safetensors.index.json",
                         "pytorch_model.bin.index.json")
# Files faster-whisper reads from a model directory, copied over when converting.
_COPY_FILES = ("tokenizer.json", "preprocessor_config.json")

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "video-translator", "ct2")


def classify_whisper_model(spec: str, repo_files: Optional[Iterable[str]] = None,
                           builtin_names: Iterable[str] = (), is_dir: bool = False) -> str:
    """
    Decide how to load `spec`. `repo_files` is the file list of the Hugging Face repo
    (only needed when `spec` is neither built in nor a local directory).
    """
    if is_dir:
        return LOCAL
    if spec in set(builtin_names):
        return BUILTIN
    files = set(repo_files or ())
    if "model.bin" in files:
        return CT2_REPO
    if "config.json" in files and files.intersection(_TRANSFORMERS_WEIGHTS):
        return TRANSFORMERS_REPO
    raise ValueError(
        f"Cannot load Whisper model {spec!r}: not a faster-whisper name, a local directory, "
        f"a CTranslate2 repo (model.bin) or a transformers Whisper repo (config.json + weights)."
    )


def ct2_cache_path(spec: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """Where the converted copy of a transformers repo is kept."""
    return os.path.join(cache_dir, spec.replace("/", "--") + "-ct2-float16")


def resolve_whisper_model(spec: str, cache_dir: Optional[str] = None) -> str:
    """
    Return a model name or directory that `whisperx.load_model` / `faster_whisper.WhisperModel`
    accepts, converting a transformers Whisper repo to CTranslate2 on first use.
    """
    cache_dir = cache_dir or os.environ.get("VT_CT2_CACHE") or DEFAULT_CACHE_DIR
    from faster_whisper.utils import available_models

    builtin = available_models()
    if os.path.isdir(spec) or spec in builtin:
        return spec

    from huggingface_hub import list_repo_files

    files = list_repo_files(spec)
    kind = classify_whisper_model(spec, files, builtin_names=builtin)
    if kind == CT2_REPO:
        return spec

    out_dir = ct2_cache_path(spec, cache_dir)
    if os.path.isfile(os.path.join(out_dir, "model.bin")):
        logging.info(f"[ASR] Using cached CTranslate2 conversion of {spec}: {out_dir}")
        return out_dir

    import ctranslate2

    logging.info(f"[ASR] Converting {spec} (transformers) to CTranslate2 float16 → {out_dir} (one-off)")
    os.makedirs(cache_dir, exist_ok=True)
    converter = ctranslate2.converters.TransformersConverter(
        spec,
        copy_files=[f for f in _COPY_FILES if f in files],
        load_as_float16=True,
    )
    converter.convert(out_dir, quantization="float16", force=True)
    return out_dir
