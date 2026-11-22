"""Unit tests for asr_utils model classification (stdlib only)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr_utils import (  # noqa: E402
    BUILTIN, CT2_REPO, LOCAL, TRANSFORMERS_REPO, classify_whisper_model, ct2_cache_path,
)

BUILTINS = ("large-v3", "large-v3-turbo", "medium")


class ClassifyWhisperModel(unittest.TestCase):
    def test_builtin_name(self):
        self.assertEqual(classify_whisper_model("large-v3", builtin_names=BUILTINS), BUILTIN)

    def test_local_directory(self):
        self.assertEqual(classify_whisper_model("/models/x", is_dir=True), LOCAL)

    def test_ctranslate2_repo(self):
        files = ["config.json", "model.bin", "tokenizer.json", "vocabulary.json"]
        self.assertEqual(
            classify_whisper_model("deepdml/faster-whisper-large-v3-turbo-ct2", files, BUILTINS), CT2_REPO
        )

    def test_transformers_repo_with_bin(self):
        # File list of vasista22/whisper-hindi-large-v2 (no tokenizer.json, has pytorch_model.bin).
        files = ["config.json", "pytorch_model.bin", "flax_model.msgpack", "vocab.json", "merges.txt",
                 "preprocessor_config.json", "tokenizer_config.json"]
        self.assertEqual(
            classify_whisper_model("vasista22/whisper-hindi-large-v2", files, BUILTINS), TRANSFORMERS_REPO
        )

    def test_transformers_repo_with_sharded_safetensors(self):
        files = ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                 "model.safetensors.index.json", "tokenizer.json"]
        self.assertEqual(
            classify_whisper_model("collabora/whisper-large-v2-hindi", files, BUILTINS), TRANSFORMERS_REPO
        )

    def test_unloadable_repo_fails_loudly(self):
        with self.assertRaises(ValueError):
            classify_whisper_model("someone/not-a-model", ["README.md"], BUILTINS)


class Ct2CachePath(unittest.TestCase):
    def test_one_folder_per_repo(self):
        self.assertEqual(
            ct2_cache_path("vasista22/whisper-hindi-large-v2", "/cache"),
            "/cache/vasista22--whisper-hindi-large-v2-ct2-float16",
        )


if __name__ == "__main__":
    unittest.main()
