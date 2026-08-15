#!/usr/bin/env python3
"""
Offline tests for the OpenAI-compatible provider config examples
(Ollama, DeepSeek, Moonshot/Kimi). Follows the pattern of test_minimax.py.

Unit tests (no API key required):
  - Config file loading and validation
  - TeleMemoryConfig construction from each config

Integration tests are gated behind environment variables:
  - DEEPSEEK_API_KEY for DeepSeek
  - MOONSHOT_API_KEY for Moonshot
  - TELEMEM_TEST_OLLAMA=1 for a locally running Ollama server
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_DIR = Path(__file__).parent.parent / "config"

# provider key -> (config filename, expected base_url fragment, expected model prefix(es))
PROVIDERS = {
    "ollama": ("config.ollama.yaml", "localhost:11434", None),
    "deepseek": ("config.deepseek.yaml", "api.deepseek.com", ("deepseek-",)),
    "moonshot": ("config.moonshot.yaml", "api.moonshot.", ("kimi-", "moonshot-")),
}


def _load(name):
    with open(CONFIG_DIR / PROVIDERS[name][0], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestProviderConfigFiles(unittest.TestCase):
    """Shared structural checks for every provider config example."""

    def test_config_files_exist_and_parse(self):
        for name, (filename, _, _) in PROVIDERS.items():
            with self.subTest(provider=name):
                path = CONFIG_DIR / filename
                self.assertTrue(path.exists(), f"Config file not found: {path}")
                self.assertIsInstance(_load(name), dict)

    def test_required_sections(self):
        for name in PROVIDERS:
            data = _load(name)
            for section in ("llm", "embedder", "vector_store"):
                with self.subTest(provider=name, section=section):
                    self.assertIn(section, data)

    def test_llm_base_url(self):
        for name, (_, base_url_fragment, _) in PROVIDERS.items():
            with self.subTest(provider=name):
                llm_config = _load(name)["llm"]["config"]
                self.assertIn("openai_base_url", llm_config)
                self.assertIn(base_url_fragment, llm_config["openai_base_url"])

    def test_llm_model_name(self):
        for name, (_, _, model_prefixes) in PROVIDERS.items():
            if model_prefixes is None:  # Ollama: any local model is fine
                continue
            with self.subTest(provider=name):
                model = _load(name)["llm"]["config"]["model"]
                self.assertTrue(
                    model.startswith(model_prefixes),
                    f"Unexpected {name} model name: {model}",
                )

    def test_temperature_in_valid_range(self):
        for name in PROVIDERS:
            with self.subTest(provider=name):
                temp = _load(name)["llm"]["config"].get("temperature")
                if temp is not None:
                    self.assertGreaterEqual(temp, 0.0)
                    self.assertLessEqual(temp, 2.0)

    def test_embedder_provider_is_openai_compatible(self):
        for name in PROVIDERS:
            with self.subTest(provider=name):
                self.assertEqual(_load(name)["embedder"]["provider"], "openai")

    def test_buffer_and_threshold(self):
        for name in PROVIDERS:
            with self.subTest(provider=name):
                data = _load(name)
                self.assertIsInstance(data["buffer_size"], int)
                self.assertIsInstance(data["similarity_threshold"], float)


class TestOllamaConfigSpecifics(unittest.TestCase):
    """Ollama is the fully-local setup; dimensions must be self-consistent."""

    def test_embedder_and_vector_store_dims_match(self):
        data = _load("ollama")
        emb_dims = data["embedder"]["config"].get("embedding_dims")
        store_dims = data["vector_store"]["config"].get("embedding_model_dims")
        self.assertIsNotNone(emb_dims, "Ollama config must pin embedding_dims")
        self.assertEqual(emb_dims, store_dims,
                         "embedder dims must match the FAISS index dims")

    def test_embedder_is_local(self):
        emb = _load("ollama")["embedder"]["config"]
        self.assertIn("localhost:11434", emb["openai_base_url"],
                      "Ollama config should be fully local, incl. embeddings")


class TestTeleMemoryConfigConstruction(unittest.TestCase):
    """Each config file must produce a valid TeleMemoryConfig."""

    def test_load_config_builds_telememoryconfig(self):
        from telemem.utils import load_config
        from telemem.configs import TeleMemoryConfig

        test_keys = {
            "DEEPSEEK_API_KEY": "deepseek-test-key",
            "OPENAI_API_KEY": "openai-test-key",
        }
        with patch.dict(os.environ, test_keys):
            for name, (filename, _, _) in PROVIDERS.items():
                with self.subTest(provider=name):
                    config = load_config(str(CONFIG_DIR / filename))
                    self.assertIsInstance(config, TeleMemoryConfig)
                    self.assertEqual(config.llm.provider, "openai")
                    self.assertEqual(config.vector_store.provider, "faiss")

    def test_deepseek_keys_expand_from_environment(self):
        from telemem.utils import load_config

        test_keys = {
            "DEEPSEEK_API_KEY": "deepseek-test-key",
            "OPENAI_API_KEY": "openai-test-key",
        }
        with patch.dict(os.environ, test_keys):
            config = load_config(str(CONFIG_DIR / "config.deepseek.yaml"))

        self.assertEqual(config.llm.config["api_key"], "deepseek-test-key")
        self.assertEqual(config.embedder.config["api_key"], "openai-test-key")

    def test_missing_deepseek_key_fails_with_variable_name(self):
        from telemem.utils import load_config

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "openai-test-key"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                load_config(str(CONFIG_DIR / "config.deepseek.yaml"))

    def test_json_environment_expansion_cannot_change_config_structure(self):
        from telemem.utils import load_config

        data = _load("ollama")
        data["llm"]["config"]["api_key"] = "${TELEMEM_TEST_API_KEY}"
        injected_text = "secret\nvector_store: replaced"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with patch.dict(os.environ, {"TELEMEM_TEST_API_KEY": injected_text}):
                config = load_config(str(path))

        self.assertEqual(config.llm.config["api_key"], injected_text)
        self.assertEqual(config.vector_store.provider, "faiss")


@unittest.skipUnless(os.getenv("DEEPSEEK_API_KEY"), "DEEPSEEK_API_KEY not set")
class TestDeepSeekIntegration(unittest.TestCase):
    def test_chat_completion(self):
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ["DEEPSEEK_API_KEY"],
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=8,
        )
        self.assertTrue(response.choices[0].message.content.strip())


@unittest.skipUnless(os.getenv("MOONSHOT_API_KEY"), "MOONSHOT_API_KEY not set")
class TestMoonshotIntegration(unittest.TestCase):
    def test_chat_completion(self):
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.moonshot.cn/v1",
            api_key=os.environ["MOONSHOT_API_KEY"],
        )
        response = client.chat.completions.create(
            model="kimi-k2-0905-preview",
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=8,
        )
        self.assertTrue(response.choices[0].message.content.strip())


@unittest.skipUnless(os.getenv("TELEMEM_TEST_OLLAMA"), "TELEMEM_TEST_OLLAMA not set")
class TestOllamaIntegration(unittest.TestCase):
    def test_chat_completion(self):
        from openai import OpenAI

        config = _load("ollama")
        client = OpenAI(
            base_url=config["llm"]["config"]["openai_base_url"],
            api_key="ollama",
        )
        response = client.chat.completions.create(
            model=config["llm"]["config"]["model"],
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        )
        self.assertTrue(response.choices[0].message.content.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
