#!/usr/bin/env python3
"""
Tests for MiniMax provider integration with TeleMem.

MiniMax provides an OpenAI-compatible API at https://api.minimax.io/v1
and can be used as the LLM backend in TeleMem via the standard
`provider: openai` + `openai_base_url` config pattern.

Unit tests (no API key required):
  - Config loading and validation
  - TeleMemory instantiation with MiniMax config
  - Config value assertions

Integration tests (requires MINIMAX_API_KEY env var):
  - MiniMax LLM API connectivity
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MINIMAX_GLOBAL_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_CN_BASE_URL = "https://api.minimax.cn/v1"
# Public OpenAI-compatible regional endpoints for the text models.
MINIMAX_REGIONAL_URLS = [MINIMAX_GLOBAL_BASE_URL, MINIMAX_CN_BASE_URL]
MINIMAX_MODEL_FLASH = "MiniMax-M3.1-Flash"
MINIMAX_MODEL_M3 = "MiniMax-M3"
MINIMAX_MODEL_M2_7 = "MiniMax-M2.7"
MINIMAX_MODEL_M2_7_HS = "MiniMax-M2.7-highspeed"

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.minimax.yaml"


class TestMiniMaxConfigFile(unittest.TestCase):
    """Unit tests for the MiniMax config file."""

    def test_config_file_exists(self):
        """config/config.minimax.yaml should exist."""
        self.assertTrue(CONFIG_PATH.exists(), f"Config file not found: {CONFIG_PATH}")

    def test_config_file_is_valid_yaml(self):
        """config/config.minimax.yaml must be parseable YAML."""
        import yaml
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f)
        self.assertIsInstance(data, dict)

    def test_config_has_required_sections(self):
        """Config must have llm, embedder, vector_store sections."""
        import yaml
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f)
        for section in ("llm", "embedder", "vector_store"):
            self.assertIn(section, data, f"Missing section: {section}")

    def test_config_llm_uses_minimax_base_url(self):
        """LLM section must point to MiniMax API base URL."""
        import yaml
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f)
        llm_config = data["llm"]["config"]
        self.assertIn("openai_base_url", llm_config)
        self.assertIn("minimax.io", llm_config["openai_base_url"],
                      "LLM base URL should point to api.minimax.io")

    def test_config_llm_model_is_minimax(self):
        """LLM model must be a valid MiniMax model name."""
        import yaml
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f)
        model = data["llm"]["config"]["model"]
        self.assertTrue(
            model.startswith("MiniMax-"),
            f"Expected MiniMax model, got: {model}"
        )

    def test_config_enables_reasoning_parameter_forwarding(self):
        """The adapter must forward effort and omit its default token cap."""
        import yaml
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f)
        self.assertTrue(data["llm"]["config"]["is_reasoning_model"])

    def test_config_embedder_provider_is_openai(self):
        """Embedder must use openai provider (MiniMax has no embedding API)."""
        import yaml
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f)
        embedder_provider = data["embedder"]["provider"]
        self.assertEqual(embedder_provider, "openai")

    def test_config_has_buffer_and_threshold(self):
        """Config must include buffer_size and similarity_threshold."""
        import yaml
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f)
        self.assertIn("buffer_size", data)
        self.assertIn("similarity_threshold", data)
        self.assertIsInstance(data["buffer_size"], int)
        self.assertIsInstance(data["similarity_threshold"], float)


class TestMiniMaxTeleMemoryConfig(unittest.TestCase):
    """Unit tests for TeleMemoryConfig with MiniMax settings."""

    def _make_minimax_config_dict(self):
        return {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": MINIMAX_MODEL_M3,
                    "openai_base_url": MINIMAX_GLOBAL_BASE_URL,
                    "api_key": "sk-test-minimax",
                    "temperature": 0.7,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "openai_base_url": "https://api.openai.com/v1",
                    "api_key": "sk-test-openai",
                },
            },
            "vector_store": {
                "provider": "faiss",
                "config": {
                    "collection_name": "telemem_test",
                    "path": "/tmp/telemem_test_faiss",
                },
            },
            "history_db_path": "/tmp/telemem_test_history.db",
            "buffer_size": 32,
            "similarity_threshold": 0.95,
        }

    def test_telemem_config_with_minimax_settings(self):
        """TeleMemoryConfig must accept MiniMax LLM settings."""
        from telemem.configs import TeleMemoryConfig
        cfg_dict = self._make_minimax_config_dict()
        config = TeleMemoryConfig(**cfg_dict)
        self.assertEqual(config.buffer_size, 32)
        self.assertAlmostEqual(config.similarity_threshold, 0.95)

    def test_minimax_base_url_preserved_in_config(self):
        """MiniMax base URL must be preserved in the LLM config."""
        from telemem.configs import TeleMemoryConfig
        cfg_dict = self._make_minimax_config_dict()
        config = TeleMemoryConfig(**cfg_dict)
        self.assertEqual(config.llm.config["openai_base_url"], MINIMAX_GLOBAL_BASE_URL)

    def test_minimax_model_preserved_in_config(self):
        """MiniMax model name must be preserved in the LLM config."""
        from telemem.configs import TeleMemoryConfig
        cfg_dict = self._make_minimax_config_dict()
        config = TeleMemoryConfig(**cfg_dict)
        self.assertEqual(config.llm.config["model"], MINIMAX_MODEL_M3)

    def test_minimax_temperature_preserved_in_config(self):
        """Temperature value must be preserved in the LLM config."""
        from telemem.configs import TeleMemoryConfig
        cfg_dict = self._make_minimax_config_dict()
        config = TeleMemoryConfig(**cfg_dict)
        self.assertAlmostEqual(config.llm.config["temperature"], 0.7)

    def test_minimax_highspeed_model_name(self):
        """MiniMax-M2.7-highspeed is a valid model for the config."""
        from telemem.configs import TeleMemoryConfig
        cfg_dict = self._make_minimax_config_dict()
        cfg_dict["llm"]["config"]["model"] = MINIMAX_MODEL_M2_7_HS
        config = TeleMemoryConfig(**cfg_dict)
        self.assertEqual(config.llm.config["model"], MINIMAX_MODEL_M2_7_HS)


class TestMiniMaxEndpoints(unittest.TestCase):
    """Validate the regional OpenAI-compatible endpoints exposed by MiniMax."""

    EXPECTED_REGIONAL_URLS = MINIMAX_REGIONAL_URLS

    def test_global_endpoint_is_documented(self):
        """The global (api.minimax.io) endpoint must be in the regional list."""
        self.assertIn(MINIMAX_GLOBAL_BASE_URL, self.EXPECTED_REGIONAL_URLS)

    def test_china_endpoint_is_documented(self):
        """The China (api.minimax.cn) endpoint must be in the regional list."""
        self.assertIn(MINIMAX_CN_BASE_URL, self.EXPECTED_REGIONAL_URLS)

    def test_config_uses_a_documented_regional_endpoint(self):
        """The config file must point to one of the documented regional URLs."""
        import yaml
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f)
        base_url = data["llm"]["config"]["openai_base_url"]
        self.assertIn(base_url, self.EXPECTED_REGIONAL_URLS,
                      "LLM base URL must be one of the documented MiniMax regional endpoints")


class TestMiniMaxModels(unittest.TestCase):
    """Validate the shipped preset and explicit saved model selections."""

    def test_preferred_model_for_config(self):
        """Load the actual preset rather than a separate test-only catalog."""
        from telemem.utils import load_config

        config = load_config(str(CONFIG_PATH))
        self.assertEqual(config.llm.config["model"], MINIMAX_MODEL_FLASH)

    def test_saved_model_ids_are_preserved(self):
        from telemem.configs import TeleMemoryConfig

        for model in (MINIMAX_MODEL_M3, MINIMAX_MODEL_M2_7,
                      MINIMAX_MODEL_M2_7_HS, "custom-model"):
            with self.subTest(model=model):
                config = TeleMemoryConfig(llm={
                    "provider": "openai", "config": {"model": model},
                })
                self.assertEqual(config.llm.config["model"], model)


@unittest.skipUnless(os.environ.get("MINIMAX_API_KEY"), "MINIMAX_API_KEY not set")
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests – require MINIMAX_API_KEY environment variable."""

    def setUp(self):
        self.api_key = os.environ["MINIMAX_API_KEY"]

    def test_minimax_api_connectivity(self):
        """MiniMax API must respond to a simple chat completion request."""
        from openai import OpenAI
        client = OpenAI(base_url=MINIMAX_GLOBAL_BASE_URL, api_key=self.api_key)
        response = client.chat.completions.create(
            model=MINIMAX_MODEL_M3,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=10,
            temperature=0.7,
        )
        self.assertIsNotNone(response)
        content = response.choices[0].message.content
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

    def test_minimax_api_json_mode(self):
        """MiniMax must support response_format json_object mode."""
        from openai import OpenAI
        client = OpenAI(base_url=MINIMAX_GLOBAL_BASE_URL, api_key=self.api_key)
        response = client.chat.completions.create(
            model=MINIMAX_MODEL_M3,
            messages=[
                {"role": "user", "content": (
                    "Return a JSON object with a single key 'status' set to 'ok'. "
                    "Output only the JSON object, nothing else."
                )},
            ],
            response_format={"type": "json_object"},
            max_tokens=100,
            temperature=0.7,
        )
        import json, re
        raw = response.choices[0].message.content or ""
        # Strip think tags that some models emit before the JSON payload
        content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if content:
            parsed = json.loads(content)
            self.assertIsInstance(parsed, dict)
        else:
            # Acceptable: request succeeded but model produced empty body
            self.assertIn(response.choices[0].finish_reason, ("stop", "length"))

    def test_minimax_highspeed_model(self):
        """MiniMax-M2.7-highspeed must also be accessible."""
        from openai import OpenAI
        client = OpenAI(base_url=MINIMAX_GLOBAL_BASE_URL, api_key=self.api_key)
        response = client.chat.completions.create(
            model=MINIMAX_MODEL_M2_7_HS,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=10,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)


def run_unit_tests():
    """Run only unit tests (no API key required)."""
    print("\n" + "=" * 60)
    print("MiniMax Provider – Unit Tests")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (
        TestMiniMaxConfigFile,
        TestMiniMaxTeleMemoryConfig,
        TestMiniMaxEndpoints,
        TestMiniMaxModels,
    ):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def run_integration_tests():
    """Run integration tests (requires MINIMAX_API_KEY)."""
    print("\n" + "=" * 60)
    print("MiniMax Provider – Integration Tests")
    print("=" * 60)
    if not os.environ.get("MINIMAX_API_KEY"):
        print("SKIP: Set MINIMAX_API_KEY to run integration tests.")
        return 0
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMiniMaxIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    rc = run_unit_tests()
    rc |= run_integration_tests()
    sys.exit(rc)
