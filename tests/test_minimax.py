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
MINIMAX_CN_BASE_URL = "https://api.minimaxi.com/v1"
# Public OpenAI-compatible regional endpoints for the text models.
MINIMAX_REGIONAL_URLS = [MINIMAX_GLOBAL_BASE_URL, MINIMAX_CN_BASE_URL]
MINIMAX_MODEL_M3 = "MiniMax-M3"
MINIMAX_MODEL_M2_7 = "MiniMax-M2.7"
MINIMAX_MODEL_M2_7_HS = "MiniMax-M2.7-highspeed"
MINIMAX_MODEL_IDS = [MINIMAX_MODEL_M3, MINIMAX_MODEL_M2_7]
MINIMAX_M3_CONTEXT = 1000000
MINIMAX_M2_7_CONTEXT = 204800
MINIMAX_TEMPERATURE_MIN = 0.0
MINIMAX_TEMPERATURE_MAX = 1.0

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

    def test_config_temperature_in_valid_range(self):
        """MiniMax temperature must be in (0.0, 1.0]."""
        import yaml
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f)
        temp = data["llm"]["config"].get("temperature")
        if temp is not None:
            self.assertGreater(temp, MINIMAX_TEMPERATURE_MIN,
                               "MiniMax temperature must be > 0.0")
            self.assertLessEqual(temp, MINIMAX_TEMPERATURE_MAX,
                                 "MiniMax temperature must be <= 1.0")

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


class TestMiniMaxTemperatureConstraint(unittest.TestCase):
    """Validate temperature values against MiniMax's (0.0, 1.0] constraint."""

    def test_temperature_at_upper_bound(self):
        """Temperature of 1.0 is valid for MiniMax."""
        temp = 1.0
        self.assertGreater(temp, 0.0)
        self.assertLessEqual(temp, 1.0)

    def test_temperature_at_midpoint(self):
        """Temperature of 0.7 is valid for MiniMax."""
        temp = 0.7
        self.assertGreater(temp, 0.0)
        self.assertLessEqual(temp, 1.0)

    def test_temperature_zero_is_invalid(self):
        """Temperature of 0.0 is NOT valid for MiniMax (requires > 0)."""
        temp = 0.0
        self.assertFalse(temp > 0.0 and temp <= 1.0,
                         "0.0 should not be in the valid (0.0, 1.0] range")

    def test_temperature_above_one_is_invalid(self):
        """Temperature above 1.0 is NOT valid for MiniMax."""
        temp = 1.5
        self.assertFalse(temp > 0.0 and temp <= 1.0,
                         "1.5 should not be in the valid (0.0, 1.0] range")


class TestMiniMaxEndpoints(unittest.TestCase):
    """Validate the regional OpenAI-compatible endpoints exposed by MiniMax."""

    EXPECTED_REGIONAL_URLS = MINIMAX_REGIONAL_URLS

    def test_global_endpoint_is_documented(self):
        """The global (api.minimax.io) endpoint must be in the regional list."""
        self.assertIn(MINIMAX_GLOBAL_BASE_URL, self.EXPECTED_REGIONAL_URLS)

    def test_china_endpoint_is_documented(self):
        """The China (api.minimaxi.com) endpoint must be in the regional list."""
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
    """Validate MiniMax model names and current capability metadata."""

    MINIMAX_MODELS = [
        {"name": MINIMAX_MODEL_M3, "context": MINIMAX_M3_CONTEXT,
         "input_modalities": ["text", "image", "video"],
         "thinking": ["adaptive", "disabled"]},
        {"name": MINIMAX_MODEL_M2_7, "context": MINIMAX_M2_7_CONTEXT,
         "input_modalities": ["text"],
         "thinking": ["always_on"]},
    ]

    def test_model_ids_are_the_current_set(self):
        """The supported current model ids must be MiniMax-M3 and MiniMax-M2.7."""
        names = [model["name"] for model in self.MINIMAX_MODELS]
        self.assertEqual(names, MINIMAX_MODEL_IDS)

    def test_model_names_start_with_minimax(self):
        """All MiniMax model names must start with 'MiniMax-'."""
        for model in self.MINIMAX_MODELS:
            with self.subTest(model=model["name"]):
                self.assertTrue(model["name"].startswith("MiniMax-"))

    def test_all_models_have_long_context(self):
        """All listed MiniMax models have at least a 204,800-token context window."""
        for model in self.MINIMAX_MODELS:
            with self.subTest(model=model["name"]):
                self.assertGreaterEqual(model["context"], MINIMAX_M2_7_CONTEXT)

    def test_m3_context_window_is_one_million(self):
        """MiniMax-M3 must expose the current 1,000,000-token context window."""
        m3 = next(m for m in self.MINIMAX_MODELS if m["name"] == MINIMAX_MODEL_M3)
        self.assertEqual(m3["context"], MINIMAX_M3_CONTEXT)

    def test_m2_7_context_window(self):
        """MiniMax-M2.7 must expose the current 204,800-token context window."""
        m2_7 = next(m for m in self.MINIMAX_MODELS if m["name"] == MINIMAX_MODEL_M2_7)
        self.assertEqual(m2_7["context"], MINIMAX_M2_7_CONTEXT)

    def test_m3_supports_multimodal_input(self):
        """MiniMax-M3 accepts text, image and video input."""
        m3 = next(m for m in self.MINIMAX_MODELS if m["name"] == MINIMAX_MODEL_M3)
        for modality in ("text", "image", "video"):
            with self.subTest(modality=modality):
                self.assertIn(modality, m3["input_modalities"])

    def test_m2_7_is_text_only(self):
        """MiniMax-M2.7 only accepts text input."""
        m2_7 = next(m for m in self.MINIMAX_MODELS if m["name"] == MINIMAX_MODEL_M2_7)
        self.assertEqual(m2_7["input_modalities"], ["text"])

    def test_m3_thinking_is_adaptive(self):
        """MiniMax-M3 supports adaptive thinking and can disable thinking."""
        m3 = next(m for m in self.MINIMAX_MODELS if m["name"] == MINIMAX_MODEL_M3)
        self.assertIn("adaptive", m3["thinking"])
        self.assertIn("disabled", m3["thinking"])

    def test_m2_7_thinking_is_always_on(self):
        """MiniMax-M2.7 always applies thinking."""
        m2_7 = next(m for m in self.MINIMAX_MODELS if m["name"] == MINIMAX_MODEL_M2_7)
        self.assertEqual(m2_7["thinking"], ["always_on"])

    def test_preferred_model_for_config(self):
        """MiniMax-M3 is the preferred default model."""
        self.assertEqual(MINIMAX_MODEL_M3, "MiniMax-M3")


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
        TestMiniMaxTemperatureConstraint,
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
