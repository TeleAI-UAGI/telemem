"""Offline contract tests for the DeepSeek Harness Cordis integration."""

import re
import unittest
from pathlib import Path

import yaml

from telemem import __version__
from test_mcp import EXPECTED_TOOLS


OVERLAY_PATH = Path(__file__).parent.parent / "examples" / "deepseek-harness.cordis.yml"


class CordisLoader(yaml.SafeLoader):
    """Parse Cordis ``!!js`` values as inert strings for structural tests."""


CordisLoader.add_constructor(
    "tag:yaml.org,2002:js",
    lambda loader, node: loader.construct_scalar(node),
)


def _config():
    with open(OVERLAY_PATH, "r", encoding="utf-8") as stream:
        overlay = yaml.load(stream, Loader=CordisLoader)
    return overlay[0]["insert"][0]["config"]


class TestDeepSeekHarnessOverlay(unittest.TestCase):
    def test_overlay_uses_official_mcp_client_contract(self):
        self.assertTrue(OVERLAY_PATH.exists())
        with open(OVERLAY_PATH, "r", encoding="utf-8") as stream:
            overlay = yaml.load(stream, Loader=CordisLoader)

        self.assertEqual(len(overlay), 1)
        entry = overlay[0]["insert"][0]
        self.assertEqual(entry["id"], "memory-telemem")
        self.assertEqual(entry["name"], "@deepseek-ai/dsh-mcp-client")

        config = entry["config"]
        self.assertEqual(config["serverName"], "telemem")
        self.assertEqual(config["transport"], "stdio")
        self.assertEqual(config["command"], "uvx")
        self.assertTrue(config["failOnStartupError"])

    def test_overlay_pins_the_package_release(self):
        self.assertEqual(_config()["args"], [f"telemem=={__version__}"])

    def test_scrubbed_environment_is_forwarded_without_secrets(self):
        config = _config()
        self.assertEqual(
            set(config["env"]),
            {
                "DEEPSEEK_API_KEY",
                "OPENAI_API_KEY",
                "TELEMEM_CONFIG",
                "TELEMEM_DEFAULT_USER_ID",
            },
        )
        source = OVERLAY_PATH.read_text(encoding="utf-8")
        self.assertNotRegex(source, r"\bsk-[A-Za-z0-9]")
        for name in config["env"]:
            self.assertIn(f"process.env.{name}", config["env"][name])

    def test_public_tool_names_fit_deepseek_function_contract(self):
        for raw_name in EXPECTED_TOOLS:
            with self.subTest(tool=raw_name):
                public_name = f"mcp__telemem__{raw_name}"
                self.assertLessEqual(len(public_name), 64)
                self.assertRegex(public_name, re.compile(r"^[A-Za-z0-9_-]+$"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
