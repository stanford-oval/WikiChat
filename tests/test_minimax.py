"""Tests for MiniMax LLM provider integration."""

import os
import sys
import unittest
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, "./")


# ---------------------------------------------------------------------------
# Unit tests – no API calls, run unconditionally
# ---------------------------------------------------------------------------


class TestMiniMaxConfig(unittest.TestCase):
    """Verify MiniMax is properly declared in llm_config.yaml."""

    @classmethod
    def setUpClass(cls):
        with open("llm_config.yaml") as f:
            cls.config = yaml.safe_load(f)
        cls.endpoints = cls.config.get("llm_endpoints", [])

    def _find_minimax_endpoint(self):
        for ep in self.endpoints:
            if ep.get("api_base") == "https://api.minimax.io/v1":
                return ep
        return None

    def test_minimax_endpoint_exists(self):
        """MiniMax endpoint should be present in llm_config.yaml."""
        ep = self._find_minimax_endpoint()
        self.assertIsNotNone(ep, "MiniMax endpoint not found in llm_config.yaml")

    def test_minimax_api_key_env_var(self):
        """MiniMax endpoint should reference the MINIMAX_API_KEY env var."""
        ep = self._find_minimax_endpoint()
        self.assertEqual(ep["api_key"], "MINIMAX_API_KEY")

    def test_minimax_api_base(self):
        """MiniMax endpoint should use https://api.minimax.io/v1."""
        ep = self._find_minimax_endpoint()
        self.assertEqual(ep["api_base"], "https://api.minimax.io/v1")

    def test_minimax_engine_map_has_m27(self):
        """Engine map should include minimax-m27."""
        ep = self._find_minimax_endpoint()
        self.assertIn("minimax-m27", ep["engine_map"])
        self.assertEqual(ep["engine_map"]["minimax-m27"], "openai/MiniMax-M2.7")

    def test_minimax_engine_map_has_m27_highspeed(self):
        """Engine map should include minimax-m27-highspeed."""
        ep = self._find_minimax_endpoint()
        self.assertIn("minimax-m27-highspeed", ep["engine_map"])
        self.assertEqual(
            ep["engine_map"]["minimax-m27-highspeed"],
            "openai/MiniMax-M2.7-highspeed",
        )

    def test_minimax_engine_map_has_m25(self):
        """Engine map should include minimax-m25."""
        ep = self._find_minimax_endpoint()
        self.assertIn("minimax-m25", ep["engine_map"])
        self.assertEqual(ep["engine_map"]["minimax-m25"], "openai/MiniMax-M2.5")

    def test_minimax_engine_map_has_m25_highspeed(self):
        """Engine map should include minimax-m25-highspeed."""
        ep = self._find_minimax_endpoint()
        self.assertIn("minimax-m25-highspeed", ep["engine_map"])
        self.assertEqual(
            ep["engine_map"]["minimax-m25-highspeed"],
            "openai/MiniMax-M2.5-highspeed",
        )

    def test_minimax_engine_names_unique(self):
        """MiniMax engine names should not collide with other providers."""
        minimax_ep = self._find_minimax_endpoint()
        minimax_engines = set(minimax_ep["engine_map"].keys())
        for ep in self.endpoints:
            if ep is minimax_ep:
                continue
            other_engines = set(ep.get("engine_map", {}).keys())
            collision = minimax_engines & other_engines
            self.assertFalse(
                collision,
                f"Engine name collision with another endpoint: {collision}",
            )

    def test_minimax_litellm_prefix(self):
        """All MiniMax model values should use openai/ prefix for LiteLLM routing."""
        ep = self._find_minimax_endpoint()
        for engine_name, model_name in ep["engine_map"].items():
            self.assertTrue(
                model_name.startswith("openai/"),
                f"Model {model_name} for engine {engine_name} should use openai/ prefix",
            )


class TestBackendServerMiniMax(unittest.TestCase):
    """Verify MiniMax models appear in the Chainlit web UI dropdown."""

    @classmethod
    def setUpClass(cls):
        with open("backend_server.py") as f:
            cls.source = f.read()

    def test_minimax_m27_in_dropdown(self):
        self.assertIn("minimax-m27", self.source)

    def test_minimax_m27_highspeed_in_dropdown(self):
        self.assertIn("minimax-m27-highspeed", self.source)


class TestReadmeMiniMax(unittest.TestCase):
    """Verify MiniMax is mentioned in README.md."""

    @classmethod
    def setUpClass(cls):
        with open("README.md") as f:
            cls.readme = f.read()

    def test_minimax_mentioned_in_readme(self):
        self.assertIn("MiniMax", self.readme)

    def test_minimax_api_key_in_readme(self):
        self.assertIn("MINIMAX_API_KEY", self.readme)

    def test_minimax_platform_link_in_readme(self):
        self.assertIn("https://platform.minimaxi.com/", self.readme)


class TestMiniMaxEngineResolution(unittest.TestCase):
    """Test that ChainLite config loading correctly resolves MiniMax engines."""

    def test_config_engine_map_resolution(self):
        """Simulate how ChainLite picks an engine from llm_config.yaml."""
        with open("llm_config.yaml") as f:
            config = yaml.safe_load(f)
        endpoints = config.get("llm_endpoints", [])

        # Simulate chainlite's pick_llm_resource by filtering endpoints with
        # the target engine and a valid (non-empty) api_key env var set.
        # We patch the env var to make the endpoint qualify.
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-12345"}):
            for engine_name in [
                "minimax-m27",
                "minimax-m27-highspeed",
                "minimax-m25",
                "minimax-m25-highspeed",
            ]:
                matched = [
                    ep
                    for ep in endpoints
                    if engine_name in ep.get("engine_map", {})
                    and (
                        "api_key" not in ep
                        or os.environ.get(ep["api_key"])
                    )
                ]
                self.assertTrue(
                    len(matched) > 0,
                    f"No endpoint found for engine '{engine_name}' with valid API key",
                )
                # Verify the resolved model name
                model = matched[0]["engine_map"][engine_name]
                self.assertTrue(
                    model.startswith("openai/MiniMax-M2"),
                    f"Unexpected model name: {model}",
                )
                # Verify api_base is set
                self.assertEqual(
                    matched[0]["api_base"],
                    "https://api.minimax.io/v1",
                )


class TestMiniMaxTemperatureHandling(unittest.TestCase):
    """Verify temperature defaults are compatible with MiniMax."""

    def test_default_temperature_is_zero(self):
        """ChainLite default temperature is 0, which MiniMax accepts."""
        # MiniMax API accepts temperature=0 (confirmed since 2026-03-17).
        # The default in ChatLiteLLM is 0, so no clamping is needed.
        from chainlite.chat_lite_llm import ChatLiteLLM

        llm = ChatLiteLLM(model="openai/MiniMax-M2.7")
        self.assertEqual(llm.temperature, 0)

    def test_custom_temperature(self):
        """ChatLiteLLM should accept custom temperatures for MiniMax."""
        from chainlite.chat_lite_llm import ChatLiteLLM

        llm = ChatLiteLLM(model="openai/MiniMax-M2.7", temperature=0.7)
        self.assertEqual(llm.temperature, 0.7)

    def test_api_params_include_base_url(self):
        """ChatLiteLLM _default_params should include api_base when set."""
        from chainlite.chat_lite_llm import ChatLiteLLM

        llm = ChatLiteLLM(
            model="openai/MiniMax-M2.7",
            api_base="https://api.minimax.io/v1",
            api_key="test-key",
        )
        params = llm._default_params
        self.assertEqual(params["api_base"], "https://api.minimax.io/v1")
        self.assertEqual(params["api_key"], "test-key")
        self.assertEqual(params["model"], "openai/MiniMax-M2.7")


# ---------------------------------------------------------------------------
# Integration tests – require MINIMAX_API_KEY, skipped otherwise
# ---------------------------------------------------------------------------

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
skip_no_key = pytest.mark.skipif(
    not MINIMAX_API_KEY,
    reason="MINIMAX_API_KEY not set",
)


def _strip_think_tags(text: str) -> str:
    """Strip <think>...</think> tags from MiniMax M2.7 reasoning output."""
    import re

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@skip_no_key
def test_minimax_m27_completion():
    """Smoke-test: send a simple prompt to MiniMax-M2.7 via LiteLLM."""
    import litellm

    response = litellm.completion(
        model="openai/MiniMax-M2.7",
        api_base="https://api.minimax.io/v1",
        api_key=MINIMAX_API_KEY,
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=256,
        temperature=0,
    )
    text = _strip_think_tags(response.choices[0].message.content or "")
    assert text, "Response should not be empty after stripping think tags"
    assert "4" in text


@skip_no_key
def test_minimax_m25_highspeed_completion():
    """Smoke-test: send a simple prompt to MiniMax-M2.5-highspeed."""
    import litellm

    response = litellm.completion(
        model="openai/MiniMax-M2.5-highspeed",
        api_base="https://api.minimax.io/v1",
        api_key=MINIMAX_API_KEY,
        messages=[{"role": "user", "content": "Name the capital of France in one word."}],
        max_tokens=256,
        temperature=0,
    )
    text = _strip_think_tags(response.choices[0].message.content or "")
    assert text, "Response should not be empty"
    assert "Paris" in text


@skip_no_key
def test_minimax_streaming():
    """Test that streaming works with MiniMax models via LiteLLM."""
    import litellm

    response = litellm.completion(
        model="openai/MiniMax-M2.5-highspeed",
        api_base="https://api.minimax.io/v1",
        api_key=MINIMAX_API_KEY,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=64,
        temperature=0,
        stream=True,
    )
    chunks = []
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            chunks.append(delta)
    full_output = "".join(chunks)
    assert len(full_output) > 0
