"""Tests for LLM client layer: retry, auth, protocol."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from matrix.llm import (
    AnthropicClient,
    DeepSeekClient,
    LLMAuthError,
    LLMError,
    LLMTransientError,
    build_llm_client,
)


class TestDeepSeekClient:
    def test_retries_transient_error_once(self):
        """DeepSeek client should retry once on LLMTransientError."""
        calls = 0

        def fake_post_json(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise LLMTransientError("model provider returned 503: service busy")
            return {"choices": [{"message": {"content": "ok"}}]}

        client = DeepSeekClient(api_key="test-key", timeout_sec=5)
        with patch("matrix.llm.http.post_json", fake_post_json), patch("matrix.llm.http.time.sleep"):
            assert client.complete("system", []) == "ok"
        assert calls == 2

    def test_does_not_retry_auth_error(self):
        """DeepSeek client should NOT retry on LLMAuthError."""
        calls = 0

        def fake_post_json(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise LLMAuthError("model provider authentication failed")

        client = DeepSeekClient(api_key="test-key", timeout_sec=5)
        with patch("matrix.llm.http.post_json", fake_post_json), pytest.raises(LLMAuthError):
            client.complete("system", [])
        assert calls == 1

    def test_reports_transient_failure_after_retry_limit(self):
        """DeepSeek should raise LLMTransientError after max retries."""
        client = DeepSeekClient(api_key="test-key", timeout_sec=5)
        with (
            patch("matrix.llm.http.post_json", side_effect=LLMTransientError("timed out")),
            patch("matrix.llm.http.time.sleep"),
            pytest.raises(LLMTransientError, match="after 2 attempts"),
        ):
            client.complete("system", [])

    def test_handles_missing_content(self):
        """DeepSeek should raise LLMError when response has no content."""
        def fake_post_json(*_args, **_kwargs):
            return {"choices": [{"message": {}}]}

        client = DeepSeekClient(api_key="test-key")
        with patch("matrix.llm.http.post_json", fake_post_json), pytest.raises(LLMError, match="message content"):
            client.complete("system", [])

    def test_uses_custom_base_url(self):
        """DeepSeek should use the configured base_url."""
        def fake_post_json(url, *_args, **_kwargs):
            assert "custom.api.com" in url
            return {"choices": [{"message": {"content": "ok"}}]}

        client = DeepSeekClient(api_key="test-key", base_url="https://custom.api.com")
        with patch("matrix.llm.http.post_json", fake_post_json):
            assert client.complete("system", []) == "ok"


class TestAnthropicClient:
    def test_handles_text_content(self):
        """Anthropic should extract text from content blocks."""
        def fake_post_json(*_args, **_kwargs):
            return {"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}]}

        client = AnthropicClient(api_key="test-key")
        with patch("matrix.llm.http.post_json", fake_post_json):
            assert client.complete("system", []) == "hello world"

    def test_handles_empty_content_list(self):
        """Anthropic should return empty string when content list is empty."""
        def fake_post_json(*_args, **_kwargs):
            return {"content": []}

        client = AnthropicClient(api_key="test-key")
        with patch("matrix.llm.http.post_json", fake_post_json):
            assert client.complete("system", []) == ""


class TestBuildLLMClient:
    def test_builds_deepseek_by_default(self):
        client = build_llm_client(provider="deepseek", deepseek_api_key="test-key")
        assert isinstance(client, DeepSeekClient)

    def test_builds_anthropic(self):
        client = build_llm_client(provider="anthropic", anthropic_api_key="claude-key")
        assert isinstance(client, AnthropicClient)

    def test_builds_deepseek_with_custom_model(self):
        client = build_llm_client(
            provider="deepseek",
            deepseek_api_key="test-key",
            model="deepseek-reasoner",
        )
        assert isinstance(client, DeepSeekClient)
        assert client.model == "deepseek-reasoner"


class TestLLMErrorHierarchy:
    def test_transient_is_llm_error(self):
        assert issubclass(LLMTransientError, LLMError)

    def test_auth_is_llm_error(self):
        assert issubclass(LLMAuthError, LLMError)

    def test_transient_not_caught_by_auth(self):
        err = LLMTransientError("timeout")
        assert not isinstance(err, LLMAuthError)