"""Tests for LLM client layer: retry, auth, protocol."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from matrix.llm import (
    CodexCLIClient,
    DeepSeekClient,
    LLMAuthError,
    LLMError,
    LLMTransientError,
    build_llm_client,
)


def _mock_response(text: str) -> dict:
    """Build a mock Responses API response."""
    return {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
    }


class TestDeepSeekClient:
    def test_converts_multimodal_content_to_responses_input(self):
        client = DeepSeekClient(api_key="test-key")
        items = client._convert_messages_to_input([{
            "role": "user",
            "content": [
                {"type": "text", "text": "描述图片"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                },
            ],
        }])

        assert items == [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "描述图片"},
                {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="},
            ],
        }]

    def test_retries_transient_error_once(self):
        """DeepSeek client should retry once on LLMTransientError."""
        calls = 0

        def fake_post_json(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise LLMTransientError("model provider returned 503: service busy")
            return _mock_response("ok")

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
            pytest.raises(LLMTransientError, match="after 4 attempts"),
        ):
            client.complete("system", [])

    def test_handles_missing_content(self):
        """DeepSeek should raise LLMError when response has no content."""
        def fake_post_json(*_args, **_kwargs):
            return {"output": [{"type": "message", "content": []}]}

        client = DeepSeekClient(api_key="test-key")
        with patch("matrix.llm.http.post_json", fake_post_json), pytest.raises(LLMError, match="content is empty"):
            client.complete("system", [])

    def test_uses_custom_base_url(self):
        """DeepSeek should use the configured base_url."""
        def fake_post_json(url, *_args, **_kwargs):
            assert "custom.api.com" in url
            return _mock_response("ok")

        client = DeepSeekClient(api_key="test-key", base_url="https://custom.api.com")
        with patch("matrix.llm.http.post_json", fake_post_json):
            assert client.complete("system", []) == "ok"


class TestBuildLLMClient:
    def test_builds_deepseek_by_default(self):
        client = build_llm_client(provider="deepseek", deepseek_api_key="test-key")
        assert isinstance(client, DeepSeekClient)

    def test_builds_deepseek_with_custom_model(self):
        client = build_llm_client(
            provider="deepseek",
            deepseek_api_key="test-key",
            model="deepseek-reasoner",
        )
        assert isinstance(client, DeepSeekClient)
        assert client.model == "deepseek-reasoner"


class TestCodexCLIClient:
    def test_materializes_data_uri_images_and_adds_cli_image_arguments(self, tmp_path):
        client = CodexCLIClient(binary="codex", workdir=str(tmp_path))
        image_paths = []
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "描述图片"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                },
            ],
        }]

        prepared = client._materialize_images(messages, str(tmp_path), image_paths)

        assert len(image_paths) == 1
        assert (tmp_path / "attachment-0.png").read_bytes() == b"image"
        assert prepared[0]["content"][1]["type"] == "text"
        command = client._build_command("prompt", image_paths)
        assert command[-3:] == ["prompt", "--image", image_paths[0]]


class TestLLMErrorHierarchy:
    def test_transient_is_llm_error(self):
        assert issubclass(LLMTransientError, LLMError)

    def test_auth_is_llm_error(self):
        assert issubclass(LLMAuthError, LLMError)

    def test_transient_not_caught_by_auth(self):
        err = LLMTransientError("timeout")
        assert not isinstance(err, LLMAuthError)
