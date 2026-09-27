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
from matrix.llm.deepseek import _ensure_tool_call_outputs


def _assistant_tool_call(call_id: str, name: str = "search") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }],
    }


def _tool_output(call_id: str, text: str = '{"ok": true}') -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def _assert_paired(items: list[dict]) -> None:
    """Every function_call must be followed by its function_call_output."""
    calls = [i for i in items if i.get("type") == "function_call"]
    outputs = {i.get("call_id") for i in items if i.get("type") == "function_call_output"}
    assert calls, "expected at least one function_call item"
    for call in calls:
        assert call.get("call_id") in outputs, (
            f"missing function_call_output for {call.get('call_id')}"
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


class TestToolCallOutputPairing:
    """Responses API rejects unmatched function_call items with 400.

    ``No tool output found for tool call <id>`` happens whenever an assistant
    message carries tool_calls whose tool results were never appended (tool
    raised, was skipped, or dropped by context truncation).
    """

    def test_leaves_complete_pairs_untouched(self):
        messages = [
            {"role": "user", "content": "查一下"},
            _assistant_tool_call("call-1"),
            _tool_output("call-1"),
        ]
        assert _ensure_tool_call_outputs(messages) == messages

    def test_injects_placeholder_for_missing_output(self):
        messages = [
            {"role": "user", "content": "查一下"},
            _assistant_tool_call("call-1"),
        ]
        repaired = _ensure_tool_call_outputs(messages)

        assert len(repaired) == 3
        assert repaired[2]["role"] == "tool"
        assert repaired[2]["tool_call_id"] == "call-1"
        assert repaired[2]["content"]

    def test_fills_every_gap_in_a_multi_call_message(self):
        messages = [
            _assistant_tool_call("call-1"),
            _tool_output("call-1"),
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call-2", "function": {"name": "a", "arguments": "{}"}},
                {"id": "call-3", "function": {"name": "b", "arguments": "{}"}},
            ]},
            _tool_output("call-3"),
        ]
        repaired = _ensure_tool_call_outputs(messages)

        paired = {m.get("tool_call_id") for m in repaired if m["role"] == "tool"}
        assert paired == {"call-1", "call-2", "call-3"}

    def test_keeps_indexes_stable_across_multiple_gaps(self):
        messages = [
            _assistant_tool_call("call-1"),          # gap
            _assistant_tool_call("call-2"),          # gap
            _tool_output("call-2"),
        ]
        repaired = _ensure_tool_call_outputs(messages)

        order = [
            (m["role"], m.get("tool_call_id"))
            for m in repaired
            if m["role"] == "tool"
        ]
        assert order == [("tool", "call-1"), ("tool", "call-2")]

    def test_converted_input_items_are_fully_paired(self):
        client = DeepSeekClient(api_key="test-key")
        items = client._convert_messages_to_input([
            {"role": "user", "content": "查一下"},
            _assistant_tool_call("call-1"),
        ])

        _assert_paired(items)

    def test_payload_input_is_paired_after_truncation_drops_a_result(self):
        """A tool result dropped by truncation must not break the request."""
        client = DeepSeekClient(api_key="test-key")
        payload = client._build_payload(
            "system",
            [
                {"role": "user", "content": "查一下"},
                _assistant_tool_call("call-1"),
            ],
            tools=[{
                "type": "function",
                "name": "search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            }],
        )

        _assert_paired(payload["input"])

    def test_ignores_tool_calls_without_ids(self):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "a", "arguments": "{}"}},
            ]},
        ]
        assert _ensure_tool_call_outputs(messages) == messages


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
