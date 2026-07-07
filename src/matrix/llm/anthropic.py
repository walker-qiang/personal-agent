"""Anthropic Claude API client."""

from __future__ import annotations

import json
from typing import Any, Iterator

from .errors import LLMError
from .http import post_json_stream, post_json_with_retry
from .protocol import FunctionCallResult, ToolCall
from .truncate import truncate_messages


# Maximum characters per message before truncation
_DEFAULT_MAX_MESSAGE_CHARS = 8000


class AnthropicClient:
    """LLM client for Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-latest",
        max_tokens: int = 8192,
        timeout_sec: float = 45.0,
        max_message_chars: int = _DEFAULT_MAX_MESSAGE_CHARS,
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.max_message_chars = max_message_chars

    def _truncate(self, messages: list[dict[str, str]], system: str) -> list[dict[str, str]]:
        if self.max_message_chars <= 0:
            return messages
        return truncate_messages(
            messages,
            system_prompt=system,
            max_tokens=self.max_message_chars // 2,
            reserve_tokens=500,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._truncate(messages, system),
        }
        data = post_json_with_retry(
            "https://api.anthropic.com/v1/messages",
            payload,
            self._headers(),
            self.timeout_sec,
        )
        try:
            chunks = data["content"]
            return "".join(
                str(chunk.get("text", "")) for chunk in chunks if chunk.get("type") == "text"
            )
        except (KeyError, TypeError) as err:
            raise LLMError("Anthropic response did not include text content") from err

    def stream_complete(self, system: str, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream completion tokens from Anthropic Messages API.

        Uses SSE streaming (stream=True). Yields text delta chunks.
        """
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._truncate(messages, system),
            "stream": True,
        }
        for raw in post_json_stream(
            "https://api.anthropic.com/v1/messages",
            payload,
            self._headers(),
            self.timeout_sec,
        ):
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        yield text
            elif event.get("type") == "message_stop":
                return

    def function_call(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> FunctionCallResult:
        """Call Anthropic with native tool use.

        Anthropic uses a different tool format. Convert from OpenAI-compatible format.
        Returns a FunctionCallResult with either content or tool_calls.
        """
        # Convert tools to Anthropic format
        anthropic_tools = []
        for t in tools:
            anthropic_tools.append({
                "name": t["name"],
                "description": t["description"],
                "input_schema": t.get("input_schema", {
                    "type": "object",
                    "properties": {},
                }),
            })

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._truncate(messages, system),
            "tools": anthropic_tools,
        }

        # Map tool_choice to Anthropic format
        if tool_choice == "auto":
            payload["tool_choice"] = {"type": "auto"}
        elif tool_choice == "any":
            payload["tool_choice"] = {"type": "any"}
        elif tool_choice == "none":
            # Anthropic doesn't support "none" tool_choice; omit tools
            payload.pop("tools", None)
            payload.pop("tool_choice", None)

        data = post_json_with_retry(
            "https://api.anthropic.com/v1/messages",
            payload,
            self._headers(),
            self.timeout_sec,
        )

        result = FunctionCallResult(
            content="",
            finish_reason=data.get("stop_reason", "end_turn"),
        )

        try:
            for block in data.get("content", []):
                if block.get("type") == "text":
                    result.content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    result.tool_calls.append(
                        ToolCall(
                            name=block.get("name", ""),
                            arguments=block.get("input", {}),
                        )
                    )
                    result.finish_reason = "tool_calls"
        except (KeyError, TypeError) as err:
            raise LLMError("Anthropic tool use response parsing failed") from err

        return result