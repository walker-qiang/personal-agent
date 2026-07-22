"""Anthropic Claude API client."""

from __future__ import annotations

import json
from typing import Any, Iterator

from .errors import LLMError
from .http import post_json_stream, post_json_with_retry
from .protocol import FunctionCallResult, ToolCall, parse_json_response
from .truncate import truncate_messages


# Maximum characters per message before truncation
_DEFAULT_MAX_MESSAGE_CHARS = 16000


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

    def _truncate(self, messages: list[dict[str, Any]], system: str) -> list[dict[str, Any]]:
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

    def complete(self, system: str, messages: list[dict[str, Any]], temperature: float | None = None) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._truncate(messages, system),
        }
        if temperature is not None:
            payload["temperature"] = temperature
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

    def complete_json(
        self,
        system: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Call Anthropic and return parsed JSON.

        Anthropic doesn't support response_format like OpenAI. Instead we use
        the "forced single tool call" pattern: define a dummy tool with the
        expected JSON schema and force the model to call it. The tool arguments
        are guaranteed to be valid JSON.

        Falls back to prompt-based JSON with parse_json_response if the tool
        call approach fails.
        """
        # Build the dummy tool definition
        tool_schema = schema or {"type": "object", "properties": {}}
        tool_def = {
            "name": "return_json",
            "description": "Return the structured result as JSON",
            "input_schema": tool_schema,
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._truncate(messages, system),
            "tools": [tool_def],
            "tool_choice": {"type": "tool", "name": "return_json"},
        }
        if temperature is not None:
            payload["temperature"] = temperature

        data = post_json_with_retry(
            "https://api.anthropic.com/v1/messages",
            payload,
            self._headers(),
            self.timeout_sec,
        )

        # Extract tool call arguments (guaranteed JSON from Anthropic)
        try:
            for block in data["content"]:
                if block.get("type") == "tool_use" and block.get("name") == "return_json":
                    return block["input"]
        except (KeyError, TypeError):
            pass

        # Fallback: try parsing text content
        try:
            text = "".join(
                str(chunk.get("text", "")) for chunk in data.get("content", [])
                if chunk.get("type") == "text"
            )
            return parse_json_response(text)
        except Exception as err:
            raise LLMError(f"Anthropic JSON output could not be parsed: {err}") from err

    def stream_complete(self, system: str, messages: list[dict[str, Any]], temperature: float | None = None) -> Iterator[str]:
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
        if temperature is not None:
            payload["temperature"] = temperature
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
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float | None = None,
    ) -> FunctionCallResult:
        """Call Anthropic with native tool use.

        Anthropic uses a different tool format. Convert from OpenAI-compatible format.
        Returns a FunctionCallResult with either content or tool_calls.
        Supports multi-turn tool messages (role="tool" with tool_call_id).
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

        # Convert messages to Anthropic format
        anthropic_messages = self._to_anthropic_messages(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": anthropic_messages,
            "tools": anthropic_tools,
        }
        if temperature is not None:
            payload["temperature"] = temperature

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
                            id=block.get("id", ""),
                            name=block.get("name", ""),
                            arguments=block.get("input", {}),
                        )
                    )
                    result.finish_reason = "tool_calls"
        except (KeyError, TypeError) as err:
            raise LLMError("Anthropic tool use response parsing failed") from err

        return result

    def _to_anthropic_messages(
        self, messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-compatible messages to Anthropic format.

        Handles: user, assistant, assistant+tool_calls, tool messages.
        """
        converted: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "tool":
                # Anthropic format: tool_result content block
                converted.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": content if isinstance(content, str) else json.dumps(content),
                        }
                    ],
                })
            elif role == "assistant" and "tool_calls" in msg:
                content_blocks: list[dict[str, Any]] = []
                if content:
                    content_blocks.append({"type": "text", "text": content})
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": json.loads(func.get("arguments", "{}"))
                        if isinstance(func.get("arguments"), str)
                        else func.get("arguments", {}),
                    })
                converted.append({"role": "assistant", "content": content_blocks})
            elif role == "assistant":
                converted.append({"role": "assistant", "content": content})
            elif role == "user":
                if isinstance(content, list):
                    # Multi-modal user message: convert content blocks
                    anthropic_blocks = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                anthropic_blocks.append({"type": "text", "text": block.get("text", "")})
                            elif block.get("type") == "image_url":
                                image_url = block.get("image_url", {})
                                url = image_url.get("url", "")
                                # Handle data: URLs (base64 encoded)
                                if url.startswith("data:"):
                                    # Parse data:image/png;base64,xxxxx
                                    header, b64_data = url.split(",", 1)
                                    media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
                                    anthropic_blocks.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": b64_data,
                                        },
                                    })
                                else:
                                    # URL-based image (not supported by Anthropic, skip)
                                    anthropic_blocks.append({"type": "text", "text": f"[图片: {url}]"})
                            else:
                                anthropic_blocks.append(block)
                    converted.append({"role": "user", "content": anthropic_blocks})
                else:
                    converted.append({"role": "user", "content": content})
            else:
                converted.append({"role": "user", "content": str(content)})

        return self._truncate(converted, "")