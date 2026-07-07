"""DeepSeek API client."""

from __future__ import annotations

import json
from typing import Any, Iterator

from .errors import LLMError
from .http import post_json_stream, post_json_with_retry
from .protocol import FunctionCallResult, ToolCall
from .truncate import truncate_messages


# Maximum characters per message before truncation
_DEFAULT_MAX_MESSAGE_CHARS = 8000


class DeepSeekClient:
    """LLM client for DeepSeek chat completions API."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_tokens: int = 8192,
        timeout_sec: float = 45.0,
        max_message_chars: int = _DEFAULT_MAX_MESSAGE_CHARS,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.max_message_chars = max_message_chars

    def _build_payload(self, system: str, messages: list[dict[str, str]]) -> dict:
        if self.max_message_chars > 0:
            messages = truncate_messages(
                messages,
                system_prompt=system,
                max_tokens=self.max_message_chars // 2,  # rough: 2 chars per token
                reserve_tokens=500,
            )
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}, *messages],
        }

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = self._build_payload(system, messages)
        data = post_json_with_retry(url, payload, self._headers(), self.timeout_sec)
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as err:
            raise LLMError("DeepSeek response did not include message content") from err

    def stream_complete(self, system: str, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream completion tokens from DeepSeek API.

        Uses SSE streaming (stream=True). Yields content delta chunks.
        """
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = self._build_payload(system, messages)
        payload["stream"] = True
        payload.setdefault("stream_options", {"include_usage": False})

        for raw in post_json_stream(url, payload, self._headers(), self.timeout_sec):
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            try:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (KeyError, IndexError, TypeError):
                continue

    def function_call(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> FunctionCallResult:
        """Call DeepSeek with native function calling.

        DeepSeek uses OpenAI-compatible tool calling format.
        Returns a FunctionCallResult with either content or tool_calls.
        """
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = self._build_payload(system, messages)

        # Sanitize tool names: replace dots with underscores for APIs that
        # only accept ^[a-zA-Z0-9_-]+$ (e.g. Agnes AI, strict OpenAI-compatible).
        name_map: dict[str, str] = {}  # sanitized → original
        payload["tools"] = []
        for t in tools:
            original = t["name"]
            sanitized = original.replace(".", "_")
            name_map[sanitized] = original
            payload["tools"].append({
                "type": "function",
                "function": {
                    "name": sanitized,
                    "description": t["description"],
                    "parameters": t.get("input_schema", {}),
                },
            })
        payload["tool_choice"] = tool_choice

        data = post_json_with_retry(url, payload, self._headers(), self.timeout_sec)
        try:
            choice = data["choices"][0]
            message = choice.get("message", {})
            finish = choice.get("finish_reason", "stop")

            result = FunctionCallResult(
                content=message.get("content") or "",
                finish_reason=finish,
            )

            raw_tool_calls = message.get("tool_calls", [])
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                # Map sanitized name back to original (e.g. "finance_holdings_summary" → "finance.holdings_summary")
                name = name_map.get(name, name)
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result.tool_calls.append(ToolCall(name=name, arguments=args))

            return result
        except (KeyError, IndexError, TypeError) as err:
            raise LLMError("DeepSeek function call response parsing failed") from err