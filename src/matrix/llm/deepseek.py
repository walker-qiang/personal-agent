"""DeepSeek API client."""

from __future__ import annotations

import json
from typing import Any, Iterator

from .errors import LLMError
from .http import post_json_stream, post_json_with_retry
from .protocol import FunctionCallResult, ToolCall, parse_json_response
from .truncate import truncate_messages


# Maximum characters per message before truncation
_DEFAULT_MAX_MESSAGE_CHARS = 16000


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

    def _build_payload(
        self, system: str, messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if self.max_message_chars > 0:
            messages = truncate_messages(
                messages,
                system_prompt=system,
                max_tokens=self.max_message_chars // 2,
                reserve_tokens=500,
            )
        payload["messages"].extend(messages)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    def complete(self, system: str, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = self._build_payload(system, messages, temperature=temperature)
        data = post_json_with_retry(url, payload, self._headers(), self.timeout_sec)
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as err:
            raise LLMError("DeepSeek response did not include message content") from err

    def complete_json(
        self,
        system: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Call DeepSeek with response_format=json_object for guaranteed JSON output.

        DeepSeek (OpenAI-compatible) supports response_format={"type": "json_object"}
        which forces the model to output valid JSON. The system prompt must contain
        the word "json" for this to work.
        """
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = self._build_payload(system, messages, temperature=temperature)
        # Force JSON output mode
        payload["response_format"] = {"type": "json_object"}
        data = post_json_with_retry(url, payload, self._headers(), self.timeout_sec)
        try:
            content = str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as err:
            raise LLMError("DeepSeek response did not include message content") from err
        try:
            return parse_json_response(content)
        except Exception as err:
            raise LLMError(f"DeepSeek JSON output could not be parsed: {err}") from err

    def stream_complete(self, system: str, messages: list[dict[str, str]], temperature: float | None = None) -> Iterator[str]:
        """Stream completion tokens from DeepSeek API.

        Uses SSE streaming (stream=True). Yields content delta chunks.
        """
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = self._build_payload(system, messages, temperature=temperature)
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
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float | None = None,
    ) -> FunctionCallResult:
        """Call DeepSeek/Agnes with native function calling.

        Returns a FunctionCallResult with either content or tool_calls.
        Supports multi-turn tool messages:
        - role="tool" with tool_call_id + content
        - role="assistant" with tool_calls[] + content
        """
        url = self.base_url.rstrip("/") + "/chat/completions"

        # Sanitize tool names: replace dots with underscores for APIs that
        # only accept ^[a-zA-Z0-9_-]+$ (e.g. Agnes AI, strict OpenAI-compatible).
        name_map: dict[str, str] = {}  # sanitized → original
        reverse_map: dict[str, str] = {}  # original → sanitized
        api_tools: list[dict[str, Any]] = []
        for t in tools:
            original = t["name"]
            sanitized = original.replace(".", "_")
            name_map[sanitized] = original
            reverse_map[original] = sanitized
            api_tools.append({
                "type": "function",
                "function": {
                    "name": sanitized,
                    "description": t["description"],
                    "parameters": t.get("input_schema", {}),
                },
            })

        # Normalize messages: ensure tool_calls use sanitized names
        api_messages: list[dict[str, Any]] = []
        for msg in messages:
            api_msg = dict(msg)
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                normalized_tcs = []
                for tc in msg["tool_calls"]:
                    ntc = dict(tc)
                    if "function" in ntc:
                        ntc["function"] = dict(ntc["function"])
                        ntc["function"]["name"] = reverse_map.get(
                            ntc["function"]["name"], ntc["function"]["name"],
                        )
                    normalized_tcs.append(ntc)
                api_msg["tool_calls"] = normalized_tcs
            api_messages.append(api_msg)

        payload = self._build_payload(
            system, api_messages, tools=api_tools, tool_choice=tool_choice, temperature=temperature,
        )

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
                # Map sanitized name back to original
                name = name_map.get(name, name)
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result.tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=name,
                    arguments=args,
                ))

            return result
        except (KeyError, IndexError, TypeError) as err:
            raise LLMError("DeepSeek function call response parsing failed") from err