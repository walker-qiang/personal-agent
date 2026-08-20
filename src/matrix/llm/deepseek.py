"""DeepSeek API client — Responses API.

Uses the DeepSeek Responses API (POST /responses) which is compatible with
OpenAI Responses API format. Key differences from the legacy Chat Completions API:

  - Endpoint: /responses (not /chat/completions)
  - System prompt → `instructions` parameter (not in `input`)
  - Messages → `input` items (role-based messages + function_call/function_call_output items)
  - Tools: flat format `{type, name, description, parameters}` (not nested under `function`)
  - Response: `output[]` array with typed items (message, function_call)
  - Streaming: semantic events (response.output_text.delta, response.completed, etc.)

The calling code (react.py, commander.py, etc.) still uses Chat Completions
message format. This client converts internally — the migration is fully
encapsulated here.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from .errors import LLMError
from .http import post_json_stream_events, post_json_with_retry
from .protocol import FunctionCallResult, ToolCall, parse_json_response
from .truncate import truncate_messages


logger = logging.getLogger("matrix.llm.deepseek")

# Maximum characters per message before truncation
_DEFAULT_MAX_MESSAGE_CHARS = 16000


class DeepSeekClient:
    """LLM client for DeepSeek Responses API."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
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

    # ── Payload building ────────────────────────────────────────────────────

    def _build_payload(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Build Responses API payload from Chat Completions-style messages."""
        if self.max_message_chars > 0:
            messages = truncate_messages(
                messages,
                system_prompt=system,
                max_tokens=self.max_message_chars // 2,
                reserve_tokens=500,
            )

        input_items = self._convert_messages_to_input(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": input_items,
            "max_output_tokens": self.max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if json_mode:
            payload["text"] = {"format": {"type": "json_object"}}
        return payload

    def _convert_messages_to_input(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert Chat Completions messages to Responses API input items.

        Handles:
        - {role: "user"/"assistant", content: "..."} → {role, content: [{type, text}]}
        - {role: "assistant", tool_calls: [...]} → {type: "function_call", ...} items
        - {role: "tool", tool_call_id, content} → {type: "function_call_output", ...}
        """
        items: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "tool":
                # Tool result → function_call_output
                items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": str(content) if content else "{}",
                })
            elif role == "assistant" and "tool_calls" in msg:
                # Assistant with tool calls → function_call items
                if content:
                    items.append({
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": str(content)}],
                    })
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    items.append({
                        "type": "function_call",
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "{}"),
                        "call_id": tc.get("id", ""),
                    })
            else:
                # Regular message (user/assistant/system)
                content_type = "input_text" if role == "user" else "output_text"
                if isinstance(content, list):
                    blocks: list[dict[str, Any]] = []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "text":
                            blocks.append({
                                "type": content_type,
                                "text": str(block.get("text", "")),
                            })
                        elif block.get("type") == "image_url":
                            image_url = block.get("image_url", {})
                            url = (
                                image_url.get("url", "")
                                if isinstance(image_url, dict) else ""
                            )
                            blocks.append({
                                "type": "input_image",
                                "image_url": url,
                            })
                        else:
                            blocks.append(block)
                    items.append({"role": role, "content": blocks})
                else:
                    items.append({
                        "role": role,
                        "content": [{"type": content_type, "text": str(content)}]
                        if content else [],
                    })
        return items

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    # ── Response parsing ────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> FunctionCallResult:
        """Parse Responses API response into FunctionCallResult.

        Response structure:
          {
            "status": "completed" | "incomplete" | "failed",
            "output": [
              {"type": "message", "content": [{"type": "output_text", "text": "..."}]},
              {"type": "function_call", "name": "...", "arguments": "...", "call_id": "..."},
            ]
          }
        """
        output = data.get("output", [])
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason = "stop"

        for item in output:
            item_type = item.get("type", "")
            if item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        text_parts.append(part.get("text", ""))
            elif item_type == "function_call":
                name = item.get("name", "")
                args_str = item.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(ToolCall(
                    id=item.get("call_id", ""),
                    name=name,
                    arguments=args,
                ))
                finish_reason = "tool_calls"

        status = data.get("status", "completed")
        if status == "incomplete":
            finish_reason = "length"
        elif status == "failed":
            finish_reason = "error"

        return FunctionCallResult(
            content="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    # ── Public API (unchanged interface) ─────────────────────────────────────

    def complete(
        self, system: str, messages: list[dict[str, Any]], temperature: float | None = None
    ) -> str:
        url = self.base_url.rstrip("/") + "/responses"
        payload = self._build_payload(system, messages, temperature=temperature)
        data = post_json_with_retry(url, payload, self._headers(), self.timeout_sec)
        result = self._parse_response(data)
        if not result.content:
            raise LLMError("DeepSeek response message content is empty")
        return result.content

    def complete_json(
        self,
        system: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Call DeepSeek with JSON output mode.

        Uses text.format={type: "json_object"} in Responses API.
        The system prompt must contain the word "json" for this to work.
        """
        if schema:
            system += (
                "\nReturn JSON matching this schema. Do not omit required fields, "
                "use null for required arrays, or return empty arrays when minItems "
                "is specified. Schema:\n"
                + json.dumps(schema, ensure_ascii=False)
            )
        url = self.base_url.rstrip("/") + "/responses"
        payload = self._build_payload(
            system, messages, temperature=temperature, json_mode=True,
        )
        data = post_json_with_retry(url, payload, self._headers(), self.timeout_sec)
        result = self._parse_response(data)
        content = result.content
        if not content:
            raise LLMError("DeepSeek JSON response was empty")
        try:
            return parse_json_response(content)
        except Exception as err:
            raise LLMError(f"DeepSeek JSON output could not be parsed: {err}") from err

    def stream_complete(
        self, system: str, messages: list[dict[str, Any]], temperature: float | None = None
    ) -> Iterator[str]:
        """Stream completion tokens from DeepSeek Responses API.

        Uses Responses API streaming (stream=True). Yields content delta chunks
        from response.output_text.delta events.
        """
        url = self.base_url.rstrip("/") + "/responses"
        payload = self._build_payload(system, messages, temperature=temperature)
        payload["stream"] = True

        for event_type, raw in post_json_stream_events(
            url, payload, self._headers(), self.timeout_sec
        ):
            # Only process output text deltas
            if event_type and not event_type.startswith("response.output_text"):
                # Check for terminal events
                if event_type in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                ):
                    return
                continue

            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("stream_complete: skipped unparseable SSE chunk: %s", raw[:100])
                continue

            # Extract delta text
            delta = chunk.get("delta", "")
            if delta:
                yield delta

    def function_call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float | None = None,
    ) -> FunctionCallResult:
        """Call DeepSeek Responses API with native function calling.

        Returns a FunctionCallResult with either content or tool_calls.
        Supports multi-turn tool messages:
        - role="tool" with tool_call_id + content
        - role="assistant" with tool_calls[] + content

        Tool names are sanitized (dots → underscores) for API compatibility.
        """
        url = self.base_url.rstrip("/") + "/responses"

        # Sanitize tool names: replace dots with underscores for APIs that
        # only accept ^[a-zA-Z0-9_-]+$ (strict OpenAI-compatible).
        name_map: dict[str, str] = {}  # sanitized → original
        reverse_map: dict[str, str] = {}  # original → sanitized
        api_tools: list[dict[str, Any]] = []
        for t in tools:
            original = t["name"]
            sanitized = original.replace(".", "_")
            name_map[sanitized] = original
            reverse_map[original] = sanitized
            # Responses API uses flat tool format (not nested under "function")
            api_tools.append({
                "type": "function",
                "name": sanitized,
                "description": t["description"],
                "parameters": t.get("input_schema", {}),
            })

        # Normalize messages: sanitize tool names in assistant tool_calls
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
            system, api_messages, tools=api_tools, tool_choice=tool_choice,
            temperature=temperature,
        )

        data = post_json_with_retry(url, payload, self._headers(), self.timeout_sec)
        result = self._parse_response(data)

        # Map sanitized tool names back to original (ToolCall is frozen, so rebuild)
        if name_map:
            result.tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=name_map.get(tc.name, tc.name),
                    arguments=tc.arguments,
                )
                for tc in result.tool_calls
            ]

        return result
