"""DeepSeek API client using the documented Chat Completions API.

DeepSeek exposes an OpenAI-compatible endpoint at
``POST https://api.deepseek.com/chat/completions``. The rest of Matrix uses
provider-neutral messages and tool definitions, so this adapter owns the
small amount of format conversion required by DeepSeek.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from .errors import LLMError
from .http import post_json, post_json_stream, post_json_with_retry
from .protocol import FunctionCallResult, LLMStreamEvent, ToolCall, parse_json_response
from .truncate import truncate_messages


logger = logging.getLogger("matrix.llm.deepseek")

_DEFAULT_MAX_MESSAGE_CHARS = 16000
_NO_TOOL_OUTPUT = '{"error": "tool produced no output"}'


def _parse_tool_arguments(value: Any) -> tuple[dict[str, Any], str]:
    """Normalize provider tool arguments without silently erasing failures."""

    if isinstance(value, dict):
        return value, ""
    if value is None or value == "":
        return {}, ""
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError) as exc:
        return {}, f"工具参数不是有效 JSON: {exc}"
    if not isinstance(parsed, dict):
        return {}, "工具参数必须是 JSON 对象"
    return parsed, ""


def _ensure_tool_call_outputs(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep every assistant tool call paired with a tool response.

    Context truncation can remove a tool response while keeping the assistant
    message that requested it. DeepSeek rejects that history, so inject a
    neutral provider-visible result and retain a warning for diagnosis.
    """

    pending: list[tuple[int, list[str]]] = []
    satisfied: set[str] = set()
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            ids = [
                str(tool_call.get("id", ""))
                for tool_call in message["tool_calls"]
                if tool_call.get("id")
            ]
            if ids:
                pending.append((index, ids))
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id", ""))
            if call_id:
                satisfied.add(call_id)

    missing_by_index = {
        index: [call_id for call_id in ids if call_id not in satisfied]
        for index, ids in pending
        if any(call_id not in satisfied for call_id in ids)
    }
    if not missing_by_index:
        return messages

    total = sum(len(ids) for ids in missing_by_index.values())
    logger.warning(
        "llm.deepseek: %d tool call(s) missing output; injecting placeholders "
        "assistant_indexes=%s",
        total,
        sorted(missing_by_index),
    )
    result = list(messages)
    for index in sorted(missing_by_index, reverse=True):
        placeholders = [
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": _NO_TOOL_OUTPUT,
            }
            for call_id in missing_by_index[index]
        ]
        result[index + 1:index + 1] = placeholders
    return result


def _sanitize_tool_name(name: Any) -> str:
    """DeepSeek function names allow alphanumerics, ``_`` and ``-`` only."""

    return str(name).replace(".", "_")


class DeepSeekClient:
    """LLM client for DeepSeek's OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-flash",
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

    def _chat_url(self) -> str:
        """Build the documented endpoint while allowing a custom base URL."""

        return self.base_url.rstrip("/") + "/chat/completions"

    def _messages(
        self,
        system: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.max_message_chars > 0:
            messages = truncate_messages(
                messages,
                system_prompt=system,
                max_tokens=self.max_message_chars // 2,
                reserve_tokens=500,
            )
        messages = _ensure_tool_call_outputs(messages)
        normalized: list[dict[str, Any]] = [
            {"role": "system", "content": system},
        ]
        for message in messages:
            copied = dict(message)
            if copied.get("role") == "assistant" and copied.get("tool_calls"):
                tool_calls = []
                for raw_tool_call in copied["tool_calls"]:
                    tool_call = dict(raw_tool_call)
                    function = dict(tool_call.get("function", {}))
                    if "name" in function:
                        function["name"] = _sanitize_tool_name(function["name"])
                    tool_call["function"] = function
                    tool_calls.append(tool_call)
                copied["tool_calls"] = tool_calls
            normalized.append(copied)
        return normalized

    def _tool_definitions(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        for tool in tools:
            function: dict[str, Any] = {
                "name": _sanitize_tool_name(tool["name"]),
                "description": tool.get("description", ""),
            }
            if "input_schema" in tool:
                function["parameters"] = tool["input_schema"]
            definitions.append({"type": "function", "function": function})
        return definitions

    def _build_payload(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Build a DeepSeek Chat Completions request."""

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "messages": self._messages(system, messages),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = self._tool_definitions(tools)
            payload["tool_choice"] = tool_choice
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> FunctionCallResult:
        """Parse a documented Chat Completions response."""

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError("DeepSeek response did not include choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMError("DeepSeek response choice is invalid")
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            raise LLMError("DeepSeek response message is invalid")

        content = message.get("content")
        text = "" if content is None else str(content)
        tool_calls: list[ToolCall] = []
        for raw_tool_call in message.get("tool_calls", []) or []:
            if not isinstance(raw_tool_call, dict):
                continue
            function = raw_tool_call.get("function") or {}
            if not isinstance(function, dict):
                function = {}
            arguments, arguments_error = _parse_tool_arguments(
                function.get("arguments", "{}"),
            )
            tool_calls.append(ToolCall(
                id=str(raw_tool_call.get("id", "")),
                name=str(function.get("name", "")),
                arguments=arguments,
                arguments_error=arguments_error,
            ))

        finish_reason = str(choice.get("finish_reason") or "stop")
        return FunctionCallResult(
            content=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _usage(data: dict[str, Any]) -> dict[str, Any]:
        """Map provider usage to Matrix's provider-neutral budget fields."""

        usage = data.get("usage")
        if not isinstance(usage, dict):
            return {}
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        normalized: dict[str, Any] = {}
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
            normalized["input_tokens"] = prompt_tokens
        if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool):
            normalized["output_tokens"] = completion_tokens
        if isinstance(usage.get("total_tokens"), int):
            normalized["total_tokens"] = usage["total_tokens"]
        if isinstance(usage.get("completion_tokens_details"), dict):
            normalized["completion_tokens_details"] = usage["completion_tokens_details"]
        return normalized

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> str:
        payload = self._build_payload(system, messages, temperature=temperature)
        data = post_json_with_retry(
            self._chat_url(), payload, self._headers(), self.timeout_sec,
        )
        result = self._parse_response(data)
        if not result.content:
            raise LLMError(
                "DeepSeek response message content is empty "
                f"(finish_reason={result.finish_reason})"
            )
        return result.content

    def complete_json(
        self,
        system: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Call DeepSeek with the documented JSON response mode."""

        if schema:
            system += (
                "\nReturn JSON matching this schema. Do not omit required fields, "
                "use null for required arrays, or return empty arrays when minItems "
                "is specified. Schema:\n"
                + json.dumps(schema, ensure_ascii=False)
            )
        payload = self._build_payload(
            system, messages, temperature=temperature, json_mode=True,
        )
        data = post_json_with_retry(
            self._chat_url(), payload, self._headers(), self.timeout_sec,
        )
        content = self._parse_response(data).content
        if not content:
            raise LLMError("DeepSeek JSON response was empty")
        try:
            return parse_json_response(content)
        except Exception as err:
            raise LLMError(f"DeepSeek JSON output could not be parsed: {err}") from err

    def complete_json_budgeted(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Make exactly one bounded request and return normalized usage."""

        payload = self._build_payload(
            system,
            messages,
            json_mode=True,
            max_tokens=max_output_tokens,
            # Low is a documented DeepSeek level and leaves room for the
            # structured answer under the stock research output ceiling.
            reasoning_effort="low",
        )
        data = post_json(self._chat_url(), payload, self._headers(), self.timeout_sec)
        result = self._parse_response(data)
        if result.finish_reason not in {"stop", "tool_calls"}:
            raise LLMError(
                "budgeted stock research did not complete "
                f"(finish_reason={result.finish_reason})"
            )
        try:
            parsed = parse_json_response(result.content)
        except Exception as err:
            raise LLMError(f"stock research JSON output could not be parsed: {err}") from err
        if not isinstance(parsed, dict):
            raise LLMError("stock research response must be a JSON object")
        return parsed, self._usage(data)

    def stream_complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Stream assistant text from Chat Completions SSE chunks."""

        payload = self._build_payload(system, messages, temperature=temperature)
        payload["stream"] = True
        for raw in post_json_stream(
            self._chat_url(), payload, self._headers(), self.timeout_sec,
        ):
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("stream_complete: skipped invalid SSE chunk: %s", raw[:100])
                continue
            choices = chunk.get("choices", [])
            if not choices or not isinstance(choices[0], dict):
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                yield str(content)

    def stream_function_call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float | None = None,
    ) -> Iterator[LLMStreamEvent]:
        """Stream text and tool-call deltas from Chat Completions SSE."""

        payload = self._build_payload(
            system,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        names: dict[str, str] = {}
        calls: dict[int, dict[str, str]] = {}
        finish_reason = "stop"
        usage: dict[str, Any] = {}

        for raw in post_json_stream(
            self._chat_url(), payload, self._headers(), self.timeout_sec,
        ):
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "stream_function_call: skipped invalid SSE chunk: %s",
                    raw[:100],
                )
                continue
            usage = self._usage(chunk) or usage
            choices = chunk.get("choices", [])
            if not choices or not isinstance(choices[0], dict):
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                yield LLMStreamEvent(kind="message_delta", content=str(content))
            for raw_tool_call in delta.get("tool_calls", []) or []:
                if not isinstance(raw_tool_call, dict):
                    continue
                index = int(raw_tool_call.get("index", 0))
                state = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if raw_tool_call.get("id"):
                    state["id"] = str(raw_tool_call["id"])
                function = raw_tool_call.get("function") or {}
                if not isinstance(function, dict):
                    function = {}
                if function.get("name"):
                    state["name"] = str(function["name"])
                    names[state["name"]] = next(
                        (
                            str(tool["name"])
                            for tool in tools
                            if _sanitize_tool_name(tool["name"]) == state["name"]
                        ),
                        state["name"],
                    )
                arguments_delta = str(function.get("arguments", ""))
                if arguments_delta:
                    state["arguments"] += arguments_delta
                yield LLMStreamEvent(
                    kind="tool_call_delta",
                    metadata={
                        "call_id": state["id"],
                        "index": index,
                        "name": names.get(state["name"], state["name"]),
                        "arguments_delta": arguments_delta,
                    },
                )

        for index in sorted(calls):
            state = calls[index]
            arguments, arguments_error = _parse_tool_arguments(state["arguments"])
            yield LLMStreamEvent(
                kind="tool_calls",
                tool_calls=(
                    ToolCall(
                        id=state["id"],
                        name=names.get(state["name"], state["name"]),
                        arguments=arguments,
                        arguments_error=arguments_error,
                    ),
                ),
            )
        yield LLMStreamEvent(
            kind="message_end",
            metadata={"finish_reason": finish_reason, "usage": usage},
        )

    def function_call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float | None = None,
    ) -> FunctionCallResult:
        """Call DeepSeek with native Chat Completions function calling."""

        payload = self._build_payload(
            system,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )
        data = post_json_with_retry(
            self._chat_url(), payload, self._headers(), self.timeout_sec,
        )
        result = self._parse_response(data)
        names = {
            _sanitize_tool_name(tool["name"]): str(tool["name"])
            for tool in tools
        }
        result.tool_calls = [
            ToolCall(
                id=tool_call.id,
                name=names.get(tool_call.name, tool_call.name),
                arguments=tool_call.arguments,
                arguments_error=tool_call.arguments_error,
            )
            for tool_call in result.tool_calls
        ]
        return result
