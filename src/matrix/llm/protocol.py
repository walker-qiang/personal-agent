"""LLM client protocol and shared types."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol


class LLMClient(Protocol):
    """Protocol for LLM provider clients.

    Messages use the OpenAI-compatible format. The ``content`` field
    can be either a plain string or a list of content blocks for
    multi-modal input (images, etc.).
    """

    def complete(self, system: str, messages: list[dict[str, Any]], temperature: float | None = None) -> str:
        ...

    def complete_json(
        self,
        system: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Call LLM and return a parsed JSON object/array.

        Uses provider-native structured output when available:
        - DeepSeek/OpenAI: response_format={"type": "json_object"}
        - Anthropic: forced single-tool call with JSON schema

        Falls back to prompt-based JSON with robust parsing.

        Args:
            system: System prompt (should instruct JSON output).
            messages: Conversation messages.
            schema: Optional JSON Schema dict for the expected output.
                When provided, providers that support schema enforcement
                will use it. When None, only json_object mode is used.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON as dict or list.

        Raises:
            LLMError: If the LLM call fails or output cannot be parsed as JSON.
        """
        ...

    def stream_complete(self, system: str, messages: list[dict[str, Any]], temperature: float | None = None) -> Iterator[str]:
        """Stream completion tokens one by one. Yields content chunks."""
        ...

    def function_call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float | None = None,
    ) -> FunctionCallResult:
        """Call LLM with native function/tool calling support.

        Supports multi-turn tool messages:
        - role="tool" with tool_call_id + content
        - role="assistant" with tool_calls[] + content
        """
        ...


@dataclass(frozen=True)
class ToolCall:
    """A parsed tool call from the LLM response."""

    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionCallResult:
    """Result of a function calling LLM invocation."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # "stop" | "tool_calls" | "length"


# ---- JSON response parsing utilities ----

_MD_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_json_response(text: str) -> dict[str, Any] | list[Any]:
    """Parse a JSON object or array from LLM text output.

    Handles:
    - Pure JSON
    - Markdown fenced JSON (```json ... ```)
    - JSON embedded in prose (extracts first {...} or [...])

    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted.
    """
    cleaned = text.strip()

    # 1. Try markdown fence extraction
    fence_match = _MD_FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # 2. Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Try extracting JSON object or array via balanced bracket matching
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:i + 1])
                    except json.JSONDecodeError:
                        break

    raise json.JSONDecodeError("No valid JSON found", text, 0)