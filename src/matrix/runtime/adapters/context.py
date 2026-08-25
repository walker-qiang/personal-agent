"""Adapter for the existing context budget helpers."""

from __future__ import annotations

from ...context.budget import check_budget
from ...tools.truncate import truncate_string
from ..domain.messages import Message
from ..ports.context import ContextPort


class MatrixContextAdapter(ContextPort):
    """Apply the existing budget check while keeping compaction above Core."""

    def __init__(self, context_window: int = 128000) -> None:
        self.context_window = context_window

    def fit(self, system_prompt: str, messages: list[Message]) -> list[Message]:
        raw_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]
        ok, _, _ = check_budget(
            raw_messages,
            system_prompt,
            context_window=self.context_window,
        )
        if ok:
            return list(messages)
        # Keep a recent, structurally valid suffix. A tool message without its
        # preceding assistant tool call is invalid for several providers.
        kept = list(messages[-6:])
        while kept and kept[0].role == "tool":
            kept.pop(0)
        if not kept:
            kept = list(messages[-1:])

        # A single oversized message can still exceed the budget. Truncate
        # only its textual content as a deterministic final fallback.
        fitted: list[Message] = []
        for message in kept:
            content = message.content
            if isinstance(content, str) and len(content) > 20000:
                content = truncate_string(content, max_chars=20000)
            fitted.append(
                Message(
                    role=message.role,
                    content=content,
                    tool_calls=message.tool_calls,
                    tool_call_id=message.tool_call_id,
                )
            )
        return fitted
