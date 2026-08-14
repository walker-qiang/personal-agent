"""Adapter for the existing context budget helpers."""

from __future__ import annotations

from ...context.budget import check_budget
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
        # The full LLM-assisted compaction policy remains an application
        # concern.  Keep the newest messages as a deterministic safe fallback.
        return list(messages[-6:])
