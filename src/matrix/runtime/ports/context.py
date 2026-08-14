"""Context budget and compaction port.

The Runtime Core only knows this small contract.  The existing Matrix context
implementation will be adapted in a later work package.
"""

from __future__ import annotations

from typing import Protocol

from ..domain.messages import Message


class ContextPort(Protocol):
    def fit(self, system_prompt: str, messages: list[Message]) -> list[Message]:
        """Return messages that fit the active execution budget."""

        ...
