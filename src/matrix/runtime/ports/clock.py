"""Clock port for deterministic transitions and tests."""

from __future__ import annotations

from typing import Protocol


class ClockPort(Protocol):
    def now(self) -> float:
        ...
