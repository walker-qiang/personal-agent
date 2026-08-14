"""Identifier generation port."""

from __future__ import annotations

from typing import Protocol


class IdPort(Protocol):
    def new_id(self, prefix: str = "") -> str:
        ...
