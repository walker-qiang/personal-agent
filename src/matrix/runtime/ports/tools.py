"""Tool executor port; Runtime does not know ToolRegistry."""

from __future__ import annotations

from typing import Protocol

from ..domain.tools import ToolRequest, ToolResult


class ToolExecutorPort(Protocol):
    def execute(self, request: ToolRequest) -> ToolResult:
        ...
