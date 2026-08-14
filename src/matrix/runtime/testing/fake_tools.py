"""Scriptable tool executor fake; it never performs external effects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..domain.tools import ToolRequest, ToolResult
from ..ports.tools import ToolExecutorPort


class FakeToolExecutor(ToolExecutorPort):
    def __init__(
        self,
        handlers: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
    ) -> None:
        self.handlers = handlers or {}
        self.requests: list[ToolRequest] = []

    def execute(self, request: ToolRequest) -> ToolResult:
        self.requests.append(request)
        handler = self.handlers.get(request.name)
        if handler is None:
            return ToolResult(
                call_id=request.call_id,
                name=request.name,
                error=f"fake tool not registered: {request.name}",
                is_error=True,
            )
        try:
            return ToolResult(
                call_id=request.call_id,
                name=request.name,
                result=handler(request.arguments),
            )
        except Exception as exc:  # pragma: no cover - exercised by callers' scenarios
            return ToolResult(
                call_id=request.call_id,
                name=request.name,
                error=str(exc),
                is_error=True,
            )
