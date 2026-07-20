"""Tool registry: declarative registration, discovery, and invocation."""

from __future__ import annotations

from typing import Any

from .base import FinanceToolError, ToolDefinition


class ToolRegistry:
    """Registry for tool definitions with validation and invocation."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._guard: object | None = None  # ToolGuard or None

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def list_tools(self) -> list[dict[str, Any]]:
        """Return all tool definitions in LLM-compatible format."""
        return [tool.to_dict() for tool in self._tools.values()]

    def tool_names(self) -> set[str]:
        """Return the set of registered tool names."""
        return set(self._tools.keys())

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate and invoke a tool by name."""
        args = arguments or {}
        if not isinstance(args, dict):
            raise FinanceToolError("arguments must be an object")
        if name not in self._tools:
            raise FinanceToolError(f"unknown tool: {name}")
        # ---- TOOL GUARD ----
        if self._guard:
            from ..guardrails.tool_guard import ToolGuardError
            ok, reason = self._guard.check(name, args)
            if not ok:
                raise ToolGuardError(f"tool blocked: {reason}")
        # ---- END TOOL GUARD ----
        tool = self._tools[name]
        return tool.handler(**args)

    def set_guard(self, guard: object) -> None:
        """Attach a ToolGuard instance for safety checks."""
        self._guard = guard

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name, or None."""
        return self._tools.get(name)