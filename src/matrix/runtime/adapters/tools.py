"""Adapter from the existing guarded ToolRegistry to Runtime tools."""

from __future__ import annotations

from typing import Any

from ...tools.registry import ToolRegistry
from ..domain.tools import ToolRequest, ToolResult
from ..ports.tools import ToolExecutorPort


class MatrixToolAdapter(ToolExecutorPort):
    """Preserve the existing validation/guard/truncation pipeline."""

    def __init__(self, registry: ToolRegistry, session_id: str = "") -> None:
        self.registry = registry
        self.session_id = session_id

    def execute(self, request: ToolRequest) -> ToolResult:
        result = self.registry.call(
            request.name,
            request.arguments,
            session_id=self.session_id or request.operation_id,
        )
        if isinstance(result, dict) and "error" in result:
            return ToolResult(
                call_id=request.call_id,
                name=request.name,
                error=str(result["error"]),
                is_error=True,
            )
        return ToolResult(call_id=request.call_id, name=request.name, result=result)


def tool_specs(registry: ToolRegistry) -> list:
    """Convert registered tools to Runtime-owned specs without importing Core."""

    from ..domain.tools import RecoveryPolicy, ToolSpec

    return [
        ToolSpec(
            name=definition.name,
            description=definition.description,
            input_schema=definition.input_schema,
            recovery_policy=RecoveryPolicy(definition.recovery_policy),
            requires_approval=definition.requires_approval,
        )
        for definition in (registry.get_definition(name) for name in sorted(registry.tool_names()))
        if definition is not None
    ]
