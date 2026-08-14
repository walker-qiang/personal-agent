"""Adapter from the existing guarded ToolRegistry to Runtime tools."""

from __future__ import annotations

from typing import Any

from ...tools.principal import tool_principal
from ...tools.registry import ToolRegistry
from ..domain.tools import ToolRequest, ToolResult
from ..ports.tools import ToolExecutorPort


class MatrixToolAdapter(ToolExecutorPort):
    """Preserve the existing validation/guard/truncation pipeline."""

    def __init__(
        self,
        registry: ToolRegistry,
        session_id: str = "",
        owner_id: str = "default",
        mode: str = "read_only",
        allow_external_effects: bool = False,
    ) -> None:
        self.registry = registry
        self.session_id = session_id
        self.owner_id = owner_id
        self.mode = mode
        self.allow_external_effects = allow_external_effects

    def execute(self, request: ToolRequest) -> ToolResult:
        with tool_principal(
            self.owner_id, self.session_id, self.mode, self.allow_external_effects,
        ):
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
            side_effect=definition.side_effect,
        )
        for definition in (registry.get_definition(name) for name in sorted(registry.tool_names()))
        if definition is not None
    ]
