"""Single policy-aware boundary for invoking registered tools."""

from __future__ import annotations

from typing import Any

from .registry import ToolRegistry


class ToolExecutionGateway:
    """Apply Runtime policy before delegating to the existing tool pipeline."""

    def __init__(self, registry: ToolRegistry, evaluator: Any = None) -> None:
        self.registry = registry
        if evaluator is None:
            from ..runtime.domain.policy import DefaultPolicyEvaluator

            evaluator = DefaultPolicyEvaluator()
        self.evaluator = evaluator

    def call(self, request: Any, context: Any) -> dict[str, Any]:
        definition = self.registry.get_definition(request.name)
        if definition is None:
            return {
                "error": (
                    f"工具 {request.name} 不存在。可用工具: "
                    f"{', '.join(sorted(self.registry.tool_names())[:10])}"
                )
            }

        from ..runtime.adapters.tools import tool_spec_from_definition
        from ..runtime.domain.policy import (
            PolicyDecisionKind,
            ToolExecutionContext,
        )

        if context is None:
            context = ToolExecutionContext(
                owner_id="default",
                session_id="",
                source="gateway",
            )
        tool_spec = tool_spec_from_definition(definition)
        decision = self.evaluator.evaluate(
            tool_spec,
            context,
            request.arguments,
            getattr(request, "approval_grant", None),
        )
        if decision.kind is not PolicyDecisionKind.ALLOW:
            return {
                "error": decision.reason
                or f"tool {request.name} was blocked by execution policy"
            }

        return self.registry.call(
            request.name,
            request.arguments,
            session_id=context.session_id or request.operation_id,
        )
