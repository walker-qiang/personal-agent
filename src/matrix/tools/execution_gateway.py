"""Single policy-aware boundary for invoking registered tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..runtime.core.plan_compiler import PlanCompiler
from ..runtime.domain.plans import ExecutionPlan, PlanStatus
from ..runtime.domain.policy import PolicyDecisionKind, ToolExecutionContext
from ..runtime.domain.tools import ToolSpec
from .registry import ToolRegistry


ToolSpecResolver = Callable[[str], ToolSpec | None]


class ToolExecutionGateway:
    """Apply Runtime policy before delegating to the existing tool pipeline."""

    def __init__(
        self,
        registry: ToolRegistry,
        evaluator: Any = None,
        spec_resolver: ToolSpecResolver | None = None,
    ) -> None:
        self.registry = registry
        if evaluator is None:
            from ..runtime.domain.policy import DefaultPolicyEvaluator

            evaluator = DefaultPolicyEvaluator()
        self.evaluator = evaluator
        self._spec_resolver = spec_resolver
        self.compiler = PlanCompiler(self._resolve_tool_spec, evaluator)

    def _resolve_tool_spec(self, name: str) -> ToolSpec | None:
        if self._spec_resolver is not None:
            return self._spec_resolver(name)

        from ..runtime.adapters.tools import tool_spec_from_definition

        get_definition = getattr(self.registry, "get_definition", None)
        if not callable(get_definition):
            return None
        definition = get_definition(name)
        return tool_spec_from_definition(definition) if definition is not None else None

    def compile(self, request: Any, context: Any) -> ExecutionPlan:
        if context is None:
            context = ToolExecutionContext(
                owner_id="default",
                session_id="",
                source="gateway",
                operation_id=str(getattr(request, "operation_id", "")),
            )
        return self.compiler.compile(request, context)

    def execute(self, plan: ExecutionPlan) -> dict[str, Any]:
        """Re-check and execute a ready plan without partial execution."""

        if not plan.steps:
            return {"error": plan.reason or "execution plan is empty"}
        if plan.status is not PlanStatus.READY:
            return {
                "error": plan.reason
                or f"execution plan is {plan.status.value}",
            }

        results: list[dict[str, Any]] = []
        for step in plan.steps:
            decision = self.evaluator.evaluate(
                step.tool,
                plan.context,
                step.request.arguments,
                step.request.approval_grant,
            )
            if decision.kind is not PolicyDecisionKind.ALLOW:
                return {
                    "error": decision.reason
                    or f"tool {step.request.name} was blocked by execution policy",
                }
            result = self.registry.call(
                step.request.name,
                step.request.arguments,
                session_id=(
                    plan.context.session_id or step.request.operation_id
                ),
            )
            results.append(result)
            if isinstance(result, dict) and "error" in result:
                return result

        return results[0] if len(results) == 1 else {"results": results}

    def call(self, request: Any, context: Any) -> dict[str, Any]:
        """Compatibility wrapper: compile first, then execute the plan."""

        plan = self.compile(request, context)
        return self.execute(plan)
