"""Pure compilation of tool requests into policy-evaluated execution plans."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Protocol

from ..domain.plans import ExecutionPlan, PlanStatus, PlanStep
from ..domain.policy import (
    DefaultPolicyEvaluator,
    PolicyDecisionKind,
    PolicyEvaluator,
    ToolExecutionContext,
)
from ..domain.tools import ToolRequest, ToolSpec


class ToolSpecResolver(Protocol):
    """Resolve application tool metadata without invoking a handler."""

    def __call__(self, name: str) -> ToolSpec | None:
        ...


class PlanCompiler:
    """Compile one or more tool requests before execution.

    Compilation is intentionally side-effect free.  It resolves tool metadata,
    evaluates the request-scoped policy, and returns one atomic plan.  A plan
    containing a denied or approval-required step is never partially executed.
    """

    def __init__(
        self,
        resolve_tool: ToolSpecResolver | Callable[[str], ToolSpec | None],
        evaluator: PolicyEvaluator | None = None,
    ) -> None:
        self._resolve_tool = resolve_tool
        self._evaluator = evaluator or DefaultPolicyEvaluator()

    def compile(
        self,
        requests: ToolRequest | Iterable[ToolRequest],
        context: ToolExecutionContext,
    ) -> ExecutionPlan:
        request_items = (
            (requests,) if isinstance(requests, ToolRequest) else tuple(requests)
        )
        operation_id = context.operation_id or (
            request_items[0].operation_id if request_items else ""
        )
        if not context.operation_id and operation_id:
            context = replace(context, operation_id=operation_id)
        if not request_items:
            return ExecutionPlan(
                operation_id=operation_id,
                source=context.source,
                context=context,
                steps=(),
                status=PlanStatus.DENY,
                reason="execution plan must contain at least one tool request",
            )

        steps: list[PlanStep] = []
        status = PlanStatus.READY
        reason = ""
        for index, request in enumerate(request_items, start=1):
            if operation_id and request.operation_id and request.operation_id != operation_id:
                return ExecutionPlan(
                    operation_id=operation_id,
                    source=context.source,
                    context=context,
                    steps=tuple(steps),
                    status=PlanStatus.DENY,
                    reason="all plan steps must belong to the same operation",
                )

            tool = self._resolve_tool(request.name)
            if tool is None:
                return ExecutionPlan(
                    operation_id=operation_id,
                    source=context.source,
                    context=context,
                    steps=tuple(steps),
                    status=PlanStatus.DENY,
                    reason=f"tool {request.name} is not registered",
                )

            decision = self._evaluator.evaluate(
                tool,
                context,
                request.arguments,
                request.approval_grant,
            )
            steps.append(PlanStep(
                step_id=request.call_id or f"step-{index}",
                request=request,
                tool=tool,
                decision=decision,
            ))
            if decision.kind is PolicyDecisionKind.DENY:
                status = PlanStatus.DENY
                reason = decision.reason
                break
            if (
                decision.kind is PolicyDecisionKind.REQUIRE_APPROVAL
                and status is PlanStatus.READY
            ):
                status = PlanStatus.REQUIRE_APPROVAL
                reason = decision.reason

        return ExecutionPlan(
            operation_id=operation_id,
            source=context.source,
            context=context,
            steps=tuple(steps),
            status=status,
            reason=reason,
        )
