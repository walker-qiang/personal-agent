from __future__ import annotations

import pytest

from matrix.runtime.adapters.tools import MatrixToolAdapter
from matrix.runtime.domain.approvals import (
    Approval,
    ApprovalDecision,
    ApprovalStatus,
)
from matrix.runtime.domain.policy import (
    DefaultPolicyEvaluator,
    EffectGrant,
    PolicyDecisionKind,
    ToolExecutionContext,
    arguments_digest,
    effect_grant_from_approval,
)
from matrix.runtime.domain.requests import ExecutionPolicy
from matrix.runtime.domain.tools import ToolPolicyClass, ToolRequest, ToolSpec
from matrix.tools import ToolDefinition, ToolRegistry
from matrix.tools.execution_gateway import ToolExecutionGateway


def _context(
    *,
    policy: ExecutionPolicy | None = None,
    owner_id: str = "owner-a",
    operation_id: str = "operation-1",
    strict_classification: bool = False,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        owner_id=owner_id,
        session_id="session-1",
        operation_id=operation_id,
        policy=policy or ExecutionPolicy(),
        strict_classification=strict_classification,
    )


def _approval_grant(
    *,
    owner_id: str = "owner-a",
    operation_id: str = "operation-1",
    tool_name: str = "write",
    arguments: dict | None = None,
) -> EffectGrant:
    call_arguments = arguments or {"value": 1}
    return EffectGrant(
        owner_id=owner_id,
        operation_id=operation_id,
        approval_set_id="approval-set-1",
        approval_id="approval-1",
        tool_name=tool_name,
        arguments_digest=arguments_digest(call_arguments),
        decided_by="owner-a",
    )


def test_policy_allows_read_only_tool() -> None:
    decision = DefaultPolicyEvaluator().evaluate(
        ToolSpec(name="lookup", policy_class=ToolPolicyClass.READ_ONLY),
        _context(),
        {},
    )

    assert decision.kind is PolicyDecisionKind.ALLOW
    assert decision.tool_class is ToolPolicyClass.READ_ONLY


def test_policy_denies_durable_write_in_read_only_mode() -> None:
    decision = DefaultPolicyEvaluator().evaluate(
        ToolSpec(name="write", policy_class=ToolPolicyClass.DURABLE_WRITE),
        _context(),
        {"value": 1},
    )

    assert decision.kind is PolicyDecisionKind.DENY
    assert "read_only" in decision.reason


def test_policy_requires_approval_for_durable_write_in_writeback_mode() -> None:
    decision = DefaultPolicyEvaluator().evaluate(
        ToolSpec(
            name="write",
            requires_approval=True,
            policy_class=ToolPolicyClass.DURABLE_WRITE,
        ),
        _context(
            policy=ExecutionPolicy(
                mode="writeback",
                allow_external_effects=True,
            ),
        ),
        {"value": 1},
    )

    assert decision.kind is PolicyDecisionKind.REQUIRE_APPROVAL


def test_policy_does_not_interrupt_for_ordinary_durable_write() -> None:
    decision = DefaultPolicyEvaluator().evaluate(
        ToolSpec(name="write", policy_class=ToolPolicyClass.DURABLE_WRITE),
        _context(
            policy=ExecutionPolicy(
                mode="writeback",
                allow_external_effects=True,
            ),
        ),
        {"value": 1},
    )

    assert decision.kind is PolicyDecisionKind.ALLOW


@pytest.mark.parametrize(
    ("owner_id", "operation_id", "tool_name", "arguments"),
    [
        ("owner-b", "operation-1", "write", {"value": 1}),
        ("owner-a", "operation-2", "write", {"value": 1}),
        ("owner-a", "operation-1", "other", {"value": 1}),
        ("owner-a", "operation-1", "write", {"value": 2}),
    ],
)
def test_policy_rejects_non_matching_approval_grant(
    owner_id: str,
    operation_id: str,
    tool_name: str,
    arguments: dict,
) -> None:
    decision = DefaultPolicyEvaluator().evaluate(
        ToolSpec(
            name=tool_name,
            requires_approval=True,
            policy_class=ToolPolicyClass.DURABLE_WRITE,
        ),
        _context(
            policy=ExecutionPolicy(
                mode="writeback",
                allow_external_effects=True,
            ),
            owner_id=owner_id,
            operation_id=operation_id,
        ),
        arguments,
        _approval_grant(),
    )

    assert decision.kind is PolicyDecisionKind.DENY
    assert "does not match" in decision.reason


def test_arguments_digest_is_canonical() -> None:
    assert arguments_digest({"a": 1, "b": 2}) == arguments_digest(
        {"b": 2, "a": 1},
    )


def test_effect_grant_can_be_built_from_resolved_approval() -> None:
    approval = Approval(
        approval_id="approval-1",
        approval_set_id="approval-set-1",
        owner_id="owner-a",
        operation_id="operation-1",
        tool_call_id="call-1",
        tool_name="write",
        sanitized_arguments={"value": 1},
        status=ApprovalStatus.APPROVED,
        decision=ApprovalDecision.APPROVE,
        decided_by="owner-a",
    )

    grant = effect_grant_from_approval(approval)

    assert grant.matches(
        owner_id="owner-a",
        operation_id="operation-1",
        tool_name="write",
        arguments={"value": 1},
    )


def test_strict_policy_denies_unclassified_tool() -> None:
    decision = DefaultPolicyEvaluator().evaluate(
        ToolSpec(name="custom.lookup", policy_class=ToolPolicyClass.UNCLASSIFIED),
        _context(strict_classification=True),
        {},
    )

    assert decision.kind is PolicyDecisionKind.DENY
    assert "explicit classification" in decision.reason


def test_compatibility_policy_allows_unclassified_read_tool() -> None:
    decision = DefaultPolicyEvaluator().evaluate(
        ToolSpec(name="custom.lookup", policy_class=ToolPolicyClass.UNCLASSIFIED),
        _context(),
        {},
    )

    assert decision.kind is PolicyDecisionKind.ALLOW


def test_gateway_does_not_invoke_handler_when_policy_denies() -> None:
    calls: list[dict] = []
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="write",
        description="write",
        input_schema={"type": "object", "properties": {}},
        handler=lambda **kwargs: calls.append(kwargs) or {"ok": True},
        policy_class=ToolPolicyClass.DURABLE_WRITE.value,
    ))

    result = ToolExecutionGateway(registry).call(
        ToolRequest(
            operation_id="operation-1",
            call_id="call-1",
            name="write",
            arguments={"value": 1},
        ),
        _context(),
    )

    assert "error" in result
    assert calls == []


def test_compatibility_gateway_can_execute_unclassified_tool() -> None:
    calls: list[dict] = []
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="custom.lookup",
        description="lookup",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
        },
        handler=lambda **kwargs: calls.append(kwargs) or {"ok": True},
    ))

    result = ToolExecutionGateway(registry).call(
        ToolRequest(
            operation_id="operation-1",
            call_id="call-1",
            name="custom.lookup",
            arguments={"value": 1},
        ),
        _context(),
    )

    assert result == {"ok": True}
    assert calls == [{"value": 1}]


def test_gateway_compiles_before_execution_and_supports_plan_status() -> None:
    calls: list[dict] = []
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="lookup",
        description="lookup",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
        },
        handler=lambda **kwargs: calls.append(kwargs) or {"ok": True},
    ))
    gateway = ToolExecutionGateway(registry)
    plan = gateway.compile(
        ToolRequest(
            operation_id="operation-1",
            call_id="call-1",
            name="lookup",
            arguments={"value": 1},
        ),
        _context(),
    )

    assert plan.ready
    assert plan.steps[0].decision.kind is PolicyDecisionKind.ALLOW
    assert calls == []
    assert gateway.execute(plan) == {"ok": True}
    assert calls == [{"value": 1}]


def test_strict_matrix_tool_adapter_rejects_unclassified_tool() -> None:
    calls: list[dict] = []
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="custom.lookup",
        description="lookup",
        input_schema={"type": "object", "properties": {}},
        handler=lambda **kwargs: calls.append(kwargs) or {"ok": True},
    ))
    adapter = MatrixToolAdapter(
        registry,
        session_id="session-1",
        owner_id="owner-a",
        strict_classification=True,
    )

    result = adapter.execute(ToolRequest(
        operation_id="operation-1",
        call_id="call-1",
        name="custom.lookup",
        arguments={},
    ))

    assert result.is_error is True
    assert "explicit classification" in result.error
    assert calls == []
