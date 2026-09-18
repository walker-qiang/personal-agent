"""Unified policy contracts for tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Protocol

from .requests import ExecutionPolicy
from .tools import ToolPolicyClass, ToolSpec


class PolicyDecisionKind(str, Enum):
    """The only outcomes a tool policy evaluation may produce."""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(frozen=True)
class ToolExecutionContext:
    """Request-scoped identity and policy presented to the tool gateway."""

    owner_id: str
    session_id: str
    source: str = "runtime"
    operation_id: str = ""
    orchestration_run_id: str = ""
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    allowed_tool_classes: frozenset[ToolPolicyClass] | None = None
    strict_classification: bool = False

    def __post_init__(self) -> None:
        if not self.owner_id.strip():
            raise ValueError("tool execution owner_id must not be empty")
        if not self.source.strip():
            raise ValueError("tool execution source must not be empty")


@dataclass(frozen=True)
class EffectGrant:
    """Non-forgeable-by-convention approval binding for one tool call.

    The Runtime creates this value only after an approval has been resolved.
    The gateway verifies every binding again immediately before invoking the
    handler, including the canonical argument digest.
    """

    owner_id: str
    operation_id: str
    approval_set_id: str
    approval_id: str
    tool_name: str
    arguments_digest: str
    decided_by: str
    decision_source: str = ""
    decision: str = "approve"

    def __post_init__(self) -> None:
        for field_name in (
            "owner_id",
            "operation_id",
            "approval_set_id",
            "approval_id",
            "tool_name",
            "arguments_digest",
            "decided_by",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"effect grant {field_name} must not be empty")
        if self.decision != "approve":
            raise ValueError("effect grant decision must be approve")

    def matches(
        self,
        *,
        owner_id: str,
        operation_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        return (
            self.owner_id == owner_id
            and self.operation_id == operation_id
            and self.tool_name == tool_name
            and self.arguments_digest == arguments_digest(arguments)
        )


@dataclass(frozen=True)
class PolicyDecision:
    """Decision returned by the pure policy evaluator."""

    kind: PolicyDecisionKind
    tool_class: ToolPolicyClass
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.kind is PolicyDecisionKind.ALLOW


class PolicyEvaluator(Protocol):
    """Port for a pure, request-scoped policy evaluator."""

    def evaluate(
        self,
        tool: ToolSpec,
        context: ToolExecutionContext,
        arguments: dict[str, Any],
        approval_grant: EffectGrant | None = None,
    ) -> PolicyDecision:
        ...


class DefaultPolicyEvaluator:
    """Evaluate the currently supported Runtime policy semantics.

    This intentionally preserves the existing behavior while moving the
    decision into one reusable boundary.  More restrictive rules for code
    execution and delegated agents can be added here without changing tool
    handlers or callers.
    """

    def evaluate(
        self,
        tool: ToolSpec,
        context: ToolExecutionContext,
        arguments: dict[str, Any],
        approval_grant: EffectGrant | None = None,
    ) -> PolicyDecision:
        tool_class = _coerce_policy_class(tool.policy_class)
        policy = context.policy
        if (
            context.strict_classification
            and tool_class is ToolPolicyClass.UNCLASSIFIED
        ):
            return PolicyDecision(
                PolicyDecisionKind.DENY,
                tool_class,
                (
                    f"tool {tool.name} has no declared policy class; "
                    "execution requires explicit classification"
                ),
            )
        if (
            context.allowed_tool_classes is not None
            and tool_class not in {
                _coerce_policy_class(item)
                for item in context.allowed_tool_classes
            }
        ):
            return PolicyDecision(
                PolicyDecisionKind.DENY,
                tool_class,
                (
                    f"tool {tool.name} ({tool_class.value}) is not allowed "
                    f"from source {context.source}"
                ),
            )
        effectful = tool.side_effect or tool_class in {
            ToolPolicyClass.DURABLE_WRITE,
        }

        if effectful and not policy.allow_external_effects:
            return PolicyDecision(
                PolicyDecisionKind.DENY,
                tool_class,
                (
                    f"tool {tool.name} is blocked by agent mode {policy.mode}; "
                    "use an approved writeback operation"
                ),
            )

        approval_required = tool.requires_approval or (
            effectful and policy.require_approval
        )
        if approval_required:
            if approval_grant is None:
                return PolicyDecision(
                    PolicyDecisionKind.REQUIRE_APPROVAL,
                    tool_class,
                    f"tool {tool.name} requires Runtime approval before execution",
                )
            if not approval_grant.matches(
                owner_id=context.owner_id,
                operation_id=context.operation_id,
                tool_name=tool.name,
                arguments=arguments,
            ):
                return PolicyDecision(
                    PolicyDecisionKind.DENY,
                    tool_class,
                    "approval grant does not match the requested tool or arguments",
                )

        return PolicyDecision(PolicyDecisionKind.ALLOW, tool_class)


def arguments_digest(arguments: dict[str, Any]) -> str:
    """Return a stable, non-sensitive digest for one tool argument object."""

    encoded = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def effect_grant_from_approval(
    approval: Any,
    *,
    arguments: dict[str, Any] | None = None,
) -> EffectGrant:
    """Create a gateway grant from a resolved Runtime approval."""

    approval_arguments = dict(
        approval.sanitized_arguments if arguments is None else arguments
    )
    decision = getattr(approval, "decision", None)
    decision_value = getattr(decision, "value", decision)
    return EffectGrant(
        owner_id=approval.owner_id,
        operation_id=approval.operation_id,
        approval_set_id=approval.approval_set_id,
        approval_id=approval.approval_id,
        tool_name=approval.tool_name,
        arguments_digest=arguments_digest(approval_arguments),
        decided_by=approval.decided_by,
        decision_source=approval.decision_source,
        decision=str(decision_value or "approve"),
    )


def _coerce_policy_class(value: ToolPolicyClass | str) -> ToolPolicyClass:
    if isinstance(value, ToolPolicyClass):
        return value
    try:
        return ToolPolicyClass(str(value))
    except ValueError:
        return ToolPolicyClass.UNCLASSIFIED
