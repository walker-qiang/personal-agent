"""Adapter from the existing guarded ToolRegistry to Runtime tools."""

from __future__ import annotations

from typing import Any

from ...runtime.domain.policy import (
    DefaultPolicyEvaluator,
    ToolExecutionContext,
)
from ...runtime.domain.requests import ExecutionPolicy
from ...runtime.domain.tools import RecoveryPolicy, ToolPolicyClass, ToolSpec
from ...tools.execution_gateway import ToolExecutionGateway
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
        policy: ExecutionPolicy | None = None,
        source: str = "runtime",
        strict_classification: bool = False,
    ) -> None:
        self.registry = registry
        self.session_id = session_id
        self.owner_id = owner_id
        self.policy = policy or ExecutionPolicy(
            mode=mode,
            allow_external_effects=allow_external_effects,
        )
        self.gateway = ToolExecutionGateway(registry, DefaultPolicyEvaluator())
        self.context = ToolExecutionContext(
            owner_id=owner_id,
            session_id=session_id,
            source=source,
            policy=self.policy,
            strict_classification=strict_classification,
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        with tool_principal(
            self.owner_id,
            self.session_id,
            self.policy.mode,
            self.policy.allow_external_effects,
        ):
            context = self.context
            if request.operation_id and request.operation_id != context.operation_id:
                context = ToolExecutionContext(
                    owner_id=context.owner_id,
                    session_id=context.session_id,
                    source=context.source,
                    operation_id=request.operation_id,
                    orchestration_run_id=context.orchestration_run_id,
                    policy=context.policy,
                    allowed_tool_classes=context.allowed_tool_classes,
                    strict_classification=context.strict_classification,
                )
            plan = self.gateway.compile(request, context)
            result = self.gateway.execute(plan)
        if isinstance(result, dict) and "error" in result:
            return ToolResult(
                call_id=request.call_id,
                name=request.name,
                error=str(result["error"]),
                is_error=True,
            )
        return ToolResult(call_id=request.call_id, name=request.name, result=result)


def tool_spec_from_definition(definition: Any) -> ToolSpec:
    """Convert one registry definition into the Runtime-owned contract."""

    return ToolSpec(
        name=definition.name,
        description=definition.description,
        input_schema=definition.input_schema,
        recovery_policy=RecoveryPolicy(definition.recovery_policy),
        requires_approval=definition.requires_approval,
        side_effect=definition.side_effect,
        policy_class=_resolve_policy_class(definition),
    )


def tool_specs(registry: ToolRegistry) -> list[ToolSpec]:
    """Convert registered tools to Runtime-owned specs without importing Core."""

    return [
        tool_spec_from_definition(definition)
        for definition in (registry.get_definition(name) for name in sorted(registry.tool_names()))
        if definition is not None
    ]


def _resolve_policy_class(definition: Any) -> ToolPolicyClass:
    explicit = getattr(definition, "policy_class", "")
    explicit = getattr(explicit, "value", explicit)
    if str(explicit).strip():
        try:
            return ToolPolicyClass(str(explicit))
        except ValueError:
            return ToolPolicyClass.UNCLASSIFIED
    if definition.side_effect:
        return ToolPolicyClass.DURABLE_WRITE
    if (
        definition.name.startswith("code.")
        or "code_execution" in definition.capabilities
    ):
        return ToolPolicyClass.CODE_EXECUTION
    if definition.name.startswith("agent_"):
        return ToolPolicyClass.AGENT_DELEGATION
    if (
        definition.name.startswith(("web_", "personal_os."))
        or "web_search" in definition.capabilities
        or "web_fetch" in definition.capabilities
    ):
        return ToolPolicyClass.EXTERNAL_READ
    if (
        definition.name.startswith("finance.")
        or definition.name == "knowledge_search"
        or definition.name == "get_stored_data"
    ):
        return ToolPolicyClass.READ_ONLY
    if definition.name == "working_memory":
        return ToolPolicyClass.DURABLE_WRITE
    if (
        "image_generation" in definition.capabilities
        or "video_generation" in definition.capabilities
    ):
        return ToolPolicyClass.DURABLE_WRITE
    return ToolPolicyClass.UNCLASSIFIED
