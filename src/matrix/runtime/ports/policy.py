"""Policy evaluator port."""

from __future__ import annotations

from typing import Any, Protocol

from ..domain.policy import (
    EffectGrant,
    PolicyDecision,
    ToolExecutionContext,
)
from ..domain.tools import ToolSpec


class PolicyEvaluatorPort(Protocol):
    def evaluate(
        self,
        tool: ToolSpec,
        context: ToolExecutionContext,
        arguments: dict[str, Any],
        approval_grant: EffectGrant | None = None,
    ) -> PolicyDecision:
        ...
