"""Tool system base types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Handler: receives arguments dict and returns result dict
ToolHandler = Callable[..., dict[str, Any]]


class FinanceToolError(Exception):
    """Raised for invalid read-only finance tool calls."""


def tool_error(
    tool_name: str,
    operation: str,
    reason: str,
    suggestion: str = "",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a structured tool error return value.

    Error messages should give the LLM enough context to self-correct:
    - what went wrong (specific, not just "failed")
    - why it went wrong (if determinable)
    - how to fix it (if a suggestion can be made)

    Args:
        tool_name: Name of the tool that failed.
        operation: What the tool was trying to do (e.g. "查询持仓").
        reason: Specific failure reason.
        suggestion: Suggested fix for the LLM.
        context: Relevant context (parameter summaries, etc.).

    Returns:
        {"error": "[tool_name] operation失败: reason。建议: suggestion。上下文: ..."}
    """
    parts = [f"[{tool_name}] {operation}失败: {reason}"]
    if suggestion:
        parts.append(f"建议: {suggestion}")
    if context:
        import json
        ctx_preview = json.dumps(context, ensure_ascii=False, default=str)[:200]
        parts.append(f"上下文: {ctx_preview}")
    return {"error": "。".join(parts)}


@dataclass(frozen=True)
class ToolDefinition:
    """Immutable definition of a registered tool.

    capabilities: tags describing what this tool can do, used by the commander
    to match tasks to tools. Examples: "market_data", "web_search", "image_generation".
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler = field(repr=False)
    capabilities: list[str] = field(default_factory=list)
    # Runtime metadata is optional so all existing registrations remain valid.
    requires_approval: bool = False
    recovery_policy: str = "manual"
    side_effect: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the tool definition in the format expected by LLM planners."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
