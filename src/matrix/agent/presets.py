"""User-facing AgentMode/Preset catalog.

Presets are application configuration.  They resolve to the small generic
``ExecutionPolicy`` understood by Runtime; Runtime never imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from ..runtime.domain.requests import ExecutionPolicy


@dataclass(frozen=True)
class AgentPreset:
    """Named user configuration, not a second Agent implementation."""

    name: str
    mode: str = "read_only"
    output_style: str = "default"
    debug_trace: bool = False

    def resolve(self, *, mode: str = "") -> ExecutionPolicy:
        resolved_mode = mode or self.mode
        approval_mode = os.environ.get("MATRIX_WRITEBACK_APPROVAL_MODE", "manual").strip().lower()
        configured_operations = tuple(
            item.strip()
            for item in os.environ.get("MATRIX_WRITEBACK_AUTO_OPERATIONS", "").split(",")
            if item.strip()
        )
        if resolved_mode == "read_only":
            return ExecutionPolicy(
                mode=resolved_mode,
                preset=self.name,
                allow_external_effects=False,
                require_approval=True,
                approval_mode="manual",
                debug_trace=self.debug_trace,
                output_style=self.output_style,
            )
        if resolved_mode == "writeback":
            return ExecutionPolicy(
                mode=resolved_mode,
                preset=self.name,
                allow_external_effects=True,
                require_approval=True,
                approval_mode=approval_mode,
                auto_approve_operations=configured_operations,
                debug_trace=self.debug_trace,
                output_style=self.output_style,
            )
        raise ValueError(
            f"unsupported agent mode: {resolved_mode}; expected read_only or writeback"
        )


PRESETS: dict[str, AgentPreset] = {
    "default": AgentPreset(name="default"),
    "investment_research": AgentPreset(
        name="investment_research", output_style="evidence_first",
    ),
}


def resolve_agent_policy(*, mode: str = "", preset: str = "") -> ExecutionPolicy:
    """Resolve a request's optional mode/preset into a Runtime policy."""

    preset_name = preset.strip() or "default"
    selected = PRESETS.get(preset_name)
    if selected is None:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(
            f"unknown agent preset: {preset_name}; available presets: {available}"
        )
    normalized_mode = mode.strip().lower()
    return selected.resolve(mode=normalized_mode)
