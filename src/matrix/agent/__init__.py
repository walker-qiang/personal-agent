"""Agent system: multi-agent orchestration with Commander + Domain Agents."""

from __future__ import annotations

from .base import AgentDefinition
from .presets import AgentPreset, PRESETS, resolve_agent_policy
from .registry import AgentRegistry

__all__ = [
    "AgentDefinition",
    "AgentPreset",
    "AgentRegistry",
    "PRESETS",
    "resolve_agent_policy",
]
