"""Agent system: multi-agent orchestration with Commander + Domain Agents."""

from __future__ import annotations

from .base import AgentDefinition
from .registry import AgentRegistry

__all__ = [
    "AgentDefinition",
    "AgentRegistry",
]