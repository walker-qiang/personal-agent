"""Multi-agent LangGraph orchestration for the Matrix Agent.

Commander + Domain Agents architecture:
  classify → commander_plan → delegate → aggregate → reflection
"""

from __future__ import annotations

from .graph import build_graph
from .state import AgentState

__all__ = [
    "AgentState",
    "build_graph",
]