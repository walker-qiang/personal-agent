"""Agent state for multi-agent LangGraph orchestration.

Uses Pydantic BaseModel for runtime validation, default values, and
serialization — the recommended approach for LangGraph production deployments.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """State flowing through the multi-agent LangGraph orchestration graph.

    Flow: commander_plan → delegate → aggregate → reflection

    Fields annotated with ``operator.add`` use list concatenation / integer
    addition as state reducers, enabling parallel agent execution via
    LangGraph Send API fan-out → fan-in without data loss.

    Pydantic provides:
    - Runtime type validation on every state transition
    - Automatic default values (no missing-key errors)
    - JSON serialization for tracing and debugging

    Dict-like access (__getitem__, get, __contains__) is provided for
    backward compatibility with existing node code that uses state["key"]
    and state.get("key") patterns.
    """

    model_config = {"extra": "allow"}

    # Core conversation fields
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    user_message: str = ""
    session_id: str = ""

    # Classification
    intent: str = ""  # "simple" (direct answer) or "delegate" (multi-agent)

    # Commander planning
    delegation_plan: list[dict[str, Any]] = Field(
        default_factory=list,
        description="[{step, agent_id, task, skill_name, purpose}]",
    )
    current_step: int = 0  # index into delegation_plan

    # Agent execution results — operator.add concatenates results from parallel branches
    agent_results: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)

    # Tool results (accumulated from all domain agents)
    tool_results: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)
    tool_call_count: Annotated[int, operator.add] = 0

    # ReAct (for domain agent execution)
    react_iteration: int = 0

    # ReAct context dict — used by the split ReAct nodes (react_prepare → react_llm → react_tool → react_evaluate)
    # Contains: messages, system, tools_json, question, iteration, consecutive_failures,
    #           consecutive_no_progress, prev_result_count, agent_id, agent_name, answer
    react: dict[str, Any] = Field(default_factory=dict)

    # Output
    final_answer: str = ""
    needs_summary: bool = False  # signal that chat.py should stream the final answer
    skip_reflection: bool = False  # skip reflection review (e.g. commander pass-through with tool data)

    # Error
    error: str = ""

    # HITL (Human-in-the-Loop)
    needs_confirmation: bool = False
    confirmed: bool = False
    pending_actions: list[dict[str, Any]] = Field(default_factory=list)

    # ---- Dict-like access for backward compatibility ----

    def __getitem__(self, key: str) -> Any:
        """Support state["key"] pattern used in node functions."""
        if key in type(self).model_fields:
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Support state.get("key", default) pattern."""
        if key in type(self).model_fields:
            return getattr(self, key)
        return default

    def __contains__(self, key: str) -> bool:
        """Support "key" in state pattern."""
        return key in type(self).model_fields or hasattr(self, key)