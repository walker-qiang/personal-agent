"""Agent state for multi-agent LangGraph orchestration."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import add_messages


class AgentState(TypedDict):
    """State flowing through the multi-agent LangGraph orchestration graph.

    Flow: commander_plan → delegate → aggregate → reflection

    Fields annotated with ``operator.add`` use list concatenation / integer
    addition as state reducers, enabling parallel agent execution via
    LangGraph Send API fan-out → fan-in without data loss.
    """

    # Core conversation fields
    messages: Annotated[list, add_messages]
    user_message: str
    session_id: str

    # Classification
    intent: str  # "simple" (direct answer) or "delegate" (multi-agent)

    # Commander planning
    delegation_plan: list[dict[str, Any]]  # [{step, agent_id, task, skill_name, purpose}]
    current_step: int  # index into delegation_plan

    # Agent execution results — operator.add concatenates results from parallel branches
    agent_results: Annotated[list[dict[str, Any]], operator.add]

    # Tool results (accumulated from all domain agents)
    tool_results: Annotated[list[dict[str, Any]], operator.add]
    tool_call_count: Annotated[int, operator.add]

    # ReAct (for domain agent execution)
    react_iteration: int

    # Output
    final_answer: str
    needs_summary: bool  # signal that chat.py should stream the final answer
    skip_reflection: bool  # skip reflection review (e.g. commander pass-through with tool data)

    # Error
    error: str