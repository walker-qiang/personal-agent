"""Agent state for LangGraph orchestration."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import add_messages


class AgentState(TypedDict):
    """State flowing through the LangGraph orchestration graph."""

    # Core conversation fields
    messages: Annotated[list, add_messages]
    user_message: str
    session_id: str

    # Classification
    intent: str  # "skill" | "react" | "plan_execute"
    skill_name: str  # matched skill name, if intent=skill

    # Tool execution
    tool_results: list[dict[str, Any]]
    tool_call_count: int

    # Plan-Execute
    current_plan: list[dict[str, Any]]

    # ReAct
    react_iteration: int

    # Findings / output
    findings: list[str]
    final_answer: str

    # Streaming summarization
    needs_summary: bool

    # Error
    error: str