"""LangGraph orchestration builder."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    classify_node,
    execute_node,
    plan_node,
    react_node,
    reflection_node,
    skill_node,
    summarize_node,
)
from .state import AgentState


def build_graph() -> StateGraph:
    """Build the LangGraph state graph for Agent orchestration.

    Flow:
    __start__ → classify → skill / react / plan → summarize → reflection → __end__
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify", classify_node)
    graph.add_node("skill", skill_node)
    graph.add_node("react", react_node)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("reflection", reflection_node)

    # Start → classify
    graph.set_entry_point("classify")

    # classify → route based on intent
    graph.add_conditional_edges(
        "classify",
        _route_by_intent,
        {
            "skill": "skill",
            "react": "react",
            "plan_execute": "plan",
            "summarize": "summarize",
        },
    )

    # skill → summarize
    graph.add_edge("skill", "summarize")

    # react → conditional: keep reacting or summarize
    graph.add_conditional_edges(
        "react",
        _route_after_react,
        {
            "react": "react",
            "summarize": "summarize",
        },
    )

    # plan → execute
    graph.add_edge("plan", "execute")

    # execute → conditional: next step or summarize
    graph.add_conditional_edges(
        "execute",
        _route_after_execute,
        {
            "execute": "execute",
            "summarize": "summarize",
        },
    )

    # summarize → reflection → end
    graph.add_edge("summarize", "reflection")
    graph.add_edge("reflection", END)

    return graph


def _route_by_intent(state: AgentState) -> str:
    """Route based on classified intent."""
    if state.get("error"):
        return "summarize"
    intent = state.get("intent", "react")
    if intent == "skill":
        return "skill"
    if intent == "plan_execute":
        return "plan_execute"
    return "react"


def _route_after_react(state: AgentState) -> str:
    """After react step: continue or summarize."""
    if state.get("error"):
        return "summarize"
    if state.get("final_answer") or state.get("needs_summary"):
        return "summarize"
    return "react"


def _route_after_execute(state: AgentState) -> str:
    """After execute step: next step or summarize."""
    if state.get("error"):
        return "summarize"
    if state.get("final_answer") or state.get("needs_summary"):
        return "summarize"
    plan = state.get("current_plan", [])
    count = state.get("tool_call_count", 0)
    if count >= len(plan):
        return "summarize"
    return "execute"