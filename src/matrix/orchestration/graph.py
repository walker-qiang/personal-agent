"""Multi-agent LangGraph orchestration builder.

Commander + Domain Agents architecture:
  commander_plan → delegate → aggregate → reflection
  With HITL: delegate → (confirm) → aggregate
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    aggregate_node,
    commander_plan_node,
    confirm_node,
    delegate_node,
    reflection_node,
)
from .state import AgentState


def build_graph() -> StateGraph:
    """Build the multi-agent LangGraph state graph.

    Flow:
    __start__ → commander_plan → delegate → (confirm) → aggregate → reflection → __end__
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("commander_plan", commander_plan_node)
    graph.add_node("delegate", delegate_node)
    graph.add_node("confirm", confirm_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("reflection", reflection_node)

    # Start → commander_plan
    graph.set_entry_point("commander_plan")

    # commander_plan → delegate (always)
    graph.add_edge("commander_plan", "delegate")

    # delegate → conditional: next step, confirm, or aggregate
    graph.add_conditional_edges(
        "delegate",
        _route_after_delegate,
        {
            "delegate": "delegate",
            "confirm": "confirm",
            "aggregate": "aggregate",
            "error": "aggregate",
        },
    )

    # confirm → aggregate
    graph.add_edge("confirm", "aggregate")

    # aggregate → reflection → end
    graph.add_edge("aggregate", "reflection")
    graph.add_edge("reflection", END)

    return graph


def _route_after_delegate(state: AgentState) -> str:
    """Route after delegate step: next step, confirm, or aggregate."""
    if state.get("error"):
        return "error"

    # Check if confirmation is needed
    if state.get("needs_confirmation") and not state.get("confirmed"):
        return "confirm"

    plan = state.get("delegation_plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan):
        return "aggregate"
    return "delegate"