"""Multi-agent LangGraph orchestration builder.

Commander + Domain Agents architecture:
  commander_plan → delegate → aggregate → reflection
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    aggregate_node,
    commander_plan_node,
    delegate_node,
    reflection_node,
)
from .state import AgentState


def build_graph() -> StateGraph:
    """Build the multi-agent LangGraph state graph.

    Flow:
    __start__ → commander_plan → delegate → aggregate → reflection → __end__
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("commander_plan", commander_plan_node)
    graph.add_node("delegate", delegate_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("reflection", reflection_node)

    # Start → commander_plan
    graph.set_entry_point("commander_plan")

    # commander_plan → delegate (always)
    graph.add_edge("commander_plan", "delegate")

    # delegate → conditional: next step or aggregate
    graph.add_conditional_edges(
        "delegate",
        _route_after_delegate,
        {
            "delegate": "delegate",
            "aggregate": "aggregate",
            "error": "aggregate",
        },
    )

    # aggregate → reflection → end
    graph.add_edge("aggregate", "reflection")
    graph.add_edge("reflection", END)

    return graph


def _route_after_delegate(state: AgentState) -> str:
    """Route after delegate step: next step or aggregate."""
    if state.get("error"):
        return "error"

    plan = state.get("delegation_plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan):
        return "aggregate"
    return "delegate"