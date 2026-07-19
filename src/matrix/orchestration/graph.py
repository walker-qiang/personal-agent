"""LangGraph orchestration builder.

Commander plans → ReAct loop (prepare → LLM → tool → evaluate) → aggregate → reflection.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    _route_after_react_evaluate,
    _route_after_react_llm,
    aggregate_node,
    commander_plan_node,
    react_evaluate_node,
    react_llm_node,
    react_prepare_node,
    react_tool_node,
    reflection_node,
)
from .state import AgentState


def build_graph() -> StateGraph:
    """Build the LangGraph state graph.

    Flow:
    __start__ → commander_plan → react_prepare → react_llm ⇄ react_tool
                                  → react_evaluate → aggregate → reflection → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("commander_plan", commander_plan_node)
    graph.add_node("react_prepare", react_prepare_node)
    graph.add_node("react_llm", react_llm_node)
    graph.add_node("react_tool", react_tool_node)
    graph.add_node("react_evaluate", react_evaluate_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("reflection", reflection_node)

    graph.set_entry_point("commander_plan")

    # commander_plan → react_prepare
    graph.add_edge("commander_plan", "react_prepare")

    # ReAct loop
    graph.add_edge("react_prepare", "react_llm")

    graph.add_conditional_edges(
        "react_llm",
        _route_after_react_llm,
        {
            "react_tool": "react_tool",
            "react_evaluate": "react_evaluate",
        },
    )

    graph.add_edge("react_tool", "react_llm")

    graph.add_conditional_edges(
        "react_evaluate",
        _route_after_react_evaluate,
        {
            "react_llm": "react_llm",
            "aggregate": "aggregate",
        },
    )

    # aggregate → reflection → END
    graph.add_edge("aggregate", "reflection")
    graph.add_edge("reflection", END)

    return graph