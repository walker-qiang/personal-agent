"""Multi-agent LangGraph orchestration builder.

Commander + Domain Agents architecture:
  commander_plan → delegate (parallel via Send) → aggregate → reflection
  With HITL: delegate → (confirm) → aggregate

Parallel execution: when commander_plan produces a multi-agent plan,
LangGraph Send API fans out to multiple delegate_node instances running
concurrently.  Results are merged via operator.add reducers on AgentState.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.types import Send

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
    __start__ → commander_plan → delegate(s) → (confirm) → aggregate → reflection → __end__

    Single-agent plan: direct edge to delegate.
    Multi-agent plan:  Send-based fan-out → parallel delegate nodes → fan-in.
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

    # commander_plan → conditional: parallel fan-out or direct delegate
    graph.add_conditional_edges(
        "commander_plan",
        _route_after_commander,
        {"delegate": "delegate"},
    )

    # delegate → confirm or aggregate (no more loop-back)
    graph.add_conditional_edges(
        "delegate",
        _route_after_delegate,
        {
            "confirm": "confirm",
            "aggregate": "aggregate",
        },
    )

    # confirm → aggregate
    graph.add_edge("confirm", "aggregate")

    # aggregate → reflection → end
    graph.add_edge("aggregate", "reflection")
    graph.add_edge("reflection", END)

    return graph


def _route_after_commander(state: AgentState):
    """Route after commander plan: parallel fan-out for multi-agent, direct for single.

    When the plan has multiple agents, returns a list of Send objects to
    dispatch each delegate_node invocation concurrently.  LangGraph waits
    for all branches to complete before continuing from delegate's edges.

    NOTE: LangGraph 1.2.x Send API passes only the `arg` dict as the state
    update to the target node, NOT the merged state.  We must include all
    necessary state fields (delegation_plan, etc.) in the arg.
    """
    plan = state.get("delegation_plan", [])
    if len(plan) <= 1:
        return "delegate"
    # Fan out: one Send per agent, each with its own current_step
    # Include essential state fields that delegate_node needs
    return [
        Send("delegate", {
            "current_step": i,
            "delegation_plan": plan,
            "user_message": state.get("user_message", ""),
            "session_id": state.get("session_id", ""),
        })
        for i in range(len(plan))
    ]


def _route_after_delegate(state: AgentState) -> str:
    """Route after delegate: confirm (HITL) or aggregate.

    No more loop-back — each delegate invocation processes exactly one
    plan step.  In parallel mode, all branches complete before this
    router runs once on the merged state.
    """
    if state.get("error"):
        return "aggregate"

    # Check if confirmation is needed (HITL for high-risk tool calls)
    if state.get("needs_confirmation") and not state.get("confirmed"):
        return "confirm"

    return "aggregate"