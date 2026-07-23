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
    _route_after_react_evaluate,
    _route_after_react_llm,
    aggregate_node,
    commander_plan_node,
    confirm_node,
    delegate_node,
    react_evaluate_node,
    react_llm_node,
    react_prepare_node,
    react_tool_node,
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
    graph.add_node("react_prepare", react_prepare_node)
    graph.add_node("react_llm", react_llm_node)
    graph.add_node("react_tool", react_tool_node)
    graph.add_node("react_evaluate", react_evaluate_node)
    graph.add_node("delegate", delegate_node)
    graph.add_node("confirm", confirm_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("reflection", reflection_node)

    # Start → commander_plan
    graph.set_entry_point("commander_plan")

    # commander_plan → conditional:
    #   single-step: react_prepare (top-level ReAct with real-time streaming)
    #   multi-step:  delegate (Send fan-out, subgraph ReAct)
    graph.add_conditional_edges(
        "commander_plan",
        _route_after_commander,
        {
            "react_prepare": "react_prepare",
            "delegate": "delegate",
        },
    )

    # ---- Top-level ReAct loop (single-step plans) ----
    # react_prepare → react_llm
    graph.add_edge("react_prepare", "react_llm")

    # react_llm → conditional: tool calls → react_tool, otherwise → react_evaluate
    graph.add_conditional_edges(
        "react_llm",
        _route_after_react_llm,
        {
            "react_tool": "react_tool",
            "react_evaluate": "react_evaluate",
        },
    )

    # react_tool → react_evaluate (check early stop / sufficiency before next LLM call)
    graph.add_edge("react_tool", "react_evaluate")

    # react_evaluate → conditional: not done → react_llm (loop), done → aggregate
    graph.add_conditional_edges(
        "react_evaluate",
        _route_after_react_evaluate,
        {
            "react_llm": "react_llm",
            "aggregate": "aggregate",
        },
    )

    # ---- Multi-step plan path (existing) ----
    # delegate → confirm or aggregate
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
    # reflection → END (with conditional Reflexion retry)
    def _after_reflection(state: AgentState) -> str:
        """Route after reflection: retry via aggregate or finish."""
        if state.get("needs_reflexion_retry"):
            return "aggregate"
        return END

    graph.add_conditional_edges(
        "reflection",
        _after_reflection,
        {"aggregate": "aggregate", END: END},
    )

    return graph


def _route_after_commander(state: AgentState):
    """Route after commander plan.

    Single-step plans:
      → "react_prepare" (top-level ReAct loop with real-time per-node streaming)
      The react_prepare node sets up the react context, then the
      react_llm → react_tool → react_evaluate loop runs as top-level
      LangGraph nodes, yielding SSE events after each node completes.

    Multi-step plans (multi-agent or subtask decomposition):
      → list of Send("delegate") for parallel fan-out.
      Each delegate_node runs the subgraph ReAct internally.
      LangGraph waits for all branches to complete before continuing.

    NOTE: LangGraph 1.2.x Send API passes only the `arg` dict as the state
    update to the target node, NOT the merged state.  We must include all
    necessary state fields (delegation_plan, plan_type, etc.) in the arg.
    """
    plan = state.get("delegation_plan", [])
    if len(plan) <= 1:
        return "react_prepare"
    # Fan out: one Send per step, each with its own current_step
    # Include essential state fields that delegate_node needs
    return [
        Send("delegate", {
            "current_step": i,
            "delegation_plan": plan,
            "plan_type": state.get("plan_type", "agent"),
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