"""Multi-agent LangGraph orchestration builder.

Commander + Domain Agents architecture with Plan-and-Execute:
  commander_plan → _route_dag_first → delegate(s) → replan_node → aggregate → reflection
  With HITL: delegate → (confirm) → aggregate

DAG-based execution: when commander_plan produces a multi-step plan with
depends_on dependencies, the DAG router fans out only ready steps (whose
dependencies are all satisfied).  After each batch completes, replan_node
checks if the plan needs revision.

Parallel execution: LangGraph Send API fans out to multiple delegate_node
instances running concurrently.  Results are merged via operator.add reducers
on AgentState.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .nodes import (
    _get_ready_steps,
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
    replan_node,
    runtime_agent_node,
    runtime_confirm_node,
    runtime_delegate_node,
)
from .state import AgentState


def build_graph() -> StateGraph:
    """Build the multi-agent LangGraph state graph.

    Flow (single-step):
    __start__ → commander_plan → react_prepare → react_llm ⇄ react_tool
              → react_evaluate → aggregate → reflection → __end__

    Flow (multi-step DAG):
    __start__ → commander_plan → DAG route → delegate(s) → replan_node
              → (replan → commander_plan) or (next batch → delegate) or aggregate
              → reflection → __end__
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("commander_plan", commander_plan_node)
    graph.add_node("react_prepare", react_prepare_node)
    graph.add_node("runtime_agent", runtime_agent_node)
    graph.add_node("runtime_confirm", runtime_confirm_node)
    graph.add_node("runtime_delegate", runtime_delegate_node)
    graph.add_node("react_llm", react_llm_node)
    graph.add_node("react_tool", react_tool_node)
    graph.add_node("react_evaluate", react_evaluate_node)
    graph.add_node("delegate", delegate_node)
    graph.add_node("confirm", confirm_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("replan_node", replan_node)

    # Start → commander_plan
    graph.set_entry_point("commander_plan")

    # commander_plan → conditional:
    #   single-step: react_prepare (top-level ReAct with real-time streaming)
    #   multi-step:  DAG-based fan-out via Send("delegate")
    graph.add_conditional_edges(
        "commander_plan",
        _route_dag_first,
        {
            "react_prepare": "react_prepare",
            "runtime_agent": "runtime_agent",
        "delegate": "delegate",
        "runtime_delegate": "runtime_delegate",
            "aggregate": "aggregate",
        },
    )

    # ---- Top-level ReAct loop (single-step plans) ----
    # react_prepare → react_llm
    graph.add_edge("react_prepare", "react_llm")
    graph.add_conditional_edges(
        "runtime_agent",
        lambda state: "runtime_confirm" if state.get("needs_confirmation") else "aggregate",
        {"runtime_confirm": "runtime_confirm", "aggregate": "aggregate"},
    )
    graph.add_edge("runtime_confirm", "aggregate")
    graph.add_edge("runtime_delegate", "replan_node")

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

    # ---- Multi-step DAG path (Plan-and-Execute) ----
    # delegate → conditional:
    #   needs_confirmation → confirm (HITL pause for high-risk operations)
    #   otherwise → replan_node (check plan validity after each batch)
    graph.add_conditional_edges(
        "delegate",
        _route_after_delegate,
        {
            "confirm": "confirm",
            "replan_node": "replan_node",
        },
    )

    # confirm → replan_node (after user confirms/cancels, resume plan execution)
    graph.add_edge("confirm", "replan_node")

    # replan_node → conditional:
    #   needs_replan → commander_plan (regenerate plan)
    #   more steps ready → Send("delegate") for next batch
    #   all done → aggregate
    graph.add_conditional_edges(
        "replan_node",
        _route_after_replan,
        {
            "commander_plan": "commander_plan",
            "delegate": "delegate",
            "aggregate": "aggregate",
        },
    )

    # confirm → replan_node (set above)

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


# ── DAG Routing ──────────────────────────────────────────────────────────────

def _route_dag_first(state: AgentState):
    """Route after commander_plan: first DAG fan-out.

    Single-step plans (len ≤ 1):
      → "react_prepare" (top-level ReAct loop, backward compatible)

    Multi-step plans with DAG dependencies:
      → [Send("delegate", ...)] for each ready step (depends_on all satisfied).
      If no ready steps (shouldn't happen on first call), → "aggregate".
    """
    plan = state.get("delegation_plan", [])
    if len(plan) <= 1:
        if state.get("runtime_mode") == "runtime":
            return "runtime_agent"
        return "react_prepare"

    completed = state.get("completed_steps", [])
    ready = _get_ready_steps(plan, completed)

    if not ready:
        return "aggregate"

    target = "runtime_delegate" if state.get("runtime_mode") == "runtime" else "delegate"
    return [
        Send(target, {
            "current_step": _plan_index(plan, s["step"]),
            "delegation_plan": plan,
            "plan_type": state.get("plan_type", "agent"),
            "user_message": state.get("user_message", ""),
        "session_id": state.get("session_id", ""),
            "owner_id": state.get("owner_id", "default"),
            "runtime_mode": state.get("runtime_mode", "legacy"),
            "orchestration_run_id": state.get("orchestration_run_id", ""),
        })
        for s in ready
    ]


def _route_after_delegate(state: AgentState):
    """Route after delegate: HITL confirmation or replan.

    needs_confirmation → "confirm" (pause for user to approve high-risk ops)
    otherwise → "replan_node" (normal plan-and-execute flow)
    """
    if state.get("needs_confirmation") and state.get("pending_actions"):
        return "confirm"
    return "replan_node"


def _route_after_replan(state: AgentState):
    """Route after replan_node: continue, replan, or finish.

    needs_replan → "commander_plan" (regenerate plan)
    more steps ready → [Send("delegate", ...)] for next batch
    all done → "aggregate"
    """
    if state.get("needs_replan"):
        return "commander_plan"

    plan = state.get("delegation_plan", [])
    completed = state.get("completed_steps", [])
    ready = _get_ready_steps(plan, completed)

    if not ready:
        # All steps done or no more executable steps
        return "aggregate"

    target = "runtime_delegate" if state.get("runtime_mode") == "runtime" else "delegate"
    return [
        Send(target, {
            "current_step": _plan_index(plan, s["step"]),
            "delegation_plan": plan,
            "plan_type": state.get("plan_type", "agent"),
            "user_message": state.get("user_message", ""),
            "session_id": state.get("session_id", ""),
            "owner_id": state.get("owner_id", "default"),
            "runtime_mode": state.get("runtime_mode", "legacy"),
            "orchestration_run_id": state.get("orchestration_run_id", ""),
        })
        for s in ready
    ]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _plan_index(plan: list[dict], step_num: int) -> int:
    """Convert 1-based step number to 0-based index in the plan list."""
    for i, s in enumerate(plan):
        if s.get("step") == step_num:
            return i
    return 0
