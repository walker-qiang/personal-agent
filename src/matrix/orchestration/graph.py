"""Multi-agent LangGraph orchestration builder.

Commander + Domain Agents architecture with Plan-and-Execute:
  commander_plan → _route_dag_first → Runtime agent(s) → replan_node → aggregate → reflection

DAG-based execution: when commander_plan produces a multi-step plan with
depends_on dependencies, the DAG router fans out only ready steps (whose
dependencies are all satisfied).  After each batch completes, replan_node
checks if the plan needs revision.

Parallel execution: LangGraph Send API fans out to multiple Runtime operations.
Results are merged via operator.add reducers on AgentState.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .nodes import (
    _get_ready_steps,
    aggregate_node,
    commander_plan_node,
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
    __start__ → commander_plan → runtime_agent → runtime_confirm?
              → aggregate → reflection → __end__

    Flow (multi-step DAG):
    __start__ → commander_plan → DAG route → runtime_delegate(s) → replan_node
              → (replan → commander_plan) or (next batch → runtime_delegate)
              or aggregate
              → reflection → __end__
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("commander_plan", commander_plan_node)
    graph.add_node("runtime_agent", runtime_agent_node)
    graph.add_node("runtime_confirm", runtime_confirm_node)
    graph.add_node("runtime_delegate", runtime_delegate_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("replan_node", replan_node)

    # Start → commander_plan
    graph.set_entry_point("commander_plan")

    # commander_plan → conditional:
    #   single-step: runtime_agent
    #   multi-step:  DAG-based fan-out via Send("runtime_delegate")
    graph.add_conditional_edges(
        "commander_plan",
        _route_dag_first,
        {
            "runtime_agent": "runtime_agent",
            "runtime_delegate": "runtime_delegate",
            "aggregate": "aggregate",
        },
    )

    graph.add_conditional_edges(
        "runtime_agent",
        lambda state: "runtime_confirm" if state.get("needs_confirmation") else "aggregate",
        {"runtime_confirm": "runtime_confirm", "aggregate": "aggregate"},
    )
    graph.add_conditional_edges(
        "runtime_confirm",
        lambda state: (
            "runtime_confirm"
            if state.get("needs_confirmation")
            else (
                "replan_node"
                if len(state.get("delegation_plan", [])) > 1
                else "aggregate"
            )
        ),
        {
            "runtime_confirm": "runtime_confirm",
            "replan_node": "replan_node",
            "aggregate": "aggregate",
        },
    )
    graph.add_conditional_edges(
        "runtime_delegate",
        lambda state: (
            "runtime_confirm"
            if state.get("needs_confirmation")
            else "replan_node"
        ),
        {
            "runtime_confirm": "runtime_confirm",
            "replan_node": "replan_node",
        },
    )

    # replan_node → conditional:
    #   needs_replan → commander_plan (regenerate plan)
    #   more steps ready → Send("runtime_delegate") for next batch
    #   all done → aggregate
    graph.add_conditional_edges(
        "replan_node",
        _route_after_replan,
        {
            "commander_plan": "commander_plan",
            "runtime_delegate": "runtime_delegate",
            "aggregate": "aggregate",
        },
    )

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
      → "runtime_agent"

    Multi-step plans with DAG dependencies:
      → [Send("runtime_delegate", ...)] for each ready step
        (depends_on all satisfied).
      If no ready steps (shouldn't happen on first call), → "aggregate".
    """
    plan = state.get("delegation_plan", [])
    if len(plan) <= 1:
        return "runtime_agent"

    completed = state.get("completed_steps", [])
    ready = _get_ready_steps(
        plan,
        completed,
        state.get("completed_step_refs", []),
        state.get("plan_revision", 0),
    )

    if not ready:
        return "aggregate"

    return [
        Send("runtime_delegate", {
            "current_step": _plan_index(plan, s["step"]),
            "delegation_plan": plan,
            "plan_type": state.get("plan_type", "agent"),
            "user_message": state.get("user_message", ""),
            "session_id": state.get("session_id", ""),
            "owner_id": state.get("owner_id", "default"),
            "plan_revision": state.get("plan_revision", 0),
            "orchestration_run_id": state.get("orchestration_run_id", ""),
            "agent_results": state.get("agent_results", []),
            "completed_step_refs": state.get("completed_step_refs", []),
        })
        for s in ready
    ]


def _route_after_replan(state: AgentState):
    """Route after replan_node: continue, replan, or finish.

    needs_replan → "commander_plan" (regenerate plan)
    more steps ready → [Send("runtime_delegate", ...)] for next batch
    all done → "aggregate"
    """
    if state.get("needs_replan"):
        return "commander_plan"

    plan = state.get("delegation_plan", [])
    completed = state.get("completed_steps", [])
    ready = _get_ready_steps(
        plan,
        completed,
        state.get("completed_step_refs", []),
        state.get("plan_revision", 0),
    )

    if not ready:
        # All steps done or no more executable steps
        return "aggregate"

    return [
        Send("runtime_delegate", {
            "current_step": _plan_index(plan, s["step"]),
            "delegation_plan": plan,
            "plan_type": state.get("plan_type", "agent"),
            "user_message": state.get("user_message", ""),
            "session_id": state.get("session_id", ""),
            "owner_id": state.get("owner_id", "default"),
            "plan_revision": state.get("plan_revision", 0),
            "orchestration_run_id": state.get("orchestration_run_id", ""),
            "agent_results": state.get("agent_results", []),
            "completed_step_refs": state.get("completed_step_refs", []),
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
