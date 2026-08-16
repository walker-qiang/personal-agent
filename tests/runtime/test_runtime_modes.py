from __future__ import annotations

from matrix.orchestration.graph import _route_dag_first
from matrix.orchestration.state import AgentState


def test_single_step_always_routes_to_runtime() -> None:
    state = AgentState(
        user_message="hello",
        delegation_plan=[{"step": 1, "agent_id": "commander", "task": "hello"}],
        runtime_mode="legacy",  # compatibility field must not change routing
    )
    assert _route_dag_first(state) == "runtime_agent"


def test_runtime_single_step_routes_to_runtime_node() -> None:
    state = AgentState(
        user_message="hello",
        delegation_plan=[{"step": 1, "agent_id": "commander", "task": "hello"}],
        runtime_mode="runtime",
    )
    assert _route_dag_first(state) == "runtime_agent"


def test_multi_step_always_routes_to_runtime_delegate() -> None:
    state = AgentState(
        user_message="hello",
        delegation_plan=[
            {"step": 1, "agent_id": "a", "task": "a"},
            {"step": 2, "agent_id": "b", "task": "b", "depends_on": [1]},
        ],
        runtime_mode="runtime",
    )
    routed = _route_dag_first(state)
    assert isinstance(routed, list)
