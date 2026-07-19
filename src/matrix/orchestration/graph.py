"""Agent orchestration pipeline.

Runs: commander_plan → react_prepare → react_llm ⇄ react_tool → react_evaluate → aggregate → reflection
"""

from __future__ import annotations

from typing import Any, Iterator

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


# Config type alias (replaces langgraph.types.RunnableConfig)
AgentConfig = dict[str, Any]


def run_agent(state: AgentState, config: AgentConfig) -> Iterator[dict[str, Any]]:
    """Run the agent pipeline, yielding state after each node.

    Yields dicts (AgentState.model_dump()) after each node execution,
    compatible with the existing streaming interface in chat/_service.py.

    Config must contain:
        configurable.llm, configurable.pipeline_llm, configurable.agent_registry,
        configurable.full_tools, configurable.trace, configurable.history,
        configurable.event_queue, configurable.question
    """
    def _merge(s: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        """Merge update into state dict, using operator.add for Annotated fields."""
        for key, val in update.items():
            if key in ("agent_results", "tool_results", "messages"):
                s[key] = s.get(key, []) + val
            elif key == "tool_call_count":
                s[key] = s.get(key, 0) + val
            else:
                s[key] = val
        return s

    s = state.model_dump()

    # 1. commander_plan
    s = _merge(s, commander_plan_node(s, config=config))
    yield s

    # 2. react_prepare
    s = _merge(s, react_prepare_node(s, config=config))
    yield s

    # 3. ReAct loop: react_llm ⇄ react_tool → react_evaluate
    while True:
        s = _merge(s, react_llm_node(s, config=config))
        yield s

        if _route_after_react_llm(s) == "react_tool":
            s = _merge(s, react_tool_node(s, config=config))
            yield s
            continue

        s = _merge(s, react_evaluate_node(s, config=config))
        yield s

        if _route_after_react_evaluate(s) == "react_llm":
            continue
        break

    # 4. aggregate
    s = _merge(s, aggregate_node(s, config=config))
    yield s

    # 5. reflection
    s = _merge(s, reflection_node(s, config=config))
    yield s