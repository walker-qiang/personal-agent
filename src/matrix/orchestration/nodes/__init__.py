"""Orchestration nodes package.

Re-exports all node functions and helpers for backward compatibility
with the original nodes.py module.
"""

from __future__ import annotations

from ._helpers import (
    _build_history_context,
    _build_react_final_answer,
    _build_tools_for_llm,
    _check_early_stop,
    _classify_query_factuality,
    _evaluate_heuristic,
    _evaluate_sufficiency,
    _extract_json,
    _extract_media_urls,
    _fix_media_answer,
    _force_tool_call,
    _get_configurable,
    _is_hallucination,
    _is_high_risk,
    _is_refusal,
    _llm_summarize_from_results,
    _now_ts,
    _push_event,
    _route_after_react_evaluate,
    _route_after_react_llm,
    _trace,
    _trace_span,
    COMMANDER_AGGREGATE_PROMPT,
    COMMANDER_PLAN_PROMPT,
    DOMAIN_AGENT_REACT_SYSTEM,
    EVALUATOR_INTERVAL,
    MAX_CONSECUTIVE_FAILURES,
    MAX_CONSECUTIVE_NO_PROGRESS,
    MAX_PLAN_STEPS,
    MAX_REACT_ITERATIONS,
    MAX_SAME_TOOL_CALLS,
    MAX_SUBTASK_ITERATIONS,
    MAX_SUBTASKS,
    MAX_TOTAL_TOOL_CALLS,
    REFLECTION_PROMPT,
    REVISE_PROMPT,
)

from .react import (
    _react_execute_tool_calls,
    react_evaluate_node,
    react_llm_node,
    react_prepare_node,
    react_tool_node,
)

from .commander import (
    _domain_react_fallback,
    _run_domain_agent_react,
    aggregate_node,
    commander_plan_node,
    confirm_node,
    delegate_node,
    reflection_node,
)

__all__ = [
    # Helpers
    "_build_history_context",
    "_build_react_final_answer",
    "_build_tools_for_llm",
    "_check_early_stop",
    "_classify_query_factuality",
    "_evaluate_heuristic",
    "_evaluate_sufficiency",
    "_extract_json",
    "_extract_media_urls",
    "_fix_media_answer",
    "_force_tool_call",
    "_get_configurable",
    "_is_hallucination",
    "_is_high_risk",
    "_is_refusal",
    "_llm_summarize_from_results",
    "_now_ts",
    "_push_event",
    "_route_after_react_evaluate",
    "_route_after_react_llm",
    "_trace",
    "_trace_span",
    # React
    "_react_execute_tool_calls",
    "react_evaluate_node",
    "react_llm_node",
    "react_prepare_node",
    "react_tool_node",
    # Commander
    "_domain_react_fallback",
    "_run_domain_agent_react",
    "aggregate_node",
    "commander_plan_node",
    "confirm_node",
    "delegate_node",
    "reflection_node",
    # Constants
    "COMMANDER_AGGREGATE_PROMPT",
    "COMMANDER_PLAN_PROMPT",
    "DOMAIN_AGENT_REACT_SYSTEM",
    "EVALUATOR_INTERVAL",
    "MAX_CONSECUTIVE_FAILURES",
    "MAX_CONSECUTIVE_NO_PROGRESS",
    "MAX_PLAN_STEPS",
    "MAX_REACT_ITERATIONS",
    "MAX_SAME_TOOL_CALLS",
    "MAX_SUBTASK_ITERATIONS",
    "MAX_SUBTASKS",
    "MAX_TOTAL_TOOL_CALLS",
    "REFLECTION_PROMPT",
    "REVISE_PROMPT",
]
