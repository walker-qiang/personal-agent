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
    _extract_media_urls,
    _focus_tools_for_task,
    _fix_media_answer,
    _force_tool_call,
    _get_configurable,
    _is_hallucination,
    _is_high_risk,
    _is_refusal,
    _llm_summarize_from_results,
    _now_ts,
    _push_event,
    _requires_browser,
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
    REFLEXION_PROMPT,
    REFLEXION_RETRY_PROMPT,
    REPLAN_PROMPT,
    REVISE_PROMPT,
)

from .react import (
    _react_execute_tool_calls,
)
from .runtime import runtime_agent_node, runtime_confirm_node, runtime_delegate_node

from .commander import (
    _domain_react_fallback,
    _get_ready_steps,
    _run_domain_agent_react,
    aggregate_node,
    commander_plan_node,
    reflection_node,
    replan_node,
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
    "_extract_media_urls",
    "_focus_tools_for_task",
    "_fix_media_answer",
    "_force_tool_call",
    "_get_configurable",
    "_is_hallucination",
    "_is_high_risk",
    "_is_refusal",
    "_llm_summarize_from_results",
    "_now_ts",
    "_push_event",
    "_requires_browser",
    "_trace",
    "_trace_span",
    # React
    "_react_execute_tool_calls",
    "runtime_agent_node",
    "runtime_confirm_node",
    "runtime_delegate_node",
    # Commander
    "_domain_react_fallback",
    "_run_domain_agent_react",
    "aggregate_node",
    "commander_plan_node",
    "reflection_node",
    "replan_node",
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
    "REFLEXION_PROMPT",
    "REFLEXION_RETRY_PROMPT",
    "REPLAN_PROMPT",
    "REVISE_PROMPT",
]
