"""Nested ReAct tool-call helper used by Agent-as-Tool compatibility.

Top-level Agent execution is Runtime-managed.  This module only retains the
guarded tool-call executor used by the nested commander helper.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ...tools import ToolRegistry
from ...tools.principal import tool_principal
from ...context import ToolResultRefStore

from ._helpers import (
    _now_ts,
    _push_event,
    _trace,
    CircuitBreaker,
    MAX_SAME_TOOL_CALLS,
    MAX_TOTAL_TOOL_CALLS,
)

logger = logging.getLogger("matrix.orchestration")

# ── Shared tool execution ────────────────────────────────────────────────────

def _react_execute_tool_calls(
    tool_calls_raw: list[dict],
    agent_tools: ToolRegistry,
    messages: list[dict],
    accumulated: list[dict],
    agent_id: str,
    session_id: str,
    cfg: dict,
    node_name: str,
    consecutive_failures: int,
    consecutive_no_progress: int,
    prev_result_count: int,
    push_events: bool = False,
    span_id: str = "",
    ref_store: ToolResultRefStore | None = None,
) -> dict:
    """Execute tool calls from LLM response with all guards.

    Used by the nested Agent-as-Tool ReAct path.

    Guards applied in order: batch-dedup → total-calls → same-tool → args-dedup
    """
    new_tool_results: list[dict[str, Any]] = []
    new_messages = list(messages)
    _called_in_batch: set[tuple[str, str]] = set()

    executed = 0
    failed = 0

    for tc_raw in tool_calls_raw:
        func = tc_raw.get("function", {})
        name = func.get("name", "")
        try:
            arguments = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            arguments = {}

        if name not in agent_tools.tool_names():
            new_messages.append({
                "role": "tool",
                "tool_call_id": tc_raw.get("id", ""),
                "content": json.dumps(
                    {"error": f"工具 {name} 不存在"},
                    ensure_ascii=False,
                ),
            })
            continue

        call_key = (name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))

        if not _pass_tool_guards(
            name, arguments, call_key, accumulated, new_tool_results,
            _called_in_batch, cfg,
        ):
            new_messages.append({
                "role": "tool",
                "tool_call_id": tc_raw.get("id", ""),
                "content": json.dumps(
                    {"skipped": True, "reason": "工具调用被防重复机制拦截，请基于已有结果回答或尝试其他工具"},
                    ensure_ascii=False,
                ),
            })
            continue

        _called_in_batch.add(call_key)
        executed += 1

        ok, tr = _execute_single_tool(
            name, arguments, tc_raw, agent_tools, agent_id, session_id,
            cfg, node_name, span_id, push_events, ref_store=ref_store,
        )
        new_tool_results.append(tr)
        new_messages.append({
            "role": "tool",
            "tool_call_id": tc_raw.get("id", ""),
            "content": json.dumps(tr.get("error", tr.get("result", {})), ensure_ascii=False),
        })
        if not ok:
            failed += 1

    # Track failures and progress
    if executed > 0 and failed >= executed:
        consecutive_failures += 1
    elif executed > 0:
        consecutive_failures = 0

    total_after = len(accumulated) + len(new_tool_results)
    if len(new_tool_results) > 0:
        consecutive_no_progress = 0
        prev_result_count = total_after
    else:
        consecutive_no_progress += 1

    return {
        "messages": new_messages,
        "new_tool_results": new_tool_results,
        "executed": executed,
        "failed": failed,
        "consecutive_failures": consecutive_failures,
        "consecutive_no_progress": consecutive_no_progress,
        "prev_result_count": prev_result_count,
        "force_summarize": executed == 0,
    }


def _pass_tool_guards(
    name: str,
    arguments: dict,
    call_key: tuple[str, str],
    accumulated: list[dict],
    new_tool_results: list[dict],
    called_in_batch: set[tuple[str, str]],
    cfg: dict,
) -> bool:
    """Check all 4 guards; return True if the tool call should proceed."""
    # Guard 1: batch-dedup
    if call_key in called_in_batch:
        logger.info("ReAct: batch-dedup skip %s (same args in batch)", name)
        _push_event(cfg, "progress", {"message": f"跳过重复调用 {name}（同批次内相同参数）"})
        return False

    # Guard 2: total-calls
    total_calls = len(accumulated) + len(new_tool_results)
    if total_calls >= MAX_TOTAL_TOOL_CALLS:
        logger.info("ReAct: skipping %s (total %d >= %d)", name, total_calls, MAX_TOTAL_TOOL_CALLS)
        _push_event(cfg, "progress", {"message": f"已收集 {total_calls} 条数据，跳过剩余工具调用"})
        return False

    # Guard 3: same-tool
    same_tool_count = (
        sum(1 for tr in accumulated if tr.get("name") == name)
        + sum(1 for tr in new_tool_results if tr.get("name") == name)
    )
    if same_tool_count >= MAX_SAME_TOOL_CALLS:
        logger.info("ReAct: skipping %s (%d >= %d)", name, same_tool_count, MAX_SAME_TOOL_CALLS)
        _push_event(cfg, "progress", {"message": f"已调用 {name} {same_tool_count} 次，跳过重复调用"})
        return False

    # Guard 4: args-dedup across accumulated + new
    args_key = json.dumps(arguments, sort_keys=True)
    for prev in accumulated + new_tool_results:
        prev_name = prev.get("name", "")
        prev_args_key = json.dumps(prev.get("arguments", {}), sort_keys=True)
        if prev_name == name and prev_args_key == args_key:
            logger.info("ReAct: dedup skip %s (same args key=%s...)", name, args_key[:80])
            return False

    logger.info("ReAct: dedup no-match for %s args_key=%s", name, args_key[:80])
    return True


def _execute_single_tool(
    name: str,
    arguments: dict,
    tc_raw: dict,
    agent_tools: ToolRegistry,
    agent_id: str,
    session_id: str,
    cfg: dict,
    node_name: str,
    span_id: str,
    push_events: bool,
    ref_store: ToolResultRefStore | None = None,
) -> tuple[bool, dict]:
    """Execute a single tool call and return (ok, tool_result_dict).

    If ref_store is provided and the result exceeds thresholds, the raw
    result is externalized and a reference object is returned instead.
    This keeps the LLM context lean while preserving full data for retrieval.
    """
    started = time.perf_counter()

    if push_events:
        _push_event(cfg, "tool_call", {"name": name, "args": arguments})

    try:
        policy = cfg.get("execution_policy")
        with tool_principal(
            str(cfg.get("user_id", "default")),
            session_id,
            getattr(policy, "mode", "read_only"),
            bool(getattr(policy, "allow_external_effects", False)),
        ):
            tool_result = agent_tools.call(name, arguments, session_id=session_id)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

        # call() returns {"error": ...} on tool execution failures (Phase 2 pipeline).
        # Check for error key to drive circuit breaker and tracing.
        if isinstance(tool_result, dict) and "error" in tool_result:
            _trace(cfg, {
                "session_id": session_id,
                "event_type": "tool_call",
                "node_name": node_name,
                "agent_id": agent_id,
                "ok": False, "tool_name": name, "arguments": arguments,
                "error": tool_result["error"][:300], "elapsed_ms": elapsed_ms, "ts": _now_ts(),
                "parent_span_id": span_id,
            })
            if push_events:
                _push_event(cfg, "tool_result", {"name": name, "error": tool_result["error"][:200]})
            # Note: circuit breaker recording is handled by ToolRegistry.call()
            # for actual handler failures. Pre-execution checks (validation,
            # guard blocks) should NOT count as failures.
            return False, {"name": name, "arguments": arguments,
                           "error": tool_result["error"], "elapsed_ms": elapsed_ms}

        # L1: ToolResultRefStore — externalize large results
        if ref_store is not None and ref_store.should_store(tool_result):
            stored = ref_store.store(name, tool_result)
            ref_obj = ref_store.build_ref_object(stored)
            logger.info(
                "ReAct: externalized tool result: tool=%s ref_id=%s orig_len=%d",
                name, stored.ref_id, stored.original_length,
            )
            _trace(cfg, {
                "session_id": session_id,
                "event_type": "tool_call",
                "node_name": node_name,
                "agent_id": agent_id,
                "ok": True, "tool_name": name, "arguments": arguments,
                "result": f"[EXTERNALIZED] refId={stored.ref_id} summary={stored.summary}",
                "elapsed_ms": elapsed_ms, "ts": _now_ts(),
                "parent_span_id": span_id,
            })
            if push_events:
                _push_event(cfg, "tool_result", {
                    "name": name,
                    "result": ref_obj,
                    "externalized": True,
                    "refId": stored.ref_id,
                })
            return True, {
                "name": name, "arguments": arguments,
                "result": ref_obj, "elapsed_ms": elapsed_ms,
                "externalized": True, "refId": stored.ref_id,
            }
        # end L1

        _trace(cfg, {
            "session_id": session_id,
            "event_type": "tool_call",
            "node_name": node_name,
            "agent_id": agent_id,
            "ok": True, "tool_name": name, "arguments": arguments,
            "result": str(tool_result)[:500],
            "elapsed_ms": elapsed_ms, "ts": _now_ts(),
            "parent_span_id": span_id,
        })
        if push_events:
            _push_event(cfg, "tool_result", {"name": name, "result": tool_result})
        return True, {"name": name, "arguments": arguments, "result": tool_result, "elapsed_ms": elapsed_ms}

    except Exception as err:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        logger.error(
            "ReAct: _execute_single_tool exception: tool=%s error=%s: %s",
            name, type(err).__name__, str(err)[:200],
        )
        _trace(cfg, {
            "session_id": session_id,
            "event_type": "tool_call",
            "node_name": node_name,
            "agent_id": agent_id,
            "ok": False, "tool_name": name, "arguments": arguments,
            "error": str(err), "elapsed_ms": elapsed_ms, "ts": _now_ts(),
            "parent_span_id": span_id,
        })
        if push_events:
            _push_event(cfg, "tool_result", {"name": name, "error": str(err)[:200]})
        # Record escape exceptions (e.g. ToolGuardError before fix) in circuit breaker
        breaker: CircuitBreaker | None = cfg.get("circuit_breaker")
        if breaker is not None:
            breaker.record_failure(name)
        return False, {"name": name, "arguments": arguments, "error": str(err), "elapsed_ms": elapsed_ms}
