"""ReAct loop nodes: prepare → LLM → tool → evaluate.

Shared tool execution logic in _react_execute_tool_calls.
Both single-step and multi-agent paths use the same implementation.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from ...llm import LLMError, LLMClient, FunctionCallResult
from ...tools import FinanceToolError, ToolRegistry
from ...agent.registry import AgentRegistry

from ._helpers import (
    DOMAIN_AGENT_REACT_SYSTEM,
    _build_history_context,
    _build_react_final_answer,
    _build_tools_for_llm,
    _check_early_stop,
    _classify_query_factuality,
    _evaluate_sufficiency,
    _get_configurable,
    _is_refusal,
    _now_ts,
    _push_event,
    _trace,
    _trace_span,
    MAX_SAME_TOOL_CALLS,
    MAX_TOTAL_TOOL_CALLS,
    EVALUATOR_INTERVAL,
)
from ..state import AgentState

logger = logging.getLogger("matrix.orchestration")

def react_prepare_node(state: AgentState, *, config: dict[str, Any]) -> dict[str, Any]:
    """Prepare the ReAct context for a single-step domain agent execution.

    Sets up the react dict with system prompt, messages, tools, and
    tracking variables.  Subsequent nodes (react_llm, react_tool,
    react_evaluate) use this context to drive the ReAct loop with
    per-node streaming yields.
    """
    cfg = _get_configurable(config)
    agent_registry: AgentRegistry = cfg["agent_registry"]
    full_tools: ToolRegistry = cfg["full_tools"]

    plan = state.get("delegation_plan", [])
    current_step = state.get("current_step", 0)
    step = plan[current_step] if current_step < len(plan) else {}
    agent_id = step.get("agent_id", "")
    task = step.get("task", state["user_message"])

    agent_def = agent_registry.get(agent_id)
    if agent_def is None:
        return {"error": f"Agent not found: {agent_id}"}

    agent_tools = agent_registry.build_tool_registry(agent_id, full_tools)
    llm_tools = _build_tools_for_llm(agent_tools)
    history_context = _build_history_context(cfg.get("history", []))
    task_content = history_context + f"请完成以下任务：{task}"

    system_prompt = DOMAIN_AGENT_REACT_SYSTEM.format(
        agent_name=agent_def.name,
        persona=agent_def.persona,
        task=task,
        today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )

    react = {
        "messages": [{"role": "user", "content": task_content}],
        "system": system_prompt,
        "tools_json": llm_tools,
        "question": state["user_message"],
        "iteration": 0,
        "consecutive_failures": 0,
        "consecutive_no_progress": 0,
        "prev_result_count": 0,
        "agent_id": agent_id,
        "agent_name": agent_def.name,
        "answer": "",
    }

    _push_event(cfg, "progress", {"message": "Agent 开始分析任务，调用工具获取数据..."})

    return {"react": react}



def react_llm_node(state: AgentState, *, config: dict[str, Any]) -> dict[str, Any]:
    """Call the LLM with function calling to decide the next action.

    Returns updated react context with the LLM response appended to messages.
    If the LLM returns tool_calls, the route sends execution to react_tool_node.
    If the LLM returns content without tool_calls, the route sends to
    react_evaluate_node for sufficiency checking.
    """
    cfg = _get_configurable(config)
    llm = cfg["llm"]
    agent_registry: AgentRegistry = cfg["agent_registry"]
    full_tools: ToolRegistry = cfg["full_tools"]

    react = state.get("react", {})
    if not react:
        return {}

    agent_id = react.get("agent_id", "")
    agent_tools = agent_registry.build_tool_registry(agent_id, full_tools)
    llm_tools = _build_tools_for_llm(agent_tools)

    messages = react.get("messages", [])
    system_prompt = react.get("system", "")
    question = react.get("question", "")
    iteration = react.get("iteration", 0)

    with _trace_span(cfg, "react_llm", session_id=state.get("session_id", ""),
                     agent_id=agent_id, iteration=iteration):
        try:
            temp = _classify_query_factuality(question)
            result: FunctionCallResult = llm.function_call(system_prompt, messages, llm_tools, temperature=temp)
        except (LLMError, ConnectionError, TimeoutError, ValueError, OSError) as e:
            msg_len = sum(len(str(m.get("content", ""))) for m in messages)
            logger.error("ReAct: LLM call failed in react_llm_node: %s: %s (msg_count=%d, total_chars=%d)", type(e).__name__, str(e)[:200], len(messages), msg_len)
            _push_event(cfg, "progress", {"message": f"LLM 调用失败: {type(e).__name__}"})
            return {
                "react": {**react, "answer": f"无法完成任务，LLM 调用失败: {type(e).__name__}。请重试。", "error": f"LLM call failed: {type(e).__name__}"},
                "error": f"LLM call failed: {type(e).__name__}",
            }

    # Push thinking event
    if result.content:
        _push_event(cfg, "thinking", {"content": result.content.strip()})
    elif result.tool_calls:
        tool_names = [tc.name for tc in result.tool_calls if tc.name in agent_tools.tool_names()]
        if tool_names:
            _push_event(cfg, "thinking", {"content": f"正在调用 {'、'.join(tool_names)} 获取数据..."})

    # Append assistant message
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": result.content or ""}
    if result.tool_calls:
        api_tool_calls = []
        for tc in result.tool_calls:
            api_tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            })
        assistant_msg["tool_calls"] = api_tool_calls

    new_messages = list(messages) + [assistant_msg]
    return {"react": {**react, "messages": new_messages}}



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
) -> dict:
    """Execute tool calls from LLM response with all guards.

    Shared by both ReAct paths:
    - Top-level: react_tool_node (push_events=False, SSE via state emission)
    - Subgraph:   _run_domain_agent_react (push_events=True, direct SSE)

    Guards applied in order:
    1. Batch-dedup: same (name, args) within a single LLM response
    2. Total-calls: MAX_TOTAL_TOOL_CALLS across all iterations
    3. Same-tool:   MAX_SAME_TOOL_CALLS for the same tool name
    4. Args-dedup:  same (name, args) across all accumulated results

    Returns a dict with all updated state values.
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
            continue

        # Guard 1: batch-dedup
        call_key = (name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
        if call_key in _called_in_batch:
            logger.info("ReAct: batch-dedup skip %s (same args in batch)", name)
            _push_event(cfg, "progress", {"message": f"跳过重复调用 {name}（同批次内相同参数）"})
            continue

        # Guard 2: total-calls
        total_calls = len(accumulated) + len(new_tool_results)
        if total_calls >= MAX_TOTAL_TOOL_CALLS:
            logger.info("ReAct: skipping %s (total %d >= %d)", name, total_calls, MAX_TOTAL_TOOL_CALLS)
            _push_event(cfg, "progress", {"message": f"已收集 {total_calls} 条数据，跳过剩余工具调用"})
            continue

        # Guard 3: same-tool
        same_tool_count = (
            sum(1 for tr in accumulated if tr.get("name") == name)
            + sum(1 for tr in new_tool_results if tr.get("name") == name)
        )
        if same_tool_count >= MAX_SAME_TOOL_CALLS:
            logger.info("ReAct: skipping %s (%d >= %d)", name, same_tool_count, MAX_SAME_TOOL_CALLS)
            _push_event(cfg, "progress", {"message": f"已调用 {name} {same_tool_count} 次，跳过重复调用"})
            continue

        # Guard 4: args-dedup across accumulated + new
        args_key = json.dumps(arguments, sort_keys=True)
        dedup_match = False
        for prev in accumulated + new_tool_results:
            prev_name = prev.get("name", "")
            prev_args = prev.get("arguments", {})
            prev_args_key = json.dumps(prev_args, sort_keys=True)
            if prev_name == name and prev_args_key == args_key:
                dedup_match = True
                break
        if dedup_match:
            logger.info("ReAct: dedup skip %s (same args key=%s...)", name, args_key[:80])
            continue
        else:
            acc_keys = [json.dumps(p.get("arguments", {}), sort_keys=True)[:60] for p in accumulated]
            logger.info("ReAct: dedup no-match for %s args_key=%s accumulated_keys=%s",
                         name, args_key[:80], acc_keys)

        _called_in_batch.add(call_key)

        executed += 1
        started = time.perf_counter()

        if push_events:
            _push_event(cfg, "tool_call", {"name": name, "args": arguments})

        try:
            tool_result = agent_tools.call(name, arguments)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
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
            new_tool_results.append({
                "name": name, "arguments": arguments,
                "result": tool_result, "elapsed_ms": elapsed_ms,
            })
            new_messages.append({
                "role": "tool",
                "tool_call_id": tc_raw.get("id", ""),
                "content": json.dumps(tool_result, ensure_ascii=False),
            })
            if push_events:
                _push_event(cfg, "tool_result", {"name": name, "result": tool_result})
        except (FinanceToolError, TypeError) as err:
            failed += 1
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            _trace(cfg, {
                "session_id": session_id,
                "event_type": "tool_call",
                "node_name": node_name,
                "agent_id": agent_id,
                "ok": False, "tool_name": name, "arguments": arguments,
                "error": str(err), "elapsed_ms": elapsed_ms, "ts": _now_ts(),
                "parent_span_id": span_id,
            })
            new_tool_results.append({
                "name": name, "arguments": arguments,
                "error": str(err), "elapsed_ms": elapsed_ms,
            })
            new_messages.append({
                "role": "tool",
                "tool_call_id": tc_raw.get("id", ""),
                "content": json.dumps({"error": str(err)}, ensure_ascii=False),
            })
            if push_events:
                _push_event(cfg, "tool_result", {"name": name, "error": str(err)[:200]})

    # ---- Failure / progress tracking ----
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



def react_tool_node(state: AgentState, *, config: dict[str, Any]) -> dict[str, Any]:
    """Execute tool calls from the latest assistant message.

    Delegates to _react_execute_tool_calls (shared with subgraph path).
    Adds force_summarize hint for the evaluate node when all calls are deduped.
    """
    cfg = _get_configurable(config)
    agent_registry: AgentRegistry = cfg["agent_registry"]
    full_tools: ToolRegistry = cfg["full_tools"]

    react = state.get("react", {})
    if not react:
        return {}

    agent_id = react.get("agent_id", "")
    agent_tools = agent_registry.build_tool_registry(agent_id, full_tools)

    messages = react.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1]
    tool_calls_raw = last_msg.get("tool_calls", [])
    if not tool_calls_raw:
        return {}

    accumulated = list(state.get("tool_results", []))
    logger.info("ReAct: react_tool_node batch_size=%d accumulated_before=%d", len(tool_calls_raw), len(accumulated))

    with _trace_span(cfg, "react_tool", session_id=state.get("session_id", ""),
                     agent_id=agent_id, iteration=react.get("iteration", 0)) as span_id:
        result = _react_execute_tool_calls(
            tool_calls_raw=tool_calls_raw,
            agent_tools=agent_tools,
            messages=messages,
            accumulated=accumulated,
            agent_id=agent_id,
            session_id=state.get("session_id", ""),
            cfg=cfg,
            node_name="react_tool",
            consecutive_failures=react.get("consecutive_failures", 0),
            consecutive_no_progress=react.get("consecutive_no_progress", 0),
            prev_result_count=react.get("prev_result_count", 0),
            push_events=False,
            span_id=span_id,
        )

    new_messages = result["messages"]
    if result["force_summarize"]:
        new_messages.append({
            "role": "user",
            "content": "所有工具调用均为重复，无需再调用工具。请直接基于以上已有的工具返回结果回答用户的问题。",
        })

    return {
        "react": {
            **react,
            "messages": new_messages,
            "consecutive_failures": result["consecutive_failures"],
            "consecutive_no_progress": result["consecutive_no_progress"],
            "prev_result_count": result["prev_result_count"],
            "force_summarize": result["force_summarize"],
        },
        "tool_results": result["new_tool_results"],
        "tool_call_count": len(result["new_tool_results"]),
    }



def react_evaluate_node(state: AgentState, *, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate whether the current results are sufficient to answer.

    Checks early stop conditions and runs the evaluator LLM.  If sufficient,
    returns the final answer.  If insufficient, prompts the LLM to improve.
    """
    cfg = _get_configurable(config)
    llm = cfg["llm"]

    react = state.get("react", {})
    if not react:
        return {}

    # If answer is already set (e.g. error from react_llm_node), route to aggregate
    if react.get("answer"):
        return _build_react_final_answer(react, state.get("tool_results", []), llm, react.get("iteration", 0))

    tool_results = state.get("tool_results", [])
    iteration = react.get("iteration", 0) + 1
    question = react.get("question", "")
    messages = react.get("messages", [])
    consecutive_failures = react.get("consecutive_failures", 0)
    consecutive_no_progress = react.get("consecutive_no_progress", 0)

    with _trace_span(cfg, "react_evaluate", session_id=state.get("session_id", ""),
                     agent_id=react.get("agent_id", ""), iteration=iteration):
        # Early stop check
        early_reason = _check_early_stop(
            tool_results, iteration, consecutive_failures, consecutive_no_progress,
        )
        if early_reason:
            logger.info("ReAct early stop at iteration %d: %s", iteration, early_reason)
            return _build_react_final_answer(react, tool_results, llm, iteration)

        # Get the latest LLM response
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_content = msg["content"]
                break

        if not last_content:
            if react.get("force_summarize"):
                return _build_react_final_answer(react, tool_results, llm, iteration)
            return {"react": {**react, "iteration": iteration}}

        # Refusal check
        if _is_refusal(last_content) and tool_results:
            new_messages = list(messages) + [{
                "role": "user",
                "content": "你刚才说无法提供数据，但实际上工具已经返回了结果。请基于以上工具返回的真实数据回答用户的问题。",
            }]
            return {
                "react": {
                    **react,
                    "messages": new_messages,
                    "iteration": iteration,
                    "consecutive_no_progress": consecutive_no_progress + 1,
                },
            }

        # Periodic evaluator
        if tool_results and iteration > 0 and iteration % EVALUATOR_INTERVAL == 0:
            sufficient, reason = _evaluate_sufficiency(question, tool_results, last_content, llm)
            if sufficient:
                react["iteration"] = iteration
                return _build_react_final_answer(react, tool_results, llm, iteration)

        # Sufficient — build final answer
        return _build_react_final_answer(react, tool_results, llm, iteration)


