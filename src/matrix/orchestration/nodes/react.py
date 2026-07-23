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

from langgraph.types import RunnableConfig

from ...llm import LLMError, LLMClient, FunctionCallResult
from ...tools import FinanceToolError, ToolRegistry
from ...agent.registry import AgentRegistry
from ...context import ToolResultRefStore
from ...context.compaction import compact_messages

from ._helpers import (
    DOMAIN_AGENT_REACT_SYSTEM,
    _build_history_context,
    _build_react_final_answer,
    _build_tools_for_llm,
    _check_early_stop,
    _classify_query_factuality,
    _evaluate_sufficiency,
    _get_configurable,
    _inject_data_index,
    _inject_working_memory,
    _is_refusal,
    _now_ts,
    _prune_tools,
    _push_event,
    _run_budget_and_compact,
    _trace,
    _trace_span,
    MAX_SAME_TOOL_CALLS,
    MAX_TOTAL_TOOL_CALLS,
    MAX_TOPLEVEL_REACT_ITERATIONS,
    EVALUATOR_INTERVAL,
)
from ..state import AgentState

logger = logging.getLogger("matrix.orchestration")

# ── ReAct nodes ──────────────────────────────────────────────────────────────

def react_prepare_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Prepare the ReAct context for a single-step domain agent execution."""
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

    # Build initial message with attachments if present
    attachments = cfg.get("attachments", [])
    if attachments:
        content_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": task_content},
        ]
        for att in attachments:
            if att.get("type") == "image":
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{att['mime_type']};base64,{att['base64']}"},
                })
        react_messages: list[dict[str, Any]] = [{"role": "user", "content": content_blocks}]
    else:
        react_messages = [{"role": "user", "content": task_content}]

    system_prompt = DOMAIN_AGENT_REACT_SYSTEM.format(
        agent_name=agent_def.name,
        persona=agent_def.persona,
        task=task,
        today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )

    # Pinned working memory + DataBus index
    system_prompt = _inject_working_memory(
        system_prompt, state.get("working_memory", {}), state.get("messages", []),
    )
    system_prompt = _inject_data_index(system_prompt, cfg.get("ref_store"), state.get("messages", []))

    react = {
        "messages": react_messages,
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


def react_llm_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Call the LLM with function calling to decide the next action."""
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

    # P2-2: Action Space pruning
    llm_tools = _prune_tools(llm_tools, messages, iteration=iteration)

    with _trace_span(cfg, "react_llm", session_id=state.get("session_id", ""),
                     agent_id=agent_id, iteration=iteration):
        try:
            # P2-3: Budget pre-check + compaction (98% threshold, free model)
            pipeline_llm = cfg.get("pipeline_llm")
            wm = state.get("working_memory", {})
            user_goal = wm.get("pinned", question)
            messages, rejected = _run_budget_and_compact(
                messages, system_prompt, pipeline_llm, user_goal,
            )
            if rejected:
                _trace(cfg, {"event_type": "budget_exceeded", "session_id": state.get("session_id", "")})
                return {
                    "messages": messages + [{
                        "role": "assistant",
                        "content": "[PROMPT_BUDGET_EXCEEDED] 上下文过长，请精简问题后重试",
                    }],
                    "react": {"iteration": react.get("iteration", 0) + 1, "done": True},
                    "agent_results": [{"agent": agent_id, "result": "BUDGET_EXCEEDED"}],
                    "tool_call_count": 0,
                    "tool_results": [],
                }
            react["messages"] = messages

            temp = _classify_query_factuality(question)
            result: FunctionCallResult = llm.function_call(system_prompt, messages, llm_tools, temperature=temp)
        except (LLMError, ConnectionError, TimeoutError, ValueError, OSError) as e:
            msg_len = sum(len(str(m.get("content", ""))) for m in messages)
            logger.error("ReAct: LLM call failed in react_llm_node: %s: %s (msg_count=%d, total_chars=%d)",
                         type(e).__name__, str(e)[:200], len(messages), msg_len)
            _push_event(cfg, "progress", {"message": f"LLM 调用失败: {type(e).__name__}"})
            return {
                "react": {**react, "answer": f"无法完成任务，LLM 调用失败: {type(e).__name__}。请重试。",
                          "error": f"LLM call failed: {type(e).__name__}"},
                "error": f"LLM call failed: {type(e).__name__}",
            }

    _push_thinking(result, agent_tools, cfg)

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


def _push_thinking(result: FunctionCallResult, agent_tools: ToolRegistry, cfg: dict[str, Any]) -> None:
    """Push a thinking event based on the LLM response."""
    if result.content:
        _push_event(cfg, "thinking", {"content": result.content.strip()})
    elif result.tool_calls:
        tool_names = [tc.name for tc in result.tool_calls if tc.name in agent_tools.tool_names()]
        if tool_names:
            _push_event(cfg, "thinking", {"content": f"正在调用 {'、'.join(tool_names)} 获取数据..."})


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

    Shared by both ReAct paths:
    - Top-level: react_tool_node (push_events=False, SSE via state emission)
    - Subgraph:   _run_domain_agent_react (push_events=True, direct SSE)

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
        tool_result = agent_tools.call(name, arguments)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

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

    except (FinanceToolError, TypeError) as err:
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
        if push_events:
            _push_event(cfg, "tool_result", {"name": name, "error": str(err)[:200]})
        return False, {"name": name, "arguments": arguments, "error": str(err), "elapsed_ms": elapsed_ms}


# ── ReAct routing ────────────────────────────────────────────────────────────

def _route_after_react_llm(state: AgentState) -> str:
    """Route after react_llm: tool calls → react_tool, otherwise → react_evaluate."""
    react = state.get("react", {})
    if not react:
        return "react_evaluate"

    messages = react.get("messages", [])
    if not messages:
        return "react_evaluate"

    last = messages[-1]
    if last.get("role") == "assistant" and last.get("tool_calls"):
        return "react_tool"
    return "react_evaluate"


def _route_after_react_evaluate(state: AgentState) -> str:
    """Route after react_evaluate: not done → react_llm, done → aggregate."""
    react = state.get("react", {})
    if not react:
        return "aggregate"

    answer = react.get("answer", "")
    if answer:
        return "aggregate"

    iteration = react.get("iteration", 0)
    if iteration >= MAX_TOPLEVEL_REACT_ITERATIONS:
        return "aggregate"

    return "react_llm"


def react_tool_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Execute tool calls from the last LLM response.

    Uses the shared _react_execute_tool_calls for guard logic,
    but emits results via state (not push_events) for LangGraph
    top-level streaming.
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

    last = messages[-1]
    if last.get("role") != "assistant" or not last.get("tool_calls"):
        return {}

    accumulated = list(state.get("tool_results", []))

    with _trace_span(cfg, "react_tool", session_id=state.get("session_id", ""),
                     agent_id=agent_id, iteration=react.get("iteration", 0)) as span_id:
        exec_result = _react_execute_tool_calls(
            tool_calls_raw=last["tool_calls"],
            agent_tools=agent_tools,
            messages=messages,
            accumulated=accumulated,
            agent_id=agent_id,
            session_id=state.get("session_id", ""),
            cfg=cfg,
            node_name="react_tool_node",
            consecutive_failures=react.get("consecutive_failures", 0),
            consecutive_no_progress=react.get("consecutive_no_progress", 0),
            prev_result_count=react.get("prev_result_count", 0),
            push_events=False,
            span_id=span_id,
            ref_store=cfg.get("ref_store"),
        )

    new_messages = exec_result["messages"]
    tool_results = list(accumulated) + exec_result["new_tool_results"]

    return {
        "react": {
            **react,
            "messages": new_messages,
            "consecutive_failures": exec_result["consecutive_failures"],
            "consecutive_no_progress": exec_result["consecutive_no_progress"],
            "prev_result_count": exec_result["prev_result_count"],
            "force_summarize": exec_result["force_summarize"],
        },
        "tool_results": exec_result["new_tool_results"],
        "tool_call_count": exec_result["executed"],
    }


def react_evaluate_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Evaluate ReAct sufficiency and decide whether to continue or finish.

    Checks: early stop, evaluator, refusal, iteration limit.
    If done, builds the final answer via _build_react_final_answer.
    """
    cfg = _get_configurable(config)
    llm = cfg["llm"]

    react = state.get("react", {})
    if not react:
        return {}

    iteration = react.get("iteration", 0) + 1
    question = react.get("question", "")
    messages = react.get("messages", [])
    system_prompt = react.get("system", "")
    tool_results = list(state.get("tool_results", []))
    consecutive_failures = react.get("consecutive_failures", 0)
    consecutive_no_progress = react.get("consecutive_no_progress", 0)

    # Check early stop
    early_reason = _check_early_stop(
        tool_results, iteration, consecutive_failures, consecutive_no_progress,
    )
    if early_reason:
        logger.info("ReAct early stop at iter %d: %s", iteration, early_reason)
        return _build_react_final_answer(react, tool_results, llm, iteration)

    # Check force_summarize — all tool calls were deduped, prompt LLM to summarize
    if react.get("force_summarize"):
        if not tool_results:
            return _build_react_final_answer(react, tool_results, llm, iteration)
        new_messages = list(messages) + [{
            "role": "user",
            "content": "所有工具调用均为重复。请基于已有结果为用户总结，不要继续调用工具。",
        }]
        logger.info("ReAct force_summarize at iter %d: prompting LLM to summarize", iteration)
        return {
            "react": {**react, "messages": new_messages, "iteration": iteration,
                      "force_summarize": False},
            "tool_results": [],
            "tool_call_count": 0,
        }

    # Check refusal
    last_msg = messages[-1] if messages else {}
    if last_msg.get("role") == "assistant":
        content = last_msg.get("content", "")
        if _is_refusal(content) and tool_results:
            new_messages = list(messages) + [{
                "role": "user",
                "content": "你刚才说无法提供数据，但实际上工具已经返回了结果。请基于以上工具返回的真实数据回答用户的问题。直接给出数据，不要说你无法提供。",
            }]
            return {
                "react": {**react, "messages": new_messages, "iteration": iteration,
                          "consecutive_no_progress": consecutive_no_progress + 1},
                "tool_results": [],
                "tool_call_count": 0,
            }

    # Periodic evaluator
    if iteration > 0 and iteration % EVALUATOR_INTERVAL == 0 and tool_results:
        sufficient, reason = _evaluate_sufficiency(question, tool_results, "", llm)
        if sufficient:
            logger.info("ReAct: evaluator sufficient at iter %d: %s", iteration, reason)
            return _build_react_final_answer(react, tool_results, llm, iteration)

    # Iteration limit
    if iteration >= MAX_TOPLEVEL_REACT_ITERATIONS:
        return _build_react_final_answer(react, tool_results, llm, iteration)

    # Content without tool calls — we're done
    if last_msg.get("role") == "assistant" and not last_msg.get("tool_calls"):
        content = last_msg.get("content", "")
        if content:
            return _build_react_final_answer(react, tool_results, llm, iteration)

    return {
        "react": {**react, "iteration": iteration},
        "tool_results": [],
        "tool_call_count": 0,
    }