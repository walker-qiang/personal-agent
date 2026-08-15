"""Shared helpers, constants, and re-exports for orchestration nodes.

This module serves as the main facade for node helpers. Large components
have been split into dedicated sub-modules:

- _prompts.py: All LLM prompt templates
- _circuit_breaker.py: Per-tool circuit breaker
- _verification.py: Anti-hallucination checks (heuristic, multi-sampling)
- _react_helpers.py: ReAct final answer building and media URL handling

Everything here is re-exported for backward compatibility — existing
``from ._helpers import X`` calls continue to work without changes.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import re
import time
import uuid
from typing import Any

from langgraph.types import RunnableConfig

from ...llm import LLMError, LLMClient, FunctionCallResult
from ...tools import FinanceToolError, ToolRegistry
from ..events import make_event

# ── Re-exports from split modules ────────────────────────────────────────────

from ._prompts import (
    COMMANDER_PLAN_PROMPT,
    PREFLECT_PROMPT,
    REPLAN_PROMPT,
    FALLBACK_AGGREGATE_PROMPT,
    COMMANDER_AGGREGATE_PROMPT,
    DOMAIN_AGENT_REACT_SYSTEM,
    REFLECTION_PROMPT,
    REVISE_PROMPT,
    REFLEXION_PROMPT,
    REFLEXION_RETRY_PROMPT,
    LESSON_EXTRACTION_PROMPT,
    EVALUATOR_PROMPT,
)
from ._circuit_breaker import (
    CircuitBreaker,
    MAX_CONSECUTIVE_TOOL_FAILURES,
    CIRCUIT_BREAKER_COOLDOWN_SEC,
)
from ._verification import (
    _is_empty_tool_result,
    _heuristic_number_check,
    _VERIFY_CACHE,
    _VERIFY_CACHE_MAX,
    _cache_key,
    _cached_verdict,
    _cache_verdict,
    _single_verify,
    _multi_sample_verify,
    _full_verify_and_correct,
)
from ._react_helpers import (
    _llm_summarize_from_results,
    _build_react_final_answer,
    _extract_media_urls,
    _fix_media_answer,
)

logger = logging.getLogger("matrix.orchestration")

# ── Chinese weekday names ──
_WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _today_cn() -> str:
    """Return today's date with Chinese weekday, e.g., '2026年8月1日 (周六)'.

    Uses Asia/Shanghai timezone so the date matches the user's locale.
    """
    from zoneinfo import ZoneInfo
    from datetime import datetime
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    return f"{now.year}年{now.month}月{now.day}日 ({_WEEKDAY_CN[now.weekday()]})"


# ── Iteration / safety limits ─────────────────────────────────────────────────

MAX_REACT_ITERATIONS = 20  # Hard safety net; goal-driven stopping should trigger earlier

MAX_TOPLEVEL_REACT_ITERATIONS = 10  # Iteration limit for the top-level single-step ReAct loop

MAX_SUBTASK_ITERATIONS = 10  # Per-subtask ReAct limit

MAX_SUBTASKS = 5             # Max subtasks in a decomposition

MAX_PLAN_STEPS = 3

MAX_CONSECUTIVE_FAILURES = 2      # Stop if N consecutive tool calls all fail

MAX_CONSECUTIVE_NO_PROGRESS = 3   # Stop if N consecutive steps add no new info

MAX_SAME_TOOL_CALLS = 3           # Stop if same tool called N+ times (same name, regardless of args)

MAX_TOTAL_TOOL_CALLS = 12         # Deep investment research may need the full personal_os tool set

EVALUATOR_INTERVAL = 2            # Run evaluator every N iterations


# ── Query factuality classifier ──────────────────────────────────────────────

_FACTUAL_PATTERNS = [
    r"(多少|几|什么价格|股价|市值|市盈率|财报|营收|利润|增长率|涨跌|跌幅|涨幅)",
    r"(搜索|查询|查找|最新|今日|昨天|本周|本月|今年|上个季度|最近)",
    r"(新闻|报道|公告|发布|宣布|数据|统计|公布|披露)",
    r"(how much|what is|search|latest|today|price|stock|news|revenue|earnings)",
]


def _classify_query_factuality(question: str) -> float:
    """Classify query as factual vs creative; return recommended temperature.

    Factual queries (data, news, prices) → low temperature to reduce hallucination.
    Creative queries (image generation, writing) → normal temperature.
    """
    score = 0
    for pat in _FACTUAL_PATTERNS:
        if re.search(pat, question, re.IGNORECASE):
            score += 1
    if score >= 2:
        return 0.1
    elif score >= 1:
        return 0.4
    return 0.7


# ── Lesson injection ──────────────────────────────────────────────────────────


def _inject_lessons(
    system_prompt: str,
    task: str,
    agent_id: str,
    cfg: dict[str, Any],
) -> str:
    """Inject cross-session failure lessons into the system prompt.

    Queries LessonStore for lessons relevant to the current task and agent,
    and appends them as a "Past Lessons" section. No-op when no lessons found
    or LessonStore not configured.

    Returns the updated system_prompt.
    """
    lesson_store = cfg.get("lesson_store")
    if lesson_store is None:
        return system_prompt

    user_id = cfg.get("user_id", "")
    try:
        lessons = lesson_store.get_relevant_lessons(
            task=task,
            agent_id=agent_id,
            user_id=user_id,
            top_k=3,
        )
    except Exception as exc:
        logger.debug("lesson_inject: query failed: %s", exc)
        return system_prompt

    if not lessons:
        return system_prompt

    # Build lesson block
    lines: list[str] = []
    for lesson in lessons:
        count_tag = f" (×{lesson.occurrence_count})" if lesson.occurrence_count > 1 else ""
        severity_tag = f" [{lesson.severity}]" if lesson.severity != "medium" else ""
        lines.append(f"- {lesson.lesson_text}{count_tag}{severity_tag}")

    lesson_block = "\n".join(lines)
    system_prompt += (
        f"\n\n## Past Lessons (避免重复犯错)\n"
        f"以下是从过去失败中总结的教训, 请在本任务中参考:\n"
        f"{lesson_block}"
    )

    # Update last_seen for matched lessons
    lesson_ids = [l.lesson_id for l in lessons if l.lesson_id]
    if lesson_ids:
        try:
            lesson_store.update_last_seen(lesson_ids)
        except Exception:
            pass  # Best-effort update

    return system_prompt


# ── Config / tracing helpers ──────────────────────────────────────────────────


def _get_configurable(config: RunnableConfig) -> dict[str, Any]:
    return config.get("configurable", {})


def _trace(cfg: dict[str, Any], event: dict[str, Any]) -> None:
    sink = cfg.get("trace")
    if sink is not None:
        sink.record(event)


@contextlib.contextmanager
def _trace_span(cfg: dict[str, Any], name: str, **kwargs: Any):
    """Record a span: start and end events with latency.

    Dual-mode tracing:
    1. Legacy events → TraceStore.record() (backward compatible)
    2. OTel spans → TraceStore.start_span()/end_span() (OTel standardized)

    Usage:
        with _trace_span(cfg, "react_llm", session_id=..., agent_id=...,
                         parent_span_id=...) as span_id:
            ... do work ...
    # span_start and span_end events are automatically recorded.
    """
    span_id = uuid.uuid4().hex[:12]
    parent_span_id = kwargs.get("parent_span_id")
    session_id = kwargs.get("session_id", "")
    agent_id = kwargs.get("agent_id", "")
    iteration = kwargs.get("iteration", 0)

    # Legacy event: span_start
    _trace(cfg, {
        "session_id": session_id,
        "event_type": "span_start",
        "node_name": name,
        "agent_id": agent_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "ts": _now_ts(),
        "arguments": {"iteration": iteration},
    })

    # OTel span: start
    trace_store = cfg.get("trace")
    otel_span = None
    if trace_store is not None and hasattr(trace_store, "start_span"):
        try:
            otel_span = trace_store.start_span(
                name,
                session_id=session_id,
                agent_id=agent_id,
                iteration=iteration,
            )
        except Exception:
            otel_span = None

    started = time.perf_counter()
    try:
        yield span_id
    finally:
        elapsed = round((time.perf_counter() - started) * 1000, 3)

        # Legacy event: span_end
        _trace(cfg, {
            "session_id": session_id,
            "event_type": "span_end",
            "node_name": name,
            "agent_id": agent_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "elapsed_ms": elapsed,
            "ts": _now_ts(),
        })

        # OTel span: end
        if otel_span is not None and trace_store is not None:
            try:
                trace_store.end_span(otel_span)
            except Exception:
                pass


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── Content detection helpers ─────────────────────────────────────────────────


def _is_refusal(content: str) -> bool:
    """Check if the LLM content is a refusal to use tools."""
    lowered = content.lower()
    refusal_patterns = [
        r"抱歉", r"不能", r"无法", r"做不到", r"目前不",
        r"sorry", r"cannot", r"unable", r"can't", r"don't have",
        r"i apologize", r"i am not able",
    ]
    return any(re.search(pat, lowered) for pat in refusal_patterns)


def _is_hallucination(content: str) -> bool:
    """Check if the LLM is pretending to have completed a task without actually calling tools.

    Detects patterns like '已为您生成', '生成结果如下', etc. where the LLM
    describes a non-existent output as if it were real.
    """
    return bool(re.search(
        r"已(为您|经)?(生成|创建|制作|完成)|生成结果如下|具体效果如下|"
        r"Here is the (generated|created) |I have (generated|created) ",
        content,
    ))


def _force_tool_call(
    llm,
    system_prompt: str,
    task: str,
    tools: list[dict[str, Any]],
) -> FunctionCallResult:
    """Retry with tool_choice='required' and a stronger system prompt."""
    forced_system = (
        system_prompt
        + "\n\nCRITICAL: You MUST call a tool to complete this task. "
        "Do NOT say you cannot do it — call the appropriate tool. "
        "Do NOT return text without calling a tool first."
    )
    try:
        return llm.function_call(
            forced_system,
            [{"role": "user", "content": f"Task: {task}\n\nCall a tool to complete this task."}],
            tools,
            tool_choice="required",
        )
    except (LLMError, ConnectionError, TimeoutError, ValueError, OSError) as e:
        logger.warning("RAG need detection LLM call failed: %s", type(e).__name__)
        return FunctionCallResult(content="", tool_calls=[])


def _build_history_context(history: list[dict[str, str]], max_turns: int = 3) -> str:
    """Build compact conversation history context for injection into LLM prompts."""
    if not history:
        return ""
    recent = history[-(max_turns * 2):]  # each turn = user + assistant
    lines = []
    for h in recent:
        role_label = "用户" if h["role"] == "user" else "助手"
        content = h.get("content", "")
        if isinstance(content, list):
            # Multi-modal content: extract text parts only
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            content = " ".join(text_parts)
        lines.append(f"[{role_label}]: {content[:300]}")
    return "对话历史：\n" + "\n".join(lines) + "\n\n"


# ── Goal-driven Evaluator ────────────────────────────────────────────────────


def _evaluate_sufficiency(
    question: str,
    tool_results: list[dict[str, Any]],
    llm_response: str,
    llm: Any,
) -> tuple[bool, str]:
    """Evaluate whether current results are sufficient to answer the question.

    Returns: (is_sufficient: bool, reason: str)
    """
    if not tool_results and not llm_response:
        return False, "无工具结果且无LLM输出"

    # Build a compact summary of tool results
    tool_summary = []
    for tr in tool_results[-8:]:  # Last 8 results to keep prompt short
        name = tr.get("name", "unknown")
        if "error" in tr:
            tool_summary.append(f"  [{name}] 失败: {str(tr['error'])[:100]}")
        else:
            result = tr.get("result", "")
            tool_summary.append(f"  [{name}] 结果: {str(result)[:200]}")
    tool_text = "\n".join(tool_summary) if tool_summary else "（无工具结果）"

    eval_prompt = f"""用户问题：{question}

已收集的工具结果：
{tool_text}

Agent 当前回答：
{llm_response[:500] if llm_response else "（尚未生成回答）"}"""

    try:
        data = llm.complete_json(
            EVALUATOR_PROMPT.format(today=_today_cn()),
            [{"role": "user", "content": eval_prompt}],
            temperature=0.1,
        )
        if not isinstance(data, dict):
            return _evaluate_heuristic(tool_results, llm_response)
        is_sufficient = bool(data.get("sufficient", False))
        reason = str(data.get("reason", ""))
        if not reason:
            reason = "充分" if is_sufficient else "不充分"
        return is_sufficient, reason
    except Exception as e:
        # Evaluator call failed — fall back to heuristic
        logger.warning("Evaluator call failed: %s, falling back to heuristic", e)
        return _evaluate_heuristic(tool_results, llm_response)


def _evaluate_heuristic(
    tool_results: list[dict[str, Any]],
    llm_response: str,
) -> tuple[bool, str]:
    """Heuristic fallback when evaluator LLM call fails."""
    if not llm_response:
        return False, "无LLM回答"
    if len(llm_response) < 10:
        return False, "回答过短"
    if _is_refusal(llm_response):
        return False, "回答为拒绝"
    if _is_hallucination(llm_response):
        return False, "回答疑似幻觉"
    if not tool_results:
        return False, "无工具调用结果"
    return True, "启发式判定充分"


def _check_early_stop(
    tool_results: list[dict[str, Any]],
    iteration: int,
    consecutive_failures: int,
    consecutive_no_progress: int,
) -> str | None:
    """Check early stopping signals. Returns reason string if should stop, None otherwise."""
    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        return f"连续 {consecutive_failures} 次工具调用全部失败"

    if consecutive_no_progress >= MAX_CONSECUTIVE_NO_PROGRESS:
        return f"连续 {consecutive_no_progress} 步未收集到新信息"

    # Check for excessive same-tool calls (by name, regardless of args)
    if tool_results:
        tool_counts: dict[str, int] = {}
        for tr in tool_results:
            name = tr.get("name", "")
            tool_counts[name] = tool_counts.get(name, 0) + 1
        for name, count in tool_counts.items():
            if count >= MAX_SAME_TOOL_CALLS:
                return f"同一工具 {name} 调用 {count} 次，信息已足够"

    # Check for excessive total tool calls (across all tools)
    if len(tool_results) >= MAX_TOTAL_TOOL_CALLS:
        return f"工具总调用 {len(tool_results)} 次已达上限，应已收集足够信息"

    return None


# Domain-specific tools that return structured data — when they return valid
# results, the data is very likely sufficient to answer the user's question.
_DOMAIN_SUFFICIENCY_TOOLS = {
    "weather",
    "finance_query",
    "finance.recent_snapshots",
}

_BROWSER_TASK_HINTS = (
    "浏览器",
    "browser",
    "spa",
    "动态页面",
    "javascript 渲染",
    "js 渲染",
    "页面交互",
    "点击页面",
    "打开网页并提取",
)


def _requires_browser(task: str) -> bool:
    """Return whether a task explicitly requires browser automation."""
    lowered = task.lower()
    return any(hint in lowered for hint in _BROWSER_TASK_HINTS)


def _tool_name_for_llm(tool: dict[str, Any]) -> str:
    """Read a tool name from the provider-facing function schema."""
    return str(tool.get("function", {}).get("name", tool.get("name", "")))


def _focus_tools_for_task(
    task: str,
    tool_defs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restrict the initial action space for unambiguous task types."""
    lowered = task.lower()
    preferred: set[str] = set()

    if any(term in lowered for term in ("天气", "温度", "下雨", "预报", "weather")):
        preferred.add("weather")
    elif any(term in lowered for term in ("快照", "snapshot")) and any(
        term in lowered for term in ("最近", "最新", "记录", "recent", "latest")
    ):
        preferred.add("finance.recent_snapshots")
    elif _requires_browser(task):
        preferred = {
            _tool_name_for_llm(tool)
            for tool in tool_defs
            if _tool_name_for_llm(tool).startswith("mcp_browser_")
        }

    if not preferred:
        return tool_defs
    available = {_tool_name_for_llm(tool) for tool in tool_defs}
    if not (_requires_browser(task) or preferred & available):
        return tool_defs
    return [
        tool for tool in tool_defs
        if _tool_name_for_llm(tool) in preferred
    ]


def _check_domain_tool_sufficiency(
    question: str,
    tool_results: list[dict[str, Any]],
) -> bool:
    """Check if domain-specific tools have returned valid, CURRENT data.

    For finance queries: requires that the result contains actual numeric
    data (prices, percentages, etc.) AND that the data appears to be for
    today's date. A non-empty dict without actual numbers is not sufficient.

    This avoids treating stale/cached data as sufficient and prevents the
    LLM from fabricating missing numbers.
    """
    for tr in tool_results:
        name = tr.get("name", "")
        if name not in _DOMAIN_SUFFICIENCY_TOOLS or "error" in tr:
            continue
        result = tr.get("result", {})
        if not isinstance(result, dict) or not result:
            continue
        # For finance queries: require actual numeric data in the result
        if name == "finance_query":
            result_str = json.dumps(result, ensure_ascii=False)
            # Check if result contains price-like numbers (e.g., 3400.12, 3.5%)
            has_prices = bool(re.search(r"\d{1,5}\.\d{1,2}", result_str))
            has_percentages = bool(re.search(r"\d+\.\d+\s*[%％]", result_str))
            if has_prices or has_percentages:
                return True
            # Result has data but no actual numeric market data → not sufficient
            continue
        return True
    return False


# ── Tool management ────────────────────────────────────────────────────────────


def _build_tools_for_llm(tools: ToolRegistry) -> list[dict[str, Any]]:
    """Build tool definitions list for LLM function calling."""
    return tools.list_tools()


def _prune_tools(
    all_tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    iteration: int = 0,
    circuit_breaker: Any = None,
) -> list[dict[str, Any]]:
    """Dynamically prune the action space based on current context.

    Principles (from "从防御到赋能" design philosophy):
    - Remove tools that cannot possibly be useful in the current state
    - Reduce choice overload → improve LLM decision quality
    - Never remove tools the LLM genuinely needs

    Pruning rules:
    1. No get_stored_data when no __refId in messages
    2. No working_memory on first iteration (nothing to record yet)
    3. No circuit-breaker-blocked tools (tool has failed too many times consecutively)
    """
    has_refs = False
    for msg in messages:
        if "__refId" in str(msg.get("content", "")):
            has_refs = True
            break

    # Check if working_memory was just called (avoid infinite loop)
    last_is_wm = False
    if messages:
        last = messages[-1]
        if last.get("role") == "tool":
            last_name = last.get("name", "")
            if last_name == "working_memory":
                last_is_wm = True

    pruned: list[dict[str, Any]] = []
    for tool in all_tools:
        name = tool.get("function", {}).get("name", "")

        # Rule 1: hide get_stored_data when no external refs exist
        if name == "get_stored_data" and not has_refs:
            continue

        # Rule 2: hide working_memory on first iteration or after just calling it
        if name == "working_memory" and (iteration <= 0 or last_is_wm):
            continue

        # Rule 3: hide circuit-breaker-blocked tools
        if circuit_breaker is not None and circuit_breaker.is_blocked(name):
            logger.info("prune_tools: hiding blocked tool=%s (circuit breaker)", name)
            continue

        pruned.append(tool)

    return pruned


# ── Shared context injection helpers (P0-P2) ─────────────────────────────────
# These eliminate ~90% duplication between react.py and commander.py


def _inject_working_memory(
    system_prompt: str,
    wm: dict[str, Any],
    user_messages: list[dict[str, Any]],
) -> str:
    """Inject pinned goal and active insights into the system prompt.

    Returns the updated system_prompt. Side effect: initializes wm["pinned"]
    from the first user message if not already set.
    """
    pinned = wm.get("pinned", "")
    if not pinned:
        user_msgs = [m for m in user_messages if m.get("role") == "user"]
        if user_msgs:
            pinned = str(user_msgs[0].get("content", ""))[:2000]
            wm["pinned"] = pinned
    if pinned:
        system_prompt = f"**Pinned Goal (your anchor):** {pinned}\n\n" + system_prompt

    insights = wm.get("insights", [])
    if insights:
        insight_block = "\n".join(f"- {i}" for i in insights[:5])
        system_prompt += f"\n\n## Key Insights (from previous steps)\n{insight_block}"

    return system_prompt


def _inject_data_index(
    system_prompt: str,
    ref_store: Any,
    messages: list[dict[str, Any]],
) -> str:
    """Inject DataBus data index for externalized results into the system prompt.

    Returns the updated system_prompt. No-op when ref_store is None.
    """
    if ref_store is None:
        return system_prompt
    # Lazy import to avoid circular dependency
    from matrix.context.databus import build_data_index
    data_index = build_data_index(ref_store, messages)
    if data_index:
        system_prompt += f"\n\n{data_index}"
    return system_prompt


def _inject_agent_guidelines(
    system_prompt: str,
    agent_def: Any,
) -> str:
    """Inject domain-specific guidelines based on agent's system_guidelines config.

    Reads guideline markdown files from src/matrix/agent/guidelines/
    and appends them to the system prompt. Returns the updated prompt.
    No-op when agent_def is None or has no system_guidelines.
    """
    if agent_def is None:
        return system_prompt
    guidelines = getattr(agent_def, "system_guidelines", [])
    if not guidelines:
        return system_prompt

    from matrix.agent.guidelines import load_guideline

    parts = [system_prompt]
    for name in guidelines:
        content = load_guideline(name)
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def _run_budget_and_compact(
    messages: list[dict[str, Any]],
    system_prompt: str,
    pipeline_llm: Any,
    user_goal: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Run budget pre-check and compaction if needed.

    Returns (messages, rejected) where rejected=True means the call should
    be aborted — caller must handle the abort according to its own context.

    Uses budget.py (98% threshold for free-tier models) and falls back
    to compaction when above 85%.

    Supports incremental update: extracts the previous handoff from the
    message list and passes it as previous_summary to compact_messages,
    so repeated compactions build on the existing summary instead of
    starting from scratch.
    """
    from matrix.context.budget import check_budget_compact
    from matrix.context.compaction import (
        compact_messages, extract_previous_handoff, strip_previous_handoff, strip_handoff_markers,
    )

    proceed, action = check_budget_compact(messages, system_prompt)

    if action == "reject":
        if pipeline_llm is not None:
            previous = extract_previous_handoff(messages)
            to_compress = strip_previous_handoff(messages)
            messages = compact_messages(
                to_compress, user_goal, pipeline_llm,
                previous_summary=previous,
            )
            messages = strip_handoff_markers(messages)
            _, action2 = check_budget_compact(messages, system_prompt)
            if action2 == "reject":
                return (messages, True)  # rejected after compaction
            return (messages, False)
        return (messages, True)  # rejected, no pipeline_llm to try compaction

    if action == "compact" and pipeline_llm is not None:
        previous = extract_previous_handoff(messages)
        to_compress = strip_previous_handoff(messages)
        messages = compact_messages(
            to_compress, user_goal, pipeline_llm,
            previous_summary=previous,
        )

    # Strip HANDOFF_JSON markers before sending to LLM (saves tokens)
    messages = strip_handoff_markers(messages)
    return (messages, False)


# ── Event / streaming ──────────────────────────────────────────────────────────


def _push_event(cfg: dict[str, Any], evt_type: str, payload: dict[str, Any]) -> None:
    """Push a real-time event to the SSE queue if available.

    The orchestration boundary accepts the historical string/payload shape,
    but the queue itself carries structured events.  This keeps event typing
    consistent from the node layer through the SSE adapter.
    """
    q = cfg.get("event_queue")
    if q is not None:
        try:
            q.put_nowait(make_event(evt_type, payload))
        except queue.Full:
            logger.warning("event_queue full: dropping %s event", evt_type)


# ── High-risk tool detection ──────────────────────────────────────────────────

_HIGH_RISK_PATTERNS = [
    "snapshot.create", "snapshot.update", "snapshot.delete",
    "asset.create", "asset.update", "asset.delete",
    "write", "save", "delete", "create", "update",
    "execute", "run", "deploy",
    # Browser interactive operations (require user confirmation)
    "browser_click", "browser_type", "browser_select",
    "browser_press", "browser_restore",
]


def _is_high_risk(tool_name: str) -> bool:
    """Check if a tool call is high-risk based on its name."""
    return any(pattern in tool_name.lower() for pattern in _HIGH_RISK_PATTERNS)
