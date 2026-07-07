"""LangGraph orchestration nodes."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langgraph.types import RunnableConfig

from ..llm import LLMError, FunctionCallResult
from ..tools import FinanceToolError, ToolRegistry
from .state import AgentState

# ---- Prompts ----

CLASSIFY_PROMPT = """You are a routing classifier. Based on the user's message, classify the intent.

Return only a JSON object with one of these intents:
- {{"intent": "skill", "skill_name": "anomaly-diagnosis"}} — ONLY when the user explicitly asks for anomaly diagnosis, change detection, or root cause analysis of portfolio changes
- {{"intent": "skill", "skill_name": "portfolio-review"}} — ONLY when the user asks for a portfolio review, performance summary, or monthly/quarterly recap
- {{"intent": "skill", "skill_name": "allocation-check"}} — ONLY when the user asks about allocation deviation, rebalancing, or target vs actual comparison
- {{"intent": "react"}} — for simple questions that can be answered with one or two tool calls (e.g., holdings count, balance check, asset lookup)
- {{"intent": "plan_execute"}} — for complex multi-step analysis that requires planning multiple tool calls

Available skills:
{skills}

IMPORTANT: Only use "skill" intent when the user's request CLEARLY matches a skill's purpose. Default to "react" for simple questions and "plan_execute" for complex analysis.
"""

REACT_SYSTEM = """You are an investment analyst agent. Use the available tools to answer the user's question.

When you have enough information to answer, respond directly in the same language as the user.
Use Markdown formatting: **bold** for key figures, `code` for asset codes, tables for data, bullet lists for breakdowns.
Money is CNY unless stated otherwise. Never fabricate data; if tool data is missing, say so."""

PLAN_SYSTEM = """You are an investment analyst agent operating in Plan-Execute mode.

Available tools: {tools}

The user needs a multi-step analysis. Generate an execution plan as a JSON array of steps.
Each step: {{"step": 1, "tool": "tool_name", "arguments": {{}}, "purpose": "why this step"}}

Return ONLY the JSON array, no other text.
"""

REFLECTION_PROMPT = """You are a quality reviewer. Check if the answer below is accurate and complete.

User question: {question}
Answer to review: {answer}

Check:
1. Does the answer directly address the question?
2. Are all claims supported by the data (no fabrication)?
3. Is the answer complete (no missing key info)?

Return ONLY a JSON object:
{{"ok": true}} — if the answer is good
{{"ok": false, "issues": ["issue 1", "issue 2"]}} — if there are problems

Do NOT rewrite the answer. Just evaluate."""

# ---- Limits ----

MAX_REACT_ITERATIONS = 5
MAX_PLAN_STEPS = 5


# ---- Helpers ----

def _extract_json(text: str) -> Any:
    """Extract a JSON object or array from text, handling markdown fences."""
    cleaned = text.strip()
    fence = re.search(
        r"```(?:json)?\s*([\[{].*?[\]}])\s*```", cleaned, flags=re.DOTALL
    )
    if fence:
        cleaned = fence.group(1)
    elif not (cleaned.startswith("{") or cleaned.startswith("[")):
        brace = cleaned.find("{")
        bracket = cleaned.find("[")
        candidates = [i for i in (brace, bracket) if i >= 0]
        start = min(candidates) if candidates else -1
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _get_configurable(config: RunnableConfig) -> dict[str, Any]:
    return config.get("configurable", {})


def _build_system_prompt(base: str, cfg: dict[str, Any]) -> str:
    role = cfg.get("role")
    if role is not None:
        return role.to_system_prompt() + "\n" + base
    return base


def _trace(cfg: dict[str, Any], event: dict[str, Any]) -> None:
    sink = cfg.get("trace")
    if sink is not None:
        sink.record(event)


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _build_tools_for_llm(tools: ToolRegistry) -> list[dict[str, Any]]:
    """Build tool definitions list for LLM function calling."""
    return tools.list_tools()


def _fallback_plan(tools: ToolRegistry) -> list[dict[str, Any]]:
    """Build a minimal fallback plan using available tools."""
    names = tools.tool_names()
    plan = []
    step = 1
    for name in ("finance.holdings_summary", "finance.bucket_allocation"):
        if name in names:
            plan.append({"step": step, "tool": name, "arguments": {}, "purpose": "分析"})
            step += 1
    if not plan and names:
        plan.append({"step": 1, "tool": names[0], "arguments": {}, "purpose": "分析"})
    return plan


# ---- Nodes ----


def classify_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Classify the user's intent: skill, react, or plan_execute."""
    # Respect pre-set intent (e.g. from tests or re-entry)
    if state.get("intent"):
        return {}
    cfg = _get_configurable(config)
    llm = cfg.get("pipeline_llm", cfg["llm"])
    skills = cfg.get("skills", [])
    user_msg = state["user_message"]
    if not user_msg.strip():
        return {"intent": "react"}

    # Dynamic skill matching (keyword-based, fast path — zero LLM cost)
    for skill in skills:
        if hasattr(skill, "matches") and skill.matches(user_msg):
            return {"intent": "skill", "skill_name": skill.name}

    # LLM-based classification
    skills_desc = ""
    if skills:
        lines = []
        for s in skills:
            name = getattr(s, "name", "?")
            desc = getattr(s, "description", "")
            lines.append(f"- {name}: {desc}")
        skills_desc = "\n".join(lines)

    try:
        response = llm.complete(
            CLASSIFY_PROMPT.format(skills=skills_desc or "No skills available."),
            [{"role": "user", "content": user_msg}],
        )
        data = _extract_json(response)
        if not isinstance(data, dict):
            return {"intent": "react"}
        return {
            "intent": data.get("intent", "react"),
            "skill_name": data.get("skill_name", ""),
        }
    except LLMError:
        return {"intent": "react", "skill_name": "", "error": "LLM classification failed"}
    except Exception:
        return {"intent": "react", "skill_name": "", "error": "Classification error"}


def react_node(
    state: AgentState,
    *,
    config: RunnableConfig,
) -> dict[str, Any]:
    """ReAct loop using native function calling with regex fallback."""
    cfg = _get_configurable(config)
    llm = cfg["llm"]
    tools: ToolRegistry = cfg["tools"]

    iteration = state.get("react_iteration", 0)
    user_msg = state["user_message"]
    tool_results = list(state.get("tool_results", []))

    if iteration >= MAX_REACT_ITERATIONS:
        return {
            "final_answer": "已达到最大分析步数，请基于已有数据回答。",
            "react_iteration": iteration + 1,
        }

    system_prompt = _build_system_prompt(REACT_SYSTEM, cfg)

    # Build context with previous results
    context = f"User question: {user_msg}\n\n"
    if tool_results:
        context += (
            "Previous tool results:\n"
            + json.dumps(tool_results, ensure_ascii=False, indent=2)
            + "\n\n"
        )
        # Check for duplicate calls and warn
        seen = set()
        dupes = set()
        for tr in tool_results:
            key = (tr.get("name"), json.dumps(tr.get("arguments", {}), sort_keys=True))
            if key in seen:
                dupes.add(key[0])
            seen.add(key)
        if dupes:
            context += (
                "WARNING: The following tools were already called with the same arguments: "
                + ", ".join(dupes)
                + ". Do NOT call them again. Use the existing results to answer the question.\n\n"
            )
    context += "What should be the next action? Answer directly if you have enough data."

    # === Primary path: native function calling ===
    try:
        llm_tools = _build_tools_for_llm(tools)
        result: FunctionCallResult = llm.function_call(
            system_prompt, [{"role": "user", "content": context}], llm_tools
        )

        # LLM returned tool calls
        if result.tool_calls:
            return _execute_tool_calls(result.tool_calls, tool_results, state, tools, cfg, "react")

        # LLM returned direct answer (no tool calls)
        if result.content:
            return {
                "final_answer": result.content.strip(),
                "react_iteration": iteration + 1,
            }

        # Empty response — fall through to regex fallback
    except (LLMError, ConnectionError, TimeoutError, ValueError, OSError) as io_err:
        import logging
        logging.getLogger("matrix").warning(
            "Function calling failed, falling back to regex: %s", io_err
        )
    except Exception:
        pass  # Unexpected error — fall through to regex fallback

    # === Fallback: regex-based ReAct parsing ===
    return _react_fallback(state, config)


def _execute_tool_calls(
    tool_calls: list,
    tool_results: list[dict],
    state: AgentState,
    tools: ToolRegistry,
    cfg: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    """Execute a batch of tool calls from LLM response.

    Returns updated state with only actually-executed calls counted.
    Deduplicated and unknown tool calls are skipped and not counted.
    """
    executed = 0
    for tc in tool_calls:
        name = getattr(tc, "name", "")
        args = getattr(tc, "arguments", {})

        if name not in tools.tool_names():
            continue

        # Deduplication
        args_key = json.dumps(args, sort_keys=True)
        dup = any(
            prev.get("name") == name
            and json.dumps(prev.get("arguments", {}), sort_keys=True) == args_key
            for prev in tool_results
        )
        if dup:
            continue

        executed += 1
        started = time.perf_counter()
        try:
            result = tools.call(name, args)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            _trace(cfg, {
                "ok": True,
                "node": source,
                "tool": name,
                "arguments": args,
                "elapsed_ms": elapsed_ms,
                "ts": _now_ts(),
            })
            tool_results.append({
                "name": name,
                "arguments": args,
                "result": result,
                "elapsed_ms": elapsed_ms,
            })
        except FinanceToolError as err:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            _trace(cfg, {
                "ok": False,
                "node": source,
                "tool": name,
                "arguments": args,
                "error": str(err),
                "elapsed_ms": elapsed_ms,
                "ts": _now_ts(),
            })
            tool_results.append({
                "name": name,
                "arguments": args,
                "error": str(err),
                "elapsed_ms": elapsed_ms,
            })

    return {
        "tool_results": tool_results,
        "react_iteration": state.get("react_iteration", 0) + 1,
        "tool_call_count": state.get("tool_call_count", 0) + executed,
    }


def _react_fallback(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Regex-based ReAct fallback for when function calling fails."""
    cfg = _get_configurable(config)
    llm = cfg["llm"]
    tools: ToolRegistry = cfg["tools"]

    iteration = state.get("react_iteration", 0)
    user_msg = state["user_message"]
    tool_results = list(state.get("tool_results", []))

    tools_list = json.dumps(tools.list_tools(), ensure_ascii=False)
    system_prompt = _build_system_prompt(
        """You are an investment analyst agent operating in ReAct mode.

Available tools: {tools}

You MUST follow this EXACT format in every response:

When you need to call a tool:
Thought: [reason about what to do next in one sentence]
Action: [tool_name exactly as listed]
Action Input: [valid JSON arguments for the tool]

When you have enough information to answer the user:
Thought: I have enough information to answer
Final Answer: [your answer to the user, using Markdown for formatting]

Strict rules:
- ONE action per turn, never return multiple actions
- Use only the listed tool names with valid JSON arguments
- Never fabricate data; if tool data is missing, say so
- Keep answers concise; money is CNY unless stated otherwise
- Reply in the same language as the user""".format(tools=tools_list),
        cfg,
    )

    context = f"User question: {user_msg}\n\n"
    if tool_results:
        context += (
            "Previous tool results:\n"
            + json.dumps(tool_results, ensure_ascii=False, indent=2)
            + "\n\n"
        )
    context += "What should be the next action?"

    try:
        response = llm.complete(system_prompt, [{"role": "user", "content": context}])
    except LLMError as err:
        return {
            "error": f"LLM error: {err}",
            "final_answer": "LLM 调用失败，请稍后重试。",
            "react_iteration": iteration + 1,
        }

    # Check for Final Answer
    final_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
    if final_match:
        return {
            "final_answer": final_match.group(1).strip(),
            "react_iteration": iteration + 1,
        }

    # Parse Action
    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", response)
    tool_name = ""
    if action_match:
        tool_name = action_match.group(1).strip()
    else:
        for tname in tools.tool_names():
            if tname in response:
                tool_name = tname
                break

    if not tool_name:
        return {
            "final_answer": response.strip(),
            "react_iteration": iteration + 1,
        }

    if tool_name not in tools.tool_names():
        return {
            "error": f"Unknown tool: {tool_name}",
            "final_answer": f"工具 {tool_name} 不可用。",
            "react_iteration": iteration + 1,
        }

    args_match = re.search(r"Action Input:\s*(.+?)(?:\n\S|\Z)", response, re.DOTALL)
    arguments: dict[str, Any] = {}
    if args_match:
        try:
            parsed = _extract_json(args_match.group(1).strip())
            if isinstance(parsed, dict):
                arguments = parsed
        except (json.JSONDecodeError, ValueError):
            arguments = {}

    started = time.perf_counter()

    # Deduplication — increment tool_call_count to prevent stall
    args_key = json.dumps(arguments, sort_keys=True)
    for prev in tool_results:
        if prev.get("name") == tool_name and json.dumps(prev.get("arguments", {}), sort_keys=True) == args_key:
            return {
                "tool_call_count": state.get("tool_call_count", 0) + 1,
                "react_iteration": iteration + 1,
                "error": f"Duplicate tool call skipped: {tool_name}",
            }

    try:
        result = tools.call(tool_name, arguments)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        _trace(cfg, {
            "ok": True, "node": "react_fallback", "tool": tool_name,
            "arguments": arguments, "elapsed_ms": elapsed_ms, "ts": _now_ts(),
        })
        tool_results.append({
            "name": tool_name, "arguments": arguments,
            "result": result, "elapsed_ms": elapsed_ms,
        })
    except FinanceToolError as err:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        _trace(cfg, {
            "ok": False, "node": "react_fallback", "tool": tool_name,
            "arguments": arguments, "error": str(err), "elapsed_ms": elapsed_ms, "ts": _now_ts(),
        })
        tool_results.append({
            "name": tool_name, "arguments": arguments,
            "error": str(err), "elapsed_ms": elapsed_ms,
        })

    return {
        "tool_results": tool_results,
        "react_iteration": iteration + 1,
        "tool_call_count": state.get("tool_call_count", 0) + 1,
    }


def plan_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Generate an execution plan for complex multi-step analysis."""
    cfg = _get_configurable(config)
    llm = cfg.get("pipeline_llm", cfg["llm"])
    tools: ToolRegistry = cfg["tools"]

    user_msg = state["user_message"]
    tools_list = json.dumps(tools.list_tools(), ensure_ascii=False)
    system_prompt = _build_system_prompt(
        PLAN_SYSTEM.format(tools=tools_list), cfg
    )

    try:
        response = llm.complete(
            system_prompt, [{"role": "user", "content": user_msg}]
        )
    except LLMError as err:
        return {
            "error": f"LLM error: {err}",
            "current_plan": _fallback_plan(tools),
        }

    try:
        plan = json.loads(response.strip())
        if not isinstance(plan, list):
            data = _extract_json(response)
            plan = data if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError):
        plan = []

    if not plan:
        plan = _fallback_plan(tools)

    return {"current_plan": plan[:MAX_PLAN_STEPS]}


def execute_node(
    state: AgentState,
    *,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Execute one step of the plan."""
    cfg = _get_configurable(config)
    tools: ToolRegistry = cfg["tools"]

    plan = state.get("current_plan", [])
    tool_results = list(state.get("tool_results", []))
    tool_call_count = state.get("tool_call_count", 0)

    if tool_call_count >= len(plan):
        return {"final_answer": "计划执行完毕。"}

    step = plan[tool_call_count]
    started = time.perf_counter()
    try:
        result = tools.call(step["tool"], step.get("arguments", {}))
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        _trace(cfg, {
            "ok": True, "node": "execute", "tool": step["tool"],
            "arguments": step.get("arguments", {}), "elapsed_ms": elapsed_ms, "ts": _now_ts(),
        })
        tool_results.append({
            "name": step["tool"], "arguments": step.get("arguments", {}),
            "result": result, "elapsed_ms": elapsed_ms,
        })
    except FinanceToolError as err:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        _trace(cfg, {
            "ok": False, "node": "execute", "tool": step["tool"],
            "arguments": step.get("arguments", {}), "error": str(err),
            "elapsed_ms": elapsed_ms, "ts": _now_ts(),
        })
        tool_results.append({
            "name": step["tool"], "arguments": step.get("arguments", {}),
            "error": str(err), "elapsed_ms": elapsed_ms,
        })

    return {
        "tool_results": tool_results,
        "tool_call_count": tool_call_count + 1,
    }


def summarize_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Prepare summarization context. LLM call is deferred to chat.py for streaming."""
    user_msg = state["user_message"]
    tool_results = state.get("tool_results", [])

    existing = state.get("final_answer", "")
    if existing and not existing.startswith("技能「"):
        return {"final_answer": existing}

    if not tool_results:
        return {"final_answer": "未获取到任何数据，请检查工具调用。"}

    return {"needs_summary": True}


def skill_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Execute a predefined skill workflow and return results.

    Injects knowledge files into the context for the summarization step.
    """
    cfg = _get_configurable(config)
    tools: ToolRegistry = cfg["tools"]
    skills = cfg.get("skills", [])
    trace_sink = cfg.get("trace")

    skill_name = state.get("skill_name", "")
    skill = None
    for s in skills:
        if getattr(s, "name", "") == skill_name:
            skill = s
            break

    if skill is None:
        return {
            "error": f"Skill not found: {skill_name}",
            "final_answer": f"技能 {skill_name} 未找到。",
        }

    # Build knowledge context from skill's knowledge files
    # read_knowledge expects <skills_dir>/<skill_name>/knowledge/
    knowledge_context = ""
    if hasattr(skill, "knowledge_files") and skill.knowledge_files:
        skills_dir = cfg.get("skills_dir")
        if skills_dir:
            try:
                entries = skill.read_knowledge(skills_dir / skill_name)
                if entries:
                    parts = ["\n\n## 领域知识\n"]
                    for entry in entries:
                        parts.append(f"### {entry['name']}\n{entry['content']}\n")
                    knowledge_context = "".join(parts)
            except (OSError, FileNotFoundError) as err:
                import logging
                logging.getLogger("matrix").warning(
                    "Failed to read knowledge for skill %s: %s", skill_name, err
                )

    # Execute skill workflow
    from ..skills.executor import execute_skill

    result = execute_skill(skill, tools, trace_sink)
    findings: list[str] = result.get("findings", [])

    if result.get("errors"):
        error_msg = "; ".join(result["errors"])
        return {
            "error": error_msg,
            "tool_results": result.get("results", []),
            "tool_call_count": result.get("steps_executed", 0),
            "findings": findings,
            "final_answer": (
                f"技能「{skill.title}」执行完成："
                f"{result['steps_executed']} 步成功，"
                f"{len(result['errors'])} 个错误。"
            ),
        }

    # Attach knowledge context to tool_results for summarization
    enriched_results = list(result.get("results", []))
    if knowledge_context:
        enriched_results.append({
            "name": "_knowledge",
            "result": knowledge_context,
        })

    return {
        "tool_results": enriched_results,
        "tool_call_count": result.get("steps_executed", 0),
        "findings": findings,
    }


def reflection_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Lightweight self-check: verify the answer addresses the question."""
    cfg = _get_configurable(config)
    llm = cfg.get("pipeline_llm", cfg["llm"])

    answer = state.get("final_answer", "")
    user_msg = state.get("user_message", "")

    if not answer or not user_msg:
        return {}

    # Skip reflection for very short answers (< ~5 Chinese chars)
    if len(answer) < 15:
        return {}

    try:
        response = llm.complete(
            REFLECTION_PROMPT.format(question=user_msg, answer=answer),
            [{"role": "user", "content": "Evaluate the answer."}],
        )
        data = _extract_json(response)
        if isinstance(data, dict) and data.get("ok") is False:
            issues = data.get("issues", [])
            if issues:
                corrected = (
                    answer
                    + "\n\n---\n> ⚠️ 自检发现问题："
                    + "；".join(issues)
                )
                return {"final_answer": corrected}
    except (LLMError, json.JSONDecodeError, ValueError):
        pass

    return {}