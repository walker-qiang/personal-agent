"""LangGraph orchestration nodes."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langgraph.types import RunnableConfig

from ..llm import LLMError
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

REACT_SYSTEM = """You are an investment analyst agent operating in ReAct mode.

Available tools: {tools}

You MUST follow this EXACT format in every response. Do NOT add any other text outside this format.

When you need to call a tool:
Thought: [reason about what to do next in one sentence]
Action: [tool_name exactly as listed]
Action Input: [valid JSON arguments for the tool]

When you have enough information to answer the user:
Thought: I have enough information to answer
Final Answer: [your answer to the user, using Markdown for formatting — use **bold** for key numbers, tables for comparisons, and bullet lists for breakdowns]

Strict rules:
- ONE action per turn, never return multiple actions
- Action MUST be exactly the tool name from the list above
- Use only the listed tool names with valid JSON arguments
- Never fabricate data; if tool data is missing, say so
- Keep answers concise; money is CNY unless stated otherwise
- Reply in the same language as the user
- Use Markdown formatting: **bold** for key figures, `code` for asset codes, tables for data, bullet lists for breakdowns
"""

PLAN_SYSTEM = """You are an investment analyst agent operating in Plan-Execute mode.

Available tools: {tools}

The user needs a multi-step analysis. Generate an execution plan as a JSON array of steps.
Each step: {{"step": 1, "tool": "tool_name", "arguments": {{}}, "purpose": "why this step"}}

Return ONLY the JSON array, no other text.
"""

SUMMARIZE_SYSTEM = """You are an investment analyst. Answer the user's question using only the provided data.

Context:
- User question: {question}
- Tool results: {tool_results}

Rules:
- Use only the provided data, never fabricate
- Money is CNY unless stated otherwise, format large numbers with commas
- Keep answers concise and well-structured
- Reply in the same language as the user
- Use Markdown formatting: **bold** for key figures and conclusions, tables for numerical comparisons, bullet lists for breakdowns, `code` for asset codes
"""

# ---- Limits ----

MAX_REACT_ITERATIONS = 5
MAX_PLAN_STEPS = 5


# ---- Helpers ----

def _extract_json(text: str) -> Any:
    """Extract a JSON object or array from text, handling markdown fences."""
    cleaned = text.strip()
    # Try markdown fence first
    fence = re.search(
        r"```(?:json)?\s*([\[{].*?[\]}])\s*```", cleaned, flags=re.DOTALL
    )
    if fence:
        cleaned = fence.group(1)
    elif not (cleaned.startswith("{") or cleaned.startswith("[")):
        # Find first JSON boundary
        brace = cleaned.find("{")
        bracket = cleaned.find("[")
        candidates = [i for i in (brace, bracket) if i >= 0]
        start = min(candidates) if candidates else -1
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _get_configurable(config: RunnableConfig) -> dict[str, Any]:
    """Extract configurable parameters from LangGraph config."""
    return config.get("configurable", {})


def _build_system_prompt(base: str, cfg: dict[str, Any]) -> str:
    """Prepend role persona to the system prompt if available."""
    role = cfg.get("role")
    if role is not None:
        return role.to_system_prompt() + "\n" + base
    return base


def _trace(cfg: dict[str, Any], event: dict[str, Any]) -> None:
    """Record a trace event if a trace sink is configured."""
    sink = cfg.get("trace")
    if sink is not None:
        sink.record(event)


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---- Nodes ----


def classify_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Classify the user's intent: skill, react, or plan_execute."""
    cfg = _get_configurable(config)
    llm = cfg["llm"]
    skills = cfg.get("skills", [])
    user_msg = state["user_message"]
    if not user_msg.strip():
        return {"intent": "react"}

    # Dynamic skill matching (keyword-based, fast path)
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
    """ReAct loop: Thought → Action → Observation → Thought..."""
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

    tools_list = json.dumps(tools.list_tools(), ensure_ascii=False)
    system_prompt = _build_system_prompt(
        REACT_SYSTEM.format(tools=tools_list), cfg
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

    # Check for Final Answer — strict format first
    final_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
    if final_match:
        return {
            "final_answer": final_match.group(1).strip(),
            "react_iteration": iteration + 1,
        }

    # Parse Action — try strict ReAct format first, then fallback
    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", response)
    tool_name = ""
    if action_match:
        tool_name = action_match.group(1).strip()
    else:
        # Fallback: try to find a tool name mentioned anywhere in the response
        for tname in tools.tool_names():
            if tname in response:
                tool_name = tname
                break

    if not tool_name:
        # No tool call found — LLM probably answered directly
        # If we have tool results from a previous iteration, use them as context
        if iteration > 0 and tool_results:
            return {
                "final_answer": response.strip()[:2000],
                "react_iteration": iteration + 1,
            }
        # First iteration: trust the LLM response as a direct answer
        # (e.g., simple greetings or questions that don't need tools)
        return {
            "final_answer": response.strip()[:2000],
            "react_iteration": iteration + 1,
        }

    if tool_name not in tools.tool_names():
        return {
            "error": f"Unknown tool: {tool_name}",
            "final_answer": f"工具 {tool_name} 不可用。",
            "react_iteration": iteration + 1,
        }

    args_match = re.search(
        r"Action Input:\s*(.+?)(?:\n\S|\Z)", response, re.DOTALL
    )
    arguments: dict[str, Any] = {}
    if args_match:
        try:
            parsed = _extract_json(args_match.group(1).strip())
            if isinstance(parsed, dict):
                arguments = parsed
        except (json.JSONDecodeError, ValueError):
            arguments = {}

    started = time.perf_counter()
    try:
        result = tools.call(tool_name, arguments)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        _trace(cfg, {
            "ok": True,
            "node": "react",
            "tool": tool_name,
            "arguments": arguments,
            "elapsed_ms": elapsed_ms,
            "ts": _now_ts(),
        })
        tool_results.append({
            "name": tool_name,
            "arguments": arguments,
            "result": result,
            "elapsed_ms": elapsed_ms,
        })
    except FinanceToolError as err:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        _trace(cfg, {
            "ok": False,
            "node": "react",
            "tool": tool_name,
            "arguments": arguments,
            "error": str(err),
            "elapsed_ms": elapsed_ms,
            "ts": _now_ts(),
        })
        tool_results.append({
            "name": tool_name,
            "arguments": arguments,
            "error": str(err),
            "elapsed_ms": elapsed_ms,
        })

    return {
        "tool_results": tool_results,
        "react_iteration": iteration + 1,
        "tool_call_count": state.get("tool_call_count", 0) + 1,
    }


def plan_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Generate an execution plan for complex multi-step analysis."""
    cfg = _get_configurable(config)
    llm = cfg["llm"]
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
            "current_plan": [
                {"step": 1, "tool": "finance.holdings_summary", "arguments": {}, "purpose": "获取当前持仓概况"},
                {"step": 2, "tool": "finance.bucket_allocation", "arguments": {}, "purpose": "检查配置偏离度"},
            ],
        }

    try:
        plan = json.loads(response.strip())
        if not isinstance(plan, list):
            data = _extract_json(response)
            plan = data if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError):
        plan = []

    if not plan:
        plan = [
            {"step": 1, "tool": "finance.holdings_summary", "arguments": {}, "purpose": "获取当前持仓概况"},
            {"step": 2, "tool": "finance.bucket_allocation", "arguments": {}, "purpose": "检查配置偏离度"},
        ]

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
            "ok": True,
            "node": "execute",
            "tool": step["tool"],
            "arguments": step.get("arguments", {}),
            "elapsed_ms": elapsed_ms,
            "ts": _now_ts(),
        })
        tool_results.append({
            "name": step["tool"],
            "arguments": step.get("arguments", {}),
            "result": result,
            "elapsed_ms": elapsed_ms,
        })
    except FinanceToolError as err:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        _trace(cfg, {
            "ok": False,
            "node": "execute",
            "tool": step["tool"],
            "arguments": step.get("arguments", {}),
            "error": str(err),
            "elapsed_ms": elapsed_ms,
            "ts": _now_ts(),
        })
        tool_results.append({
            "name": step["tool"],
            "arguments": step.get("arguments", {}),
            "error": str(err),
            "elapsed_ms": elapsed_ms,
        })

    return {
        "tool_results": tool_results,
        "tool_call_count": tool_call_count + 1,
    }


def summarize_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Generate the final answer from tool results."""
    cfg = _get_configurable(config)
    llm = cfg["llm"]

    user_msg = state["user_message"]
    tool_results = state.get("tool_results", [])

    # Only use existing final_answer if it's a real answer (not a skill placeholder)
    existing = state.get("final_answer", "")
    if existing and not existing.startswith("技能「"):
        return {"final_answer": existing}

    if not tool_results:
        return {"final_answer": "未获取到任何数据，请检查工具调用。"}

    system_prompt = _build_system_prompt(
        SUMMARIZE_SYSTEM.format(
            question=user_msg,
            tool_results=json.dumps(tool_results, ensure_ascii=False, indent=2),
        ),
        cfg,
    )

    try:
        answer = llm.complete(system_prompt, [{"role": "user", "content": "请回答。"}])
        return {"final_answer": answer.strip()}
    except LLMError as err:
        return {"final_answer": "无法生成回答，请查看原始数据。", "error": str(err)}
    except Exception:
        return {"final_answer": "无法生成回答，请查看原始数据。"}


def skill_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Execute a predefined skill workflow and return results.

    Does NOT set final_answer — let summarize_node generate the real answer.
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

    # Let summarize_node generate the real answer from tool results
    return {
        "tool_results": result.get("results", []),
        "tool_call_count": result.get("steps_executed", 0),
        "findings": findings,
    }