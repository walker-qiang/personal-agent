"""Commander, delegate, aggregate, and reflection nodes.

Commander plans delegation, delegate runs domain agent ReAct loops,
aggregate combines results, reflection reviews quality.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from langgraph.types import RunnableConfig, interrupt

from ...llm import LLMError, LLMClient, FunctionCallResult
from ...tools import FinanceToolError, ToolRegistry
from ...agent.registry import AgentRegistry

from ._helpers import (
    _build_history_context,
    _build_react_final_answer,
    _build_tools_for_llm,
    _check_early_stop,
    _classify_query_factuality,
    _evaluate_sufficiency,
    _extract_json,
    _extract_media_urls,
    _fix_media_answer,
    _get_configurable,
    _is_high_risk,
    _is_refusal,
    _push_event,
    COMMANDER_AGGREGATE_PROMPT,
    COMMANDER_PLAN_PROMPT,
    DOMAIN_AGENT_REACT_SYSTEM,
    MAX_PLAN_STEPS,
    MAX_REACT_ITERATIONS,
    MAX_SUBTASK_ITERATIONS,
    MAX_SUBTASKS,
    REFLECTION_PROMPT,
    REVISE_PROMPT,
    EVALUATOR_INTERVAL,
)
from .react import _react_execute_tool_calls
from ..anti_hallucination import verify_all_claims, build_verified_output
from ..state import AgentState

logger = logging.getLogger("matrix.orchestration")


def commander_plan_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Commander plans the delegation strategy. Entry node of the graph.

    This is the ONLY node that calls the LLM for intent classification.
    Empty plan [] = simple question (Commander handles directly).
    Non-empty plan = delegate to domain experts.
    """
    cfg = _get_configurable(config)
    llm = cfg.get("pipeline_llm", cfg["llm"])
    agent_registry: AgentRegistry = cfg["agent_registry"]

    user_msg = state["user_message"]

    if not user_msg.strip():
        return {
            "delegation_plan": [
                {"step": 1, "agent_id": "commander", "task": "回复空消息", "purpose": "直接回答"}
            ],
            "current_step": 0,
        }

    agents_desc = json.dumps(
        agent_registry.agents_for_commander(), ensure_ascii=False, indent=2
    )

    history_context = _build_history_context(cfg.get("history", []))

    try:
        response = llm.complete(
            COMMANDER_PLAN_PROMPT.format(agents=agents_desc, question=user_msg, max_subtasks=MAX_SUBTASKS),
            [{"role": "user", "content": history_context + user_msg}],
            temperature=0.1,
        )
        plan = _extract_json(response)
        if not isinstance(plan, list):
            plan = []
    except (LLMError, json.JSONDecodeError, ValueError) as e:
        logger.warning("commander_plan LLM/parse failed: %s", type(e).__name__)
        plan = []

    valid_ids = {a["id"] for a in agent_registry.agents_for_commander()}
    valid_ids.add("commander")
    plan = [s for s in plan if s.get("agent_id", "") in valid_ids]

    merged: list[dict[str, Any]] = []
    seen_agents: set[str] = set()
    for s in plan:
        aid = s.get("agent_id", "")
        if aid in seen_agents:
            for m in merged:
                if m.get("agent_id") == aid:
                    m["task"] = m["task"] + "；同时：" + s.get("task", "")
                    if s.get("skill_name"):
                        m["skill_name"] = s["skill_name"]
                    break
        else:
            seen_agents.add(aid)
            merged.append(s)
    plan = merged

    agent_ids_in_plan = {s.get("agent_id", "") for s in plan}
    is_subtask = len(plan) > 1 and len(agent_ids_in_plan) == 1
    plan_type = "subtask" if is_subtask else "agent"

    if is_subtask:
        plan = plan[:MAX_SUBTASKS]
    else:
        plan = plan[:MAX_PLAN_STEPS]

    if not plan:
        plan = [
            {"step": 1, "agent_id": "commander", "task": user_msg, "purpose": "直接回答"}
        ]
        plan_type = "agent"

    logger.info("commander: plan_type=%s steps=%d agents=%s", plan_type, len(plan), agent_ids_in_plan)

    for i, step in enumerate(plan):
        if "step" not in step:
            step["step"] = i + 1
        if "skill_name" not in step:
            step["skill_name"] = ""

    return {
        "delegation_plan": plan,
        "current_step": 0,
        "plan_type": plan_type,
    }


def delegate_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Execute one step of the delegation plan.

    Runs a domain agent's ReAct loop to complete the assigned task.
    Returns only the NEW result for this agent (single-element list).
    """
    cfg = _get_configurable(config)
    agent_registry: AgentRegistry = cfg["agent_registry"]
    full_tools: ToolRegistry = cfg["full_tools"]

    plan = state.get("delegation_plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan):
        return {}

    step = plan[current_step]
    agent_id = step.get("agent_id", "")
    task = step.get("task", state["user_message"])
    skill_name = step.get("skill_name", "")

    agent_def = agent_registry.get(agent_id)
    if agent_def is None:
        return {
            "agent_results": [{
                "agent_id": agent_id,
                "task": task,
                "error": f"Agent not found: {agent_id}",
                "findings": [],
            }],
        }

    agent_tools = agent_registry.build_tool_registry(agent_id, full_tools)
    agent_skills = agent_registry.load_skills_for_agent(agent_id)

    skill_results: list[dict[str, Any]] = []
    if skill_name:
        skill = next((s for s in agent_skills if getattr(s, "name", "") == skill_name), None)
        if skill is not None:
            from ..skills.executor import execute_skill
            skill_result = execute_skill(skill, agent_tools, cfg.get("trace"))
            skill_results = skill_result.get("results", [])
            if skill_result.get("errors"):
                return {
                    "agent_results": [{
                        "agent_id": agent_id,
                        "task": task,
                        "skill_name": skill_name,
                        "error": "; ".join(skill_result["errors"]),
                        "findings": skill_result.get("findings", []),
                        "tool_results": skill_results,
                    }],
                    "tool_results": skill_results,
                    "tool_call_count": len(skill_results),
                }

    is_subtask = state.get("plan_type") == "subtask"
    max_iter = MAX_SUBTASK_ITERATIONS if is_subtask else MAX_REACT_ITERATIONS
    result = _run_domain_agent_react(
        agent_def=agent_def,
        task=task,
        tools=agent_tools,
        skill_results=skill_results,
        cfg=cfg,
        session_id=state.get("session_id", ""),
        agent_id=agent_id,
        max_iterations=max_iter,
    )

    new_result = {
        "agent_id": agent_id,
        "task": task,
        "skill_name": skill_name,
        "result": result.get("answer", ""),
        "findings": result.get("findings", []),
        "tool_results": result.get("tool_results", []),
        "error": result.get("error", ""),
    }
    new_tool_results = result.get("tool_results", [])

    pending_actions = []
    for tr in new_tool_results:
        tool_name = tr.get("name", "")
        if _is_high_risk(tool_name) and not tr.get("error"):
            pending_actions.append({
                "agent": agent_id,
                "tool": tool_name,
                "args": tr.get("arguments", {}),
                "summary": f"{agent_id} 将调用 {tool_name}",
            })

    return {
        "agent_results": [new_result],
        "tool_results": new_tool_results,
        "tool_call_count": len(new_tool_results),
        "needs_confirmation": len(pending_actions) > 0,
        "pending_actions": pending_actions,
    }


# ── Domain Agent ReAct loop ──────────────────────────────────────────────────

def _run_domain_agent_react(
    agent_def: Any,
    task: str,
    tools: ToolRegistry,
    skill_results: list[dict[str, Any]],
    cfg: dict[str, Any],
    session_id: str = "",
    agent_id: str = "",
    max_iterations: int = MAX_REACT_ITERATIONS,
) -> dict[str, Any]:
    """Run a ReAct loop for a domain agent using standard multi-turn tool calling.

    The loop body delegates to _react_call_llm, _react_handle_tool_calls,
    and _react_handle_content for clarity.
    """
    llm = cfg["llm"]
    tool_results: list[dict[str, Any]] = list(skill_results)
    iteration = 0
    prev_result_count = len(tool_results)
    consecutive_failures = 0
    consecutive_no_progress = 0
    original_question = cfg.get("question", task)
    history_context = _build_history_context(cfg.get("history", []))

    system_prompt = DOMAIN_AGENT_REACT_SYSTEM.format(
        agent_name=agent_def.name,
        persona=agent_def.persona,
        task=task,
        today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )

    # Pinned working memory: inject user's original request
    wm = cfg.get("working_memory", {})
    pinned = wm.get("pinned", "")
    if not pinned:
        user_msgs = [m for m in cfg.get("history", []) if m.get("role") == "user"]
        if user_msgs:
            pinned = str(user_msgs[0].get("content", ""))[:2000]
    if pinned:
        system_prompt = f"**Pinned Goal (your anchor):** {pinned}\n\n" + system_prompt

    # Inject active insights
    insights = wm.get("insights", [])
    if insights:
        insight_block = "\n".join(f"- {i}" for i in insights[:5])
        system_prompt += f"\n\n## Key Insights (from previous steps)\n{insight_block}"
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": history_context + f"请完成以下任务：{task}"},
    ]

    _push_event(cfg, "progress", {"message": "Agent 开始分析任务，调用工具获取数据..."})

    while iteration < max_iterations:
        iteration += 1

        early_reason = _check_early_stop(
            tool_results, iteration, consecutive_failures, consecutive_no_progress,
        )
        if early_reason:
            logger.info("ReAct early stop at iteration %d: %s", iteration, early_reason)
            break

        llm_tools = _build_tools_for_llm(tools)
        result = _react_call_llm(llm, system_prompt, messages, llm_tools, original_question,
                                 agent_def, task, tool_results, tools)

        if isinstance(result, dict) and "answer" in result:
            # LLM call failed → fallback returned a final answer
            result["answer"] = _fix_media_answer(result["answer"], tool_results)
            return result

        if result.tool_calls:
            reaction = _react_handle_tool_calls(
                result, messages, tool_results, iteration, tools, llm, system_prompt,
                llm_tools, cfg, agent_id, session_id, original_question,
                consecutive_failures, consecutive_no_progress, prev_result_count,
            )
            if reaction["done"]:
                return reaction["result"]
            messages = reaction["messages"]
            tool_results = reaction["tool_results"]
            consecutive_failures = reaction["consecutive_failures"]
            consecutive_no_progress = reaction["consecutive_no_progress"]
            prev_result_count = reaction["prev_result_count"]
            continue

        # content / no-tools branch
        content_result = _react_handle_content(
            result, messages, tool_results, iteration, max_iterations,
            consecutive_no_progress, agent_def, task, tools, llm,
        )
        if content_result["done"]:
            return content_result["result"]
        messages = content_result["messages"]
        tool_results = content_result["tool_results"]
        consecutive_no_progress = content_result["consecutive_no_progress"]

    return _react_generate_partial_answer(tool_results, llm)


# ── ReAct helpers ────────────────────────────────────────────────────────────

def _react_call_llm(
    llm, system_prompt: str, messages: list, llm_tools: list,
    original_question: str, agent_def: Any, task: str,
    tool_results: list, tools: ToolRegistry,
) -> FunctionCallResult | dict[str, Any]:
    """Call the LLM with error handling. Returns FunctionCallResult on success,
    or a final answer dict on failure."""
    try:
        temp = _classify_query_factuality(original_question)
        return llm.function_call(system_prompt, messages, llm_tools, temperature=temp)
    except (LLMError, ConnectionError, TimeoutError, ValueError, OSError) as e:
        msg_len = sum(len(str(m.get("content", ""))) for m in messages)
        logger.error("ReAct: LLM call failed in domain_agent_react: %s (msg_count=%d, total_chars=%d)",
                     type(e).__name__, len(messages), msg_len)
        react_result = _domain_react_fallback(agent_def, task, tool_results, tools, llm)
        if react_result.get("answer") or react_result.get("tool_results"):
            return react_result
        return {"answer": "无法完成任务，请检查工具和数据。", "tool_results": tool_results, "findings": []}


def _react_handle_tool_calls(
    result: FunctionCallResult,
    messages: list,
    tool_results: list,
    iteration: int,
    tools: ToolRegistry,
    llm,
    system_prompt: str,
    llm_tools: list,
    cfg: dict,
    agent_id: str,
    session_id: str,
    original_question: str,
    consecutive_failures: int,
    consecutive_no_progress: int,
    prev_result_count: int,
) -> dict:
    """Handle the tool-calls branch of the ReAct loop.

    Returns:
        done: bool — True if the loop should stop
        result: dict — final answer if done
        messages, tool_results, consecutive_failures, consecutive_no_progress,
        prev_result_count — updated state if not done
    """
    # Push thinking
    _push_thinking_for_calls(result, tools, tool_results, cfg)

    # Append assistant message
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": result.content or ""}
    api_tool_calls = []
    for tc in result.tool_calls:
        api_tool_calls.append({
            "id": tc.id, "type": "function",
            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
        })
    assistant_msg["tool_calls"] = api_tool_calls
    messages.append(assistant_msg)

    # Execute tools with shared guard logic
    exec_result = _react_execute_tool_calls(
        tool_calls_raw=api_tool_calls, agent_tools=tools, messages=messages,
        accumulated=tool_results, agent_id=agent_id, session_id=session_id,
        cfg=cfg, node_name="delegate",
        consecutive_failures=consecutive_failures,
        consecutive_no_progress=consecutive_no_progress,
        prev_result_count=prev_result_count,
        push_events=True,
        ref_store=cfg.get("ref_store"),
    )

    messages = exec_result["messages"]
    tool_results.extend(exec_result["new_tool_results"])
    consecutive_failures = exec_result["consecutive_failures"]
    consecutive_no_progress = exec_result["consecutive_no_progress"]
    prev_result_count = exec_result["prev_result_count"]

    if exec_result["force_summarize"]:
        messages.append({
            "role": "user",
            "content": "所有工具调用均为重复。请基于已有结果为用户总结。",
        })
        try:
            final = llm.function_call(system_prompt, messages, llm_tools)
            if final.content:
                return {"done": True, "result": {
                    "answer": _fix_media_answer(final.content.strip(), tool_results),
                    "tool_results": tool_results, "findings": [],
                }}
        except (LLMError, ConnectionError, TimeoutError, ValueError, OSError) as e:
            logger.error("ReAct: LLM final call failed: %s", type(e).__name__)
        return {"done": True, "result": {
            "answer": _fix_media_answer("工具调用已完成，但无法生成总结。", tool_results),
            "tool_results": tool_results, "findings": [],
        }}

    # Periodic evaluator
    if iteration > 0 and iteration % EVALUATOR_INTERVAL == 0 and tool_results:
        sufficient, reason = _evaluate_sufficiency(original_question, tool_results, "", llm)
        if sufficient:
            logger.info("ReAct: evaluator sufficient at iter %d: %s", iteration, reason)
            return {"done": True, "result": {
                "answer": _fix_media_answer("已收集到足够数据进行分析。", tool_results),
                "tool_results": tool_results, "findings": [],
            }}

    return {
        "done": False,
        "messages": messages, "tool_results": tool_results,
        "consecutive_failures": consecutive_failures,
        "consecutive_no_progress": consecutive_no_progress,
        "prev_result_count": prev_result_count,
    }


def _push_thinking_for_calls(
    result: FunctionCallResult, tools: ToolRegistry,
    tool_results: list, cfg: dict,
) -> None:
    """Push a thinking event for tool calls."""
    if result.content:
        _push_event(cfg, "thinking", {"content": result.content.strip()})
    else:
        tool_names = []
        for tc in result.tool_calls:
            if tc.name not in tools.tool_names():
                continue
            args_key = json.dumps(tc.arguments, sort_keys=True)
            if any(
                prev.get("name") == tc.name
                and json.dumps(prev.get("arguments", {}), sort_keys=True) == args_key
                for prev in tool_results
            ):
                continue
            tool_names.append(tc.name)
        if tool_names:
            _push_event(cfg, "thinking", {"content": f"正在调用 {'、'.join(tool_names)} 获取数据..."})


def _react_handle_content(
    result: FunctionCallResult,
    messages: list,
    tool_results: list,
    iteration: int,
    max_iterations: int,
    consecutive_no_progress: int,
    agent_def: Any,
    task: str,
    tools: ToolRegistry,
    llm,
) -> dict:
    """Handle the content/no-tools branch of the ReAct loop.

    Returns {done, result, messages, tool_results, consecutive_no_progress}.
    """
    if result.content:
        if _is_refusal(result.content) and tool_results:
            if iteration < max_iterations:
                messages.append({
                    "role": "user",
                    "content": (
                        "你刚才说无法提供数据，但实际上工具已经返回了结果。"
                        "请基于以上工具返回的真实数据回答用户的问题。"
                        "直接给出天气信息，不要说你无法提供。"
                    ),
                })
                return {
                    "done": False, "messages": messages, "tool_results": tool_results,
                    "consecutive_no_progress": consecutive_no_progress + 1,
                }

        return {"done": True, "result": {
            "answer": _fix_media_answer(result.content.strip(), tool_results),
            "tool_results": tool_results, "findings": [],
        }}

    # No content, no tool calls → fallback
    react_result = _domain_react_fallback(agent_def, task, tool_results, tools, llm)
    if react_result.get("answer"):
        react_result["answer"] = _fix_media_answer(react_result["answer"], tool_results)
        return {"done": True, "result": react_result}
    if react_result.get("tool_results"):
        return {"done": False, "messages": messages, "tool_results": react_result["tool_results"],
                "consecutive_no_progress": consecutive_no_progress}

    return {"done": True, "result": {
        "answer": _fix_media_answer("无法完成任务，请检查工具和数据。", tool_results),
        "tool_results": tool_results, "findings": [],
    }}


def _react_generate_partial_answer(
    tool_results: list, llm,
) -> dict[str, Any]:
    """Generate a partial answer when max iterations reached."""
    if tool_results:
        try:
            summary_prompt = f"""你已收集了以下工具结果，但步数已达上限。请基于这些数据给出一段简洁的回答：

{json.dumps(tool_results, ensure_ascii=False, indent=2)}

请直接回答用户的问题，不要提及"步数"或"限制"。"""
            partial = llm.complete(
                "你是专业的回答助手，请基于已有数据回答。",
                [{"role": "user", "content": summary_prompt}],
                temperature=0.1,
            )
            if partial and len(partial) > 10:
                return {"answer": _fix_media_answer(partial, tool_results), "tool_results": tool_results, "findings": []}
        except Exception:
            pass
    return {
        "answer": _fix_media_answer("已达到最大分析步数，请基于已有数据回答。", tool_results),
        "tool_results": tool_results, "findings": [],
    }


# ── Fallback ReAct ───────────────────────────────────────────────────────────

def _domain_react_fallback(
    agent_def: Any,
    task: str,
    tool_results: list[dict[str, Any]],
    tools: ToolRegistry,
    llm,
) -> dict[str, Any]:
    """Regex-based ReAct fallback for domain agents."""
    tools_list = json.dumps(tools.list_tools(), ensure_ascii=False)
    system_prompt = f"""You are {agent_def.name}, a domain expert.

{agent_def.persona}

Current task: {task}

You have access to these tools: {tools_list}

IMPORTANT: You MUST call tools using this exact format:
<tool_call>tool_name</tool_call>
<arguments>
{{"param": "value"}}
</arguments>

You can call ONE tool at a time. After the tool responds, you can analyze
the result and decide whether to call another tool or output the answer.

After calling all needed tools, output your final answer with <answer>...</answer>."""

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"Complete this task: {task}"},
    ]

    for _ in range(MAX_REACT_ITERATIONS):
        try:
            response = llm.complete(system_prompt, messages, temperature=0.1)
        except (LLMError, ConnectionError, TimeoutError, ValueError, OSError) as e:
            logger.error("ReAct fallback LLM failed: %s", type(e).__name__)
            return {"answer": "", "tool_results": tool_results, "findings": []}

        tc_match = _extract_tool_call(response)
        if tc_match:
            tool_name = tc_match["name"]
            try:
                args = json.loads(tc_match["args"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            if tool_name in tools.tool_names():
                try:
                    result = tools.call(tool_name, args)
                    tool_results.append({
                        "name": tool_name, "arguments": args, "result": result,
                    })
                except (FinanceToolError, TypeError) as err:
                    tool_results.append({
                        "name": tool_name, "arguments": args, "error": str(err),
                    })
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(tool_results[-1], ensure_ascii=False)}"})
            continue

        answer_match = _extract_answer(response)
        if answer_match:
            return {"answer": answer_match, "tool_results": tool_results, "findings": []}

        # No tool call or answer — stop
        return {"answer": response.strip(), "tool_results": tool_results, "findings": []}

    return {"answer": "", "tool_results": tool_results, "findings": []}


def _extract_tool_call(text: str) -> dict[str, str] | None:
    """Extract tool call from regex-based ReAct output."""
    import re
    tc = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not tc:
        return None
    args = re.search(r"<arguments>\s*(.*?)\s*</arguments>", text, re.DOTALL)
    return {"name": tc.group(1).strip(), "args": args.group(1).strip() if args else "{}"}


def _extract_answer(text: str) -> str | None:
    """Extract answer from regex-based ReAct output."""
    import re
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ── Aggregate ────────────────────────────────────────────────────────────────

def aggregate_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Commander reviews all agent results and aggregates them into a final answer."""
    cfg = _get_configurable(config)
    llm = cfg["llm"]

    user_msg = state["user_message"]
    agent_results = state.get("agent_results", [])

    if not agent_results:
        return {"needs_summary": True}

    if len(agent_results) == 1 and agent_results[0].get("agent_id") == "commander":
        result_text = agent_results[0].get("result", "")
        if result_text:
            return {"final_answer": result_text, "needs_summary": False, "skip_reflection": True}
        return {"needs_summary": True, "skip_reflection": True}

    all_errors = all(r.get("error") for r in agent_results)
    if all_errors:
        errors = "\n".join(f"- {r['agent_id']}: {r['error']}" for r in agent_results)
        return {"final_answer": f"所有领域专家执行失败：\n{errors}", "needs_summary": False}

    results_summary = []
    for r in agent_results:
        media_urls = _extract_media_urls(r.get("tool_results", []))
        result_text = r.get("result", "")
        if media_urls and not any(u in result_text for u in media_urls):
            result_text = media_urls + "\n\n" + result_text
        results_summary.append({
            "agent": r["agent_id"],
            "task": r.get("task", ""),
            "result": result_text[:2000],
            "error": r.get("error", ""),
        })

    history_context = _build_history_context(cfg.get("history", []))
    system_prompt = COMMANDER_AGGREGATE_PROMPT.format(
        question=user_msg,
        results=json.dumps(results_summary, ensure_ascii=False, indent=2),
    )

    try:
        response = llm.complete(
            system_prompt,
            [{"role": "user", "content": history_context + "请汇总回答。"}],
            temperature=0.4,
        )
        final_answer = response.strip()

        all_tool_results = []
        for r in agent_results:
            all_tool_results.extend(r.get("tool_results", []))
        if all_tool_results:
            verification = verify_all_claims(final_answer, all_tool_results, llm)
            if verification.total > 0:
                final_answer = build_verified_output(final_answer, verification)

        return {"final_answer": final_answer, "needs_summary": False}
    except LLMError as e:
        logger.error("aggregate_node LLM failed: %s", type(e).__name__)
        parts = []
        for r in agent_results:
            if r.get("result"):
                parts.append(f"### {r['agent_id']}\n{r['result']}")
        return {"final_answer": "\n\n".join(parts) if parts else "无法汇总结果。", "needs_summary": False}


# ── Reflection ───────────────────────────────────────────────────────────────

def reflection_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Internal quality check: verify → revise if needed → output clean answer."""
    cfg = _get_configurable(config)
    llm = cfg.get("pipeline_llm", cfg["llm"])

    answer = state.get("final_answer", "")
    user_msg = state.get("user_message", "")

    if state.get("skip_reflection"):
        return {}

    if not answer or not user_msg:
        return {}

    if len(answer) < 15 or answer.startswith("技能") or answer.startswith("所有领域专家"):
        return {}

    try:
        history_context = _build_history_context(cfg.get("history", []))
        response = llm.complete(
            REFLECTION_PROMPT.format(question=user_msg, answer=answer),
            [{"role": "user", "content": history_context + "Evaluate the answer."}],
            temperature=0.1,
        )
        data = _extract_json(response)
        if not isinstance(data, dict) or data.get("ok") is not False:
            return {}

        issues = data.get("issues", [])
        if not issues:
            return {}

        revise_response = llm.complete(
            REVISE_PROMPT.format(
                question=user_msg, answer=answer,
                issues="\n".join(f"- {i}" for i in issues),
            ),
            [{"role": "user", "content": "Rewrite the answer."}],
            temperature=0.2,
        )
        revised = revise_response.strip()
        if revised and len(revised) > 10:
            return {"final_answer": revised}
    except (LLMError, json.JSONDecodeError, ValueError) as e:
        logger.warning("reflection_node revise LLM failed: %s", type(e).__name__)

    return {}


# ── Confirm (HITL) ───────────────────────────────────────────────────────────

def confirm_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """HITL confirmation node: pauses the graph for user approval."""
    pending_actions = state.get("pending_actions", [])
    if not pending_actions:
        return {"confirmed": True}

    action_summary = []
    for action in pending_actions:
        action_summary.append(f"- {action.get('summary', str(action))}")

    # Use LangGraph interrupt for HITL
    interrupt({
        "type": "confirm",
        "actions": pending_actions,
        "message": "以下操作需要你确认：\n" + "\n".join(action_summary),
    })

    # After resume, check decision
    confirmed = state.get("confirmed", False)
    if not confirmed:
        return {"error": "用户取消了操作", "confirmed": True}

    return {"confirmed": True}