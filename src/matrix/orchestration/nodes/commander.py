"""Commander, delegate, aggregate, and reflection nodes.

Commander plans delegation, delegate runs domain agent ReAct loops,
aggregate combines results, reflection reviews quality.
"""

from __future__ import annotations

import json
import logging
import time
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
    _evaluate_sufficiency,
    _extract_json,
    _extract_media_urls,
    _fix_media_answer,
    _force_tool_call,
    _get_configurable,
    _is_high_risk,
    _is_refusal,
    _now_ts,
    _push_event,
    _trace,
    COMMANDER_AGGREGATE_PROMPT,
    COMMANDER_PLAN_PROMPT,
    DOMAIN_AGENT_REACT_SYSTEM,
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
    EVALUATOR_INTERVAL,
)
from .react import _react_execute_tool_calls
from ..state import AgentState

logger = logging.getLogger("matrix.orchestration")

def commander_plan_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Commander plans the delegation strategy. Entry node of the graph.

    This is the ONLY node that calls the LLM for intent classification.
    Empty plan [] = simple question (Commander handles directly).
    Non-empty plan = delegate to domain experts.

    LLM also judges whether the question needs RAG (personal knowledge base).
    Only queries RAG when the LLM decides it's relevant.
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

    # Build conversation history context for multi-turn awareness
    history_context = _build_history_context(cfg.get("history", []))

    try:
        response = llm.complete(
            COMMANDER_PLAN_PROMPT.format(agents=agents_desc, question=user_msg, max_subtasks=MAX_SUBTASKS),
            [{"role": "user", "content": history_context + user_msg}],
            temperature=0.1,  # Low temperature for structured planning
        )
        plan = _extract_json(response)
        if not isinstance(plan, list):
            plan = []
    except (LLMError, json.JSONDecodeError, ValueError) as e:
        logger.warning("commander_plan LLM/parse failed: %s", type(e).__name__)
        plan = []

    # Filter out any steps that reference non-existent agents (e.g. LLM hallucination)
    valid_ids = {a["id"] for a in agent_registry.agents_for_commander()}
    valid_ids.add("commander")  # commander can self-assign
    plan = [s for s in plan if s.get("agent_id", "") in valid_ids]

    # Merge steps for the same agent
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

    # Detect plan type: subtask decomposition (same agent, multiple steps) vs multi-agent
    agent_ids_in_plan = {s.get("agent_id", "") for s in plan}
    is_subtask = len(plan) > 1 and len(agent_ids_in_plan) == 1
    plan_type = "subtask" if is_subtask else "agent"

    # Limit plan steps accordingly
    if is_subtask:
        plan = plan[:MAX_SUBTASKS]
    else:
        plan = plan[:MAX_PLAN_STEPS]

    # Empty plan = simple question. Always let Commander handle it via ReAct.
    if not plan:
        plan = [
            {"step": 1, "agent_id": "commander", "task": user_msg, "purpose": "直接回答"}
        ]
        plan_type = "agent"

    logger.info("commander: plan_type=%s steps=%d agents=%s", plan_type, len(plan), agent_ids_in_plan)

    # Ensure each step has required fields
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
    agent_results / tool_results / tool_call_count use operator.add reducers
    in AgentState, so LangGraph merges parallel branch outputs automatically.
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

    # Look up agent
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

    # Build agent-specific tools and skills
    agent_tools = agent_registry.build_tool_registry(agent_id, full_tools)
    agent_skills = agent_registry.load_skills_for_agent(agent_id)

    # Execute skill if specified
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

    # Run domain agent's ReAct loop
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

    logger.debug(
        "delegate: agent=%s answer_len=%d preview=%s tools=%d",
        agent_id, len(result.get("answer", "")),
        result.get("answer", "")[:80],
        len(new_tool_results),
    )

    # Check for high-risk tool calls
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

    Uses OpenAI-compatible format:
      assistant(tool_calls[]) → tool(tool_call_id, content) → assistant(...)

    Args:
        max_iterations: Per-agent ReAct iteration limit (default: 20 for full agents,
                        10 for subtasks).
    Returns: {"answer": str, "findings": list, "tool_results": list, "error": str}
    """
    llm = cfg["llm"]
    tool_results: list[dict[str, Any]] = list(skill_results)
    iteration = 0
    prev_result_count = len(tool_results)
    consecutive_failures = 0
    consecutive_no_progress = 0

    # Extract the original user question from config for evaluator
    original_question = cfg.get("question", task)

    # Build conversation history context for multi-turn awareness
    history_context = _build_history_context(cfg.get("history", []))

    # System prompt
    system_prompt = DOMAIN_AGENT_REACT_SYSTEM.format(
        agent_name=agent_def.name,
        persona=agent_def.persona,
        task=task,
        today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )

    # Build initial messages — standard multi-turn format
    task_content = history_context + f"请完成以下任务：{task}"
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": task_content},
    ]

    _push_event(cfg, "progress", {"message": "Agent 开始分析任务，调用工具获取数据..."})

    while iteration < max_iterations:
        iteration += 1

        # ---- Early stop check ----
        early_reason = _check_early_stop(
            tool_results, iteration,
            consecutive_failures, consecutive_no_progress,
        )
        if early_reason:
            logger.info("ReAct early stop at iteration %d: %s", iteration, early_reason)
            break

        llm_tools = _build_tools_for_llm(tools)

        try:
            temp = _classify_query_factuality(original_question)
            result: FunctionCallResult = llm.function_call(
                system_prompt, messages, llm_tools, temperature=temp,
            )
        except (LLMError, ConnectionError, TimeoutError, ValueError, OSError) as e:
            msg_len = sum(len(str(m.get("content", ""))) for m in messages)
            logger.error("ReAct: LLM call failed in domain_agent_react: %s (msg_count=%d, total_chars=%d)", type(e).__name__, len(messages), msg_len)
            # Fallback to regex-based ReAct
            react_result = _domain_react_fallback(agent_def, task, tool_results, tools, llm)
            if react_result.get("answer"):
                react_result["answer"] = _fix_media_answer(react_result["answer"], tool_results)
                return react_result
            if react_result.get("tool_results"):
                tool_results = react_result["tool_results"]
                continue
            return {"answer": _fix_media_answer("无法完成任务，请检查工具和数据。", tool_results), "tool_results": tool_results, "findings": []}

        if result.tool_calls:
            # Push thinking content (LLM reasoning before tool calls)
            # Note: most LLMs return empty content when calling tools, so we
            # use the LLM's content if available, otherwise generate a brief
            # message from the tool actions.
            if result.content:
                _push_event(cfg, "thinking", {"content": result.content.strip()})
            else:
                # Generate thinking message from tool calls
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
                    names_str = "、".join(tool_names)
                    _push_event(cfg, "thinking", {
                        "content": f"正在调用 {names_str} 获取数据...",
                    })

            # Append assistant message with tool_calls (standard format)
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": result.content or "",
            }
            api_tool_calls: list[dict[str, Any]] = []
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
            messages.append(assistant_msg)

            # Execute tools with shared guard logic (replaces inline loop)
            exec_result = _react_execute_tool_calls(
                tool_calls_raw=api_tool_calls,
                agent_tools=tools,
                messages=messages,
                accumulated=tool_results,
                agent_id=agent_id,
                session_id=session_id,
                cfg=cfg,
                node_name="delegate",
                consecutive_failures=consecutive_failures,
                consecutive_no_progress=consecutive_no_progress,
                prev_result_count=prev_result_count,
                push_events=True,
            )

            messages = exec_result["messages"]
            tool_results.extend(exec_result["new_tool_results"])
            consecutive_failures = exec_result["consecutive_failures"]
            consecutive_no_progress = exec_result["consecutive_no_progress"]
            prev_result_count = exec_result["prev_result_count"]

            if exec_result["force_summarize"]:
                # All duplicates — ask LLM to summarize from existing messages
                messages.append({
                    "role": "user",
                    "content": "所有工具调用均为重复。请基于已有结果为用户总结。",
                })
                try:
                    final = llm.function_call(system_prompt, messages, llm_tools)
                    if final.content:
                        return {
                            "answer": _fix_media_answer(final.content.strip(), tool_results),
                            "tool_results": tool_results,
                            "findings": [],
                        }
                except (LLMError, ConnectionError, TimeoutError, ValueError, OSError) as e:
                    logger.error("ReAct: LLM call failed in domain_agent_react final: %s (msg_count=%d)", type(e).__name__, len(messages))
                    pass
                return {
                    "answer": _fix_media_answer("工具调用已完成，但无法生成总结。", tool_results),
                    "tool_results": tool_results,
                    "findings": [],
                }

            # ---- Periodic evaluator ----
            if iteration > 0 and iteration % EVALUATOR_INTERVAL == 0 and tool_results:
                sufficient, reason = _evaluate_sufficiency(
                    original_question, tool_results, "", llm,
                )
                if sufficient:
                    logger.info("ReAct: evaluator says sufficient at iter %d: %s", iteration, reason)
                    break

            continue  # Loop back — LLM sees tool results as proper messages

        if result.content:
            # Refusal check: if the LLM says it cannot answer but we have tool
            # results, the model ignored the tool data. Append a follow-up
            # message forcing it to use the available data and loop again.
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
                    consecutive_no_progress += 1
                    continue

            # No always-run evaluator here: the periodic evaluator above already
            # checks sufficiency. Calling an extra LLM per iteration to judge
            # the answer creates false negatives that override correct answers.
            return {
                "answer": _fix_media_answer(result.content.strip(), tool_results),
                "tool_results": tool_results,
                "findings": [],
            }

        # No content and no tool calls — fallback
        react_result = _domain_react_fallback(agent_def, task, tool_results, tools, llm)
        if react_result.get("answer"):
            react_result["answer"] = _fix_media_answer(react_result["answer"], tool_results)
            return react_result
        if react_result.get("tool_results"):
            tool_results = react_result["tool_results"]
            continue

        return {"answer": _fix_media_answer("无法完成任务，请检查工具和数据。", tool_results), "tool_results": tool_results, "findings": []}

    # Max iterations reached — try to generate a partial answer from collected data
    if tool_results:
        try:
            summary_prompt = f"""你已收集了以下工具结果，但步数已达上限。请基于这些数据给出一段简洁的回答：

{json.dumps(tool_results, ensure_ascii=False, indent=2)}

请直接回答用户的问题，不要提及"步数"或"限制"。"""
            partial = llm.complete("你是专业的回答助手，请基于已有数据回答。", [{"role": "user", "content": summary_prompt}], temperature=0.1)
            if partial and len(partial) > 10:
                return {"answer": _fix_media_answer(partial, tool_results), "tool_results": tool_results, "findings": []}
        except Exception:
            pass
    return {
        "answer": _fix_media_answer("已达到最大分析步数，请基于已有数据回答。", tool_results),
        "tool_results": tool_results,
        "findings": [],
    }



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

Available tools: {tools_list}

You MUST follow this EXACT format:

When you need to call a tool:
Thought: [reason about what to do next]
Action: [tool_name exactly as listed]
Action Input: [valid JSON arguments]

When you have enough information:
Final Answer: [your answer using Markdown]

Strict rules:
- ONE action per turn
- Use only listed tool names with valid JSON arguments
- Never fabricate data
- Reply in the same language as the task"""

    context = f"Task: {task}\n\n"
    if tool_results:
        context += (
            "Previous tool results:\n"
            + json.dumps(tool_results, ensure_ascii=False, indent=2)
            + "\n\n"
        )
    context += "Complete the task."

    try:
        response = llm.complete(system_prompt, [{"role": "user", "content": context}], temperature=0.1)
    except LLMError as e:
        logger.error("domain_react_fallback LLM failed: %s", type(e).__name__)
        return {"answer": "LLM 调用失败。", "tool_results": tool_results, "findings": []}

    # Check for Final Answer
    final_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
    if final_match:
        return {"answer": final_match.group(1).strip(), "tool_results": tool_results, "findings": []}

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

    if not tool_name or tool_name not in tools.tool_names():
        return {"answer": response.strip(), "tool_results": tool_results, "findings": []}

    args_match = re.search(r"Action Input:\s*(.+?)(?:\n\S|\Z)", response, re.DOTALL)
    arguments: dict[str, Any] = {}
    if args_match:
        try:
            parsed = _extract_json(args_match.group(1).strip())
            if isinstance(parsed, dict):
                arguments = parsed
        except (json.JSONDecodeError, ValueError):
            arguments = {}

    # Deduplication
    args_key = json.dumps(arguments, sort_keys=True)
    for prev in tool_results:
        if prev.get("name") == tool_name and json.dumps(prev.get("arguments", {}), sort_keys=True) == args_key:
            return {"answer": "任务已完成。", "tool_results": tool_results, "findings": []}

    started = time.perf_counter()
    try:
        result = tools.call(tool_name, arguments)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        tool_results.append({
            "name": tool_name, "arguments": arguments,
            "result": result, "elapsed_ms": elapsed_ms,
        })
    except (FinanceToolError, TypeError) as err:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        tool_results.append({
            "name": tool_name, "arguments": arguments,
            "error": str(err), "elapsed_ms": elapsed_ms,
        })

    return {"tool_results": tool_results, "findings": []}



def aggregate_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Commander reviews all agent results and aggregates them into a final answer."""
    cfg = _get_configurable(config)
    llm = cfg["llm"]

    user_msg = state["user_message"]
    agent_results = state.get("agent_results", [])

    if not agent_results:
        return {"needs_summary": True}

    # Commander handled it directly: pass through result without re-aggregating
    # Skip reflection since commander's answer is tool-generated and reflection LLM
    # doesn't have tool context — it would flag tool data as hallucination.
    if len(agent_results) == 1 and agent_results[0].get("agent_id") == "commander":
        result_text = agent_results[0].get("result", "")
        if result_text:
            return {"final_answer": result_text, "needs_summary": False, "skip_reflection": True}
        return {"needs_summary": True, "skip_reflection": True}

    # Check if all agents failed
    all_errors = all(r.get("error") for r in agent_results)
    if all_errors:
        errors = "\n".join(
            f"- {r['agent_id']}: {r['error']}" for r in agent_results
        )
        return {
            "final_answer": f"所有领域专家执行失败：\n{errors}",
            "needs_summary": False,
        }

    # Build results summary for Commander
    results_summary = []
    for r in agent_results:
        # Extract media URLs from tool_results to ensure they survive truncation
        media_urls = _extract_media_urls(r.get("tool_results", []))
        result_text = r.get("result", "")
        if media_urls and not any(u in result_text for u in media_urls):
            result_text = media_urls + "\n\n" + result_text
        summary = {
            "agent": r["agent_id"],
            "task": r.get("task", ""),
            "result": result_text[:2000],  # larger limit for media results
            "error": r.get("error", ""),
        }
        results_summary.append(summary)

    # Commander aggregates
    history_context = _build_history_context(cfg.get("history", []))
    system_prompt = COMMANDER_AGGREGATE_PROMPT.format(
        question=user_msg,
        results=json.dumps(results_summary, ensure_ascii=False, indent=2),
    )

    try:
        response = llm.complete(
            system_prompt,
            [{"role": "user", "content": history_context + "请汇总回答。"}],
            temperature=0.4,  # Low temperature for factual aggregation
        )
        final_answer = response.strip()

        # ── Anti-hallucination verification ──
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
        # Fallback: concatenate results
        parts = []
        for r in agent_results:
            if r.get("result"):
                parts.append(f"### {r['agent_id']}\n{r['result']}")
        return {"final_answer": "\n\n".join(parts) if parts else "无法汇总结果。", "needs_summary": False}



def reflection_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Internal quality check: verify → revise if needed → output clean answer."""
    cfg = _get_configurable(config)
    llm = cfg.get("pipeline_llm", cfg["llm"])

    answer = state.get("final_answer", "")
    user_msg = state.get("user_message", "")

    # Skip reflection when commander handled directly with tools
    # (reflection LLM doesn't see tool results and would flag real data as hallucination)
    if state.get("skip_reflection"):
        return {}

    if not answer or not user_msg:
        return {}

    # Skip reflection for very short or skill-execution answers
    if len(answer) < 15 or answer.startswith("技能") or answer.startswith("所有领域专家"):
        return {}

    # Step 1: Evaluate
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

        # Step 2: Revise
        revise_response = llm.complete(
            REVISE_PROMPT.format(
                question=user_msg,
                answer=answer,
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
        pass

    return {}


# ---- High-risk tool patterns ----


def confirm_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Check if any agent result requires human confirmation.

    Uses LangGraph interrupt() to pause execution until user confirms.
    """
    pending = state.get("pending_actions", [])
    if not pending:
        return {"needs_confirmation": False}

    # Already confirmed — proceed
    if state.get("confirmed"):
        logger.debug("hitl: already confirmed, proceeding")
        return {"needs_confirmation": False, "confirmed": True}

    logger.info("hitl: pausing for confirmation, %d actions pending", len(pending))
    # Pause and wait for human input
    decision = interrupt({
        "type": "confirm_required",
        "actions": pending,
    })

    logger.info("hitl: user decision=%s", decision)
    return {
        "needs_confirmation": False,
        "confirmed": True,
        "confirm_decision": decision,
    }
