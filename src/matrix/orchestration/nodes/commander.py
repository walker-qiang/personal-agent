"""Commander, aggregate, and reflection nodes.

Commander plans delegation, aggregate combines results, reflection reviews quality.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.types import RunnableConfig

from ...llm import LLMError, LLMClient
from ...agent.registry import AgentRegistry

from ._helpers import (
    _build_history_context,
    _extract_json,
    _extract_media_urls,
    _get_configurable,
    _push_event,
    COMMANDER_AGGREGATE_PROMPT,
    COMMANDER_PLAN_PROMPT,
    MAX_PLAN_STEPS,
    MAX_SUBTASKS,
    REFLECTION_PROMPT,
    REVISE_PROMPT,
)
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
        errors = "\n".join(
            f"- {r['agent_id']}: {r['error']}" for r in agent_results
        )
        return {
            "final_answer": f"所有领域专家执行失败：\n{errors}",
            "needs_summary": False,
        }

    results_summary = []
    for r in agent_results:
        media_urls = _extract_media_urls(r.get("tool_results", []))
        result_text = r.get("result", "")
        if media_urls and not any(u in result_text for u in media_urls):
            result_text = media_urls + "\n\n" + result_text
        summary = {
            "agent": r["agent_id"],
            "task": r.get("task", ""),
            "result": result_text[:2000],
            "error": r.get("error", ""),
        }
        results_summary.append(summary)

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