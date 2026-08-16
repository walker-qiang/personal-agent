"""Commander, delegate, aggregate, and reflection nodes.

Commander plans delegation, delegate runs domain agent ReAct loops,
aggregate combines results, reflection reviews quality.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from langgraph.types import RunnableConfig

from ...llm import LLMError, LLMClient, FunctionCallResult
from ...tools import FinanceToolError, ToolRegistry
from ...agent.registry import AgentRegistry
from ...skills.router import SemanticRouter, SkillMatch

from ._helpers import (
    _build_history_context,
    _build_react_final_answer,
    _classify_query_factuality,
    _evaluate_sufficiency,
    _extract_media_urls,
    _fix_media_answer,
    _get_configurable,
    _heuristic_number_check,
    _inject_agent_guidelines,
    _inject_data_index,
    _inject_lessons,
    _inject_working_memory,
    _is_high_risk,
    _is_refusal,
    _push_event,
    _requires_browser,
    _today_cn,
    COMMANDER_AGGREGATE_PROMPT,
    COMMANDER_PLAN_PROMPT,
    DOMAIN_AGENT_REACT_SYSTEM,
    FALLBACK_AGGREGATE_PROMPT,
    LESSON_EXTRACTION_PROMPT,
    MAX_PLAN_STEPS,
    MAX_REACT_ITERATIONS,
    MAX_SUBTASKS,
    PREFLECT_PROMPT,
    REFLECTION_PROMPT,
    REFLEXION_PROMPT,
    REPLAN_PROMPT,
    REVISE_PROMPT,
)
from ..anti_hallucination import verify_all_claims, build_verified_output, _strip_all_verification_tags
from ..state import AgentState

logger = logging.getLogger("matrix.orchestration")

# ── Semantic router singleton (L1 routing) ──────────────────────────────────

_semantic_router: SemanticRouter | None = None
_semantic_router_skills_key: str = ""


def _get_semantic_router(
    agent_registry: AgentRegistry,
    threshold_override: float | None = None,
) -> SemanticRouter | None:
    """Lazily build a SemanticRouter from the registry's skills.

    The router is cached by a key derived from skill names + descriptions
    so that it rebuilds when skills are reloaded. The threshold can be
    overridden per-call from the configurable dict; when the override
    changes, the router's threshold is updated without rebuilding the index.
    """
    global _semantic_router, _semantic_router_skills_key

    try:
        all_skills = agent_registry.list_all_skills()
    except (FileNotFoundError, OSError):
        return None

    # Build a cache key from skill names + descriptions
    key = "|".join(
        f"{s.name}:{s.description[:50]}" for s in all_skills
    )
    if key == _semantic_router_skills_key and _semantic_router is not None:
        # Update threshold if overridden and different
        if threshold_override is not None and _semantic_router.threshold != threshold_override:
            _semantic_router.threshold = threshold_override
        return _semantic_router

    if not all_skills:
        _semantic_router = None
        _semantic_router_skills_key = ""
        return None

    # Lazy import to avoid circular dependency / heavy startup
    try:
        from ...rag.embedder import LocalEmbedder
        embedder = LocalEmbedder()
    except Exception as exc:
        logger.warning("semantic router: embedder init failed: %s", exc)
        return None

    _semantic_router = SemanticRouter(
        all_skills, embedder, threshold=threshold_override,
    )
    _semantic_router_skills_key = key
    return _semantic_router


def _find_skill_owner(agent_registry: AgentRegistry, skill_name: str) -> str:
    """Find which agent owns a skill by name. Returns agent_id or ''."""
    for agent_def in agent_registry._agents.values():
        try:
            agent_skills = agent_registry.load_skills_for_agent(agent_def.id)
        except (FileNotFoundError, OSError):
            continue
        if any(s.name == skill_name for s in agent_skills):
            return agent_def.id
    return ""


def _build_skill_plan(user_msg: str, skill_name: str, agent_id: str) -> dict[str, Any]:
    """Build a single-step plan that delegates to a skill."""
    return {
        "delegation_plan": [{
            "step": 1,
            "agent_id": agent_id,
            "task": user_msg,
            "skill_name": skill_name,
            "depends_on": [],
            "output_key": "skill_result",
            "purpose": f"使用 {skill_name} 技能处理",
        }],
        "current_step": 0,
        "plan_type": "agent",
    }


def commander_plan_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Commander plans the delegation strategy. Entry node of the graph.

    This is the ONLY node that calls the LLM for intent classification.
    Empty plan [] = simple question (Commander handles directly).
    Non-empty plan = delegate to domain experts.
    """
    cfg = _get_configurable(config)
    llm = cfg.get("pipeline_llm", cfg["llm"])
    agent_registry: AgentRegistry = cfg["agent_registry"]
    full_tools: ToolRegistry = cfg["full_tools"]

    user_msg = state["user_message"]

    if not user_msg.strip():
        return {
            "delegation_plan": [
                {"step": 1, "agent_id": "commander", "task": "回复空消息", "purpose": "直接回答"}
            ],
            "current_step": 0,
        }

    # ── Tiered skill routing: L0 keyword → L1 semantic → L2 LLM plan ──────
    # L0: Keyword match (zero cost, handles exact keywords like "组合复盘")
    # L1: Semantic match (embedding cosine sim, handles synonyms like "看看配置")
    # L2: LLM plan (fallback for complex/multi-step queries)
    try:
        all_skills = agent_registry.list_all_skills()
    except (FileNotFoundError, OSError):
        all_skills = []

    # ── L0: keyword match (longest-match scoring) ──────────────────────
    # Instead of first-match, collect all matching skills and pick the one
    # with the longest matching keyword. This prevents "复盘" (2 chars from
    # personal-reflection) from beating "组合复盘" (4 chars from portfolio-review).
    import re as _re
    l0_match = None
    l0_best_len = 0
    q_lower = user_msg.lower()
    for skill in all_skills:
        text = (skill.title + " " + skill.description).lower()
        words = [w for w in _re.split(r"[\s,，。！？、；：""''（）()]+", text) if len(w) >= 2]
        for w in words:
            if w in q_lower and not skill._has_negation(user_msg, w):
                if len(w) > l0_best_len:
                    l0_best_len = len(w)
                    l0_match = skill
                break  # one match per skill is enough; we want the longest overall

    if l0_match:
        owner = _find_skill_owner(agent_registry, l0_match.name)
        if owner:
            logger.info(
                "commander: L0 keyword match — skill=%s, agent=%s",
                l0_match.name, owner,
            )
            return _build_skill_plan(user_msg, l0_match.name, owner)
        logger.warning("commander: L0 skill %s matched but no agent owns it", l0_match.name)

    # ── L1: semantic match ──
    if not l0_match:
        threshold_override = cfg.get("semantic_threshold")
        router = _get_semantic_router(agent_registry, threshold_override=threshold_override)
        if router is not None and router.is_available:
            best = router.best_match(user_msg)
            if best and router.should_accept(best):
                owner = _find_skill_owner(agent_registry, best.skill_name)
                if owner:
                    logger.info(
                        "commander: L1 semantic match — skill=%s, score=%.3f, agent=%s",
                        best.skill_name, best.score, owner,
                    )
                    return _build_skill_plan(user_msg, best.skill_name, owner)
                logger.warning(
                    "commander: L1 skill %s matched (score=%.3f) but no agent owns it",
                    best.skill_name, best.score,
                )
            elif best:
                logger.info(
                    "commander: L1 best match %s score=%.3f below threshold %.2f, falling through to LLM",
                    best.skill_name, best.score, router.threshold,
                )

    agents_desc = json.dumps(
        agent_registry.agents_for_commander(full_tools), ensure_ascii=False, indent=2
    )

    history_context = _build_history_context(cfg.get("history", []))

    try:
        plan = llm.complete_json(
            COMMANDER_PLAN_PROMPT.format(agents=agents_desc, question=user_msg, max_subtasks=MAX_SUBTASKS),
            [{"role": "user", "content": history_context + user_msg}],
            temperature=0.1,
        )
        if not isinstance(plan, list):
            plan = []
    except (LLMError, json.JSONDecodeError, ValueError) as e:
        logger.warning("commander_plan LLM/parse failed: %s", type(e).__name__)
        plan = []

    valid_ids = {a["id"] for a in agent_registry.agents_for_commander()}
    valid_ids.add("commander")
    plan = [s for s in plan if s.get("agent_id", "") in valid_ids]

    logger.info("commander: raw_plan=%s", json.dumps(plan, ensure_ascii=False)[:500])

    # DAG plans have explicit depends_on relationships that must be preserved.
    # Only merge consecutive same-agent steps without depends_on (legacy optimization).
    has_dag_deps = any(s.get("depends_on") for s in plan)
    if not has_dag_deps and len(plan) > 1:
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

    # ── PreFlect: pre-execution critique for multi-step plans ──────────────
    # Only trigger for multi-step plans (len > 1) — single-step plans are
    # already optimized by L0/L1 skill routing and don't need critique.
    original_plan = list(plan)  # Save for ToT comparison
    prelect_revised = False
    if len(plan) > 1:
        try:
            plan_json = json.dumps(plan, ensure_ascii=False)
            critique = llm.complete_json(
                PREFLECT_PROMPT.format(question=user_msg, plan=plan_json),
                [{"role": "user", "content": user_msg}],
                temperature=0.0,
            )

            if critique.get("needs_revision") and critique.get("adjusted_plan"):
                adjusted = critique["adjusted_plan"]
                if isinstance(adjusted, list) and adjusted:
                    # Validate adjusted plan: same agent_ids
                    adjusted_valid = all(
                        s.get("agent_id", "") in valid_ids for s in adjusted
                    )
                    if adjusted_valid:
                        logger.info(
                            "commander: PreFlect revised plan — issues: %s",
                            critique.get("issues", []),
                        )
                        plan = adjusted[:MAX_PLAN_STEPS]
                        # Recompute plan_type
                        adjusted_agent_ids = {s.get("agent_id", "") for s in plan}
                        is_subtask = len(plan) > 1 and len(adjusted_agent_ids) == 1
                        plan_type = "subtask" if is_subtask else "agent"
                        prelect_revised = True
                    else:
                        logger.warning(
                            "commander: PreFlect adjusted plan has invalid agent_ids, keeping original"
                        )
                else:
                    logger.info(
                        "commander: PreFlect found issues but no adjusted plan — keeping original. issues: %s",
                        critique.get("issues", []),
                    )
            else:
                logger.info("commander: PreFlect approved plan (no revision needed)")
        except (LLMError, json.JSONDecodeError, ValueError) as e:
            logger.warning("commander: PreFlect failed: %s", type(e).__name__)

    # ── ToT: compare original vs revised plan when PreFlect changed it ──────
    if prelect_revised and original_plan != plan:
        try:
            from ..tot import TreeSearchEngine
            tot_engine = TreeSearchEngine(llm=llm)
            if tot_engine.should_use_tot(user_msg, len(plan)):
                tot_result = tot_engine.select_best_plan(
                    user_msg, [original_plan, plan]
                )
                if tot_result.used_tot and tot_result.best_path:
                    best = tot_result.best_path[0]
                    if best.result == original_plan:
                        logger.info(
                            "commander: ToT selected original plan (score=%.2f) over revised",
                            best.value,
                        )
                        plan = original_plan
                        adjusted_agent_ids = {s.get("agent_id", "") for s in plan}
                        is_subtask = len(plan) > 1 and len(adjusted_agent_ids) == 1
                        plan_type = "subtask" if is_subtask else "agent"
                    else:
                        logger.info(
                            "commander: ToT confirmed revised plan (score=%.2f)",
                            best.value,
                        )
        except Exception as e:
            logger.warning("commander: ToT plan selection failed: %s", type(e).__name__)

    # P2: Emit plan creation progress event
    if len(plan) > 1:
        _push_event(cfg, "progress", {
            "type": "plan_created",
            "plan_type": plan_type,
            "total_steps": len(plan),
            "steps": [
                {"step": s.get("step", 0), "agent": s.get("agent_id", ""),
                 "task": s.get("task", "")[:60], "depends_on": s.get("depends_on", [])}
                for s in plan
            ],
            "message": f"已制定 {len(plan)} 步执行计划，开始执行...",
        })

    for i, step in enumerate(plan):
        if "step" not in step:
            step["step"] = i + 1
        if "skill_name" not in step:
            step["skill_name"] = ""
        if "depends_on" not in step:
            step["depends_on"] = []
        if "output_key" not in step:
            step["output_key"] = f"step_{step.get('step', i + 1)}"

    return {
        "delegation_plan": plan,
        "current_step": 0,
        "plan_type": plan_type,
    }


# ── Nested Agent-as-Tool compatibility boundary ─────────────────────────────

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
    """Compatibility entry point; nested execution is Runtime-managed."""
    del skill_results, agent_id
    if _requires_browser(task):
        browser_tools = [
            name for name in tools.tool_names()
            if name.startswith("mcp_browser_")
        ]
        if not browser_tools:
            return {
                "answer": (
                    "当前运行环境未配置浏览器 MCP，无法执行浏览器打开、动态页面提取或页面交互。"
                    "请启用 browser MCP 后重试。"
                ),
                "tool_results": [],
                "findings": [],
                "environment_blocked": True,
            }

    from ..runtime_adapter import run_nested_agent_runtime

    nested_cfg = dict(cfg)
    if session_id:
        nested_cfg["session_id"] = session_id
    result = run_nested_agent_runtime(
        agent_def=agent_def,
        task=task,
        agent_tools=tools,
        cfg=nested_cfg,
        max_iterations=max_iterations,
    )
    return {
        "answer": result.get("answer", ""),
        "tool_results": result.get("tool_results", []),
        "findings": [],
        "operation_id": result.get("operation_id", ""),
        **({"error": result["error"]} if result.get("error") else {}),
    }



# ── Plan-and-Execute: DAG router + Replan ────────────────────────────────────

MAX_REPLAN_ATTEMPTS = 2


def _get_ready_steps(plan: list[dict[str, Any]], completed: list[int]) -> list[dict[str, Any]]:
    """Find steps whose dependencies are all satisfied.

    A step is "ready" if all steps in its depends_on list have been completed.
    Steps that are already in completed_steps are excluded.
    """
    completed_set = set(completed)
    ready = []
    for s in plan:
        step_num = s.get("step", 0)
        if step_num in completed_set:
            continue
        deps = s.get("depends_on", [])
        if all(d in completed_set for d in deps):
            ready.append(s)
    return ready


def replan_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Check execution progress and decide whether the plan needs revision.

    Called after each batch of delegate nodes completes.  Evaluates:
    1. Are results deviating from expectations?
    2. Did any step fail?
    3. Are remaining steps still valid given what we've learned?

    Returns:
        needs_replan=True → commander_plan regenerates the plan.
        needs_replan=False → DAG router continues with next batch or aggregate.
    """
    cfg = _get_configurable(config)
    llm = cfg.get("pipeline_llm", cfg["llm"])

    plan = state.get("delegation_plan", [])
    completed = state.get("completed_steps", [])
    results = state.get("agent_results", [])
    replan_attempts = state.get("replan_attempts", 0)

    # Safety: if no plan or no completed steps, skip replan
    if not plan or not completed:
        return {"needs_replan": False}

    # Safety: prevent infinite replan loops
    if replan_attempts >= MAX_REPLAN_ATTEMPTS:
        logger.warning("replan: max attempts (%d) reached, forcing continue", MAX_REPLAN_ATTEMPTS)
        return {"needs_replan": False}

    # Build summary of completed steps
    completed_summary = []
    for s in plan:
        sn = s.get("step", 0)
        if sn in completed:
            # Find matching result
            matched = [r for r in results if r.get("task") == s.get("task")]
            completed_summary.append({
                "step": sn,
                "task": s.get("task", ""),
                "result_preview": (matched[0].get("result", "")[:200] + "...") if matched else "(no result)",
                "error": matched[0].get("error", "") if matched else "",
            })

    try:
        assessment = llm.complete_json(
            REPLAN_PROMPT.format(
                plan=json.dumps(plan, ensure_ascii=False, indent=2),
                completed=json.dumps(completed_summary, ensure_ascii=False, indent=2),
                goal=state.get("user_message", ""),
            ),
            [],
            temperature=0.1,
        )
        if not isinstance(assessment, dict):
            assessment = {}
    except (LLMError, json.JSONDecodeError, ValueError) as e:
        logger.warning("replan LLM/parse failed: %s", type(e).__name__)
        return {"needs_replan": False}

    needs_revision = assessment.get("needs_revision", False)
    revised_plan = assessment.get("revised_plan", [])

    if needs_revision and revised_plan:
        logger.info("replan: plan needs revision — %s", assessment.get("reason", "no reason given"))
        # P2: Emit replan progress event
        _push_event(cfg, "progress", {
            "type": "replan",
            "reason": assessment.get("reason", "需要修正计划"),
            "attempt": replan_attempts + 1,
            "message": f"执行计划需要修正（第 {replan_attempts + 1} 次），正在重新规划...",
        })
        # Normalize revised plan: ensure depends_on and output_key fields
        for i, s in enumerate(revised_plan):
            if "step" not in s:
                s["step"] = i + 1
            if "depends_on" not in s:
                s["depends_on"] = []
            if "output_key" not in s:
                s["output_key"] = f"step_{s.get('step', i + 1)}"
            if "skill_name" not in s:
                s["skill_name"] = ""
        return {
            "delegation_plan": revised_plan,
            "needs_replan": True,
            "replan_attempts": replan_attempts + 1,
        }

    if needs_revision:
        logger.info("replan: needs_revision=true but no revised_plan provided, skipping")

    return {"needs_replan": False}


# ── Aggregate ────────────────────────────────────────────────────────────────

def aggregate_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Commander reviews all agent results and aggregates them into a final answer.

    On Reflexion retry, injects self-reflection memory as additional context
    so the LLM can produce a better answer informed by past mistakes.
    """
    cfg = _get_configurable(config)
    llm = cfg["llm"]

    user_msg = state["user_message"]
    agent_results = state.get("agent_results", [])

    # On reflexion retry, inject reflection memory into the prompt
    reflexion_memory = state.get("reflexion_memory", [])
    is_retry = state.get("needs_reflexion_retry", False)

    if not agent_results:
        result = {"needs_summary": True}
        if is_retry:
            result["needs_reflexion_retry"] = False  # clear retry flag
        return result

    if len(agent_results) == 1 and agent_results[0].get("agent_id") == "commander":
        result_text = agent_results[0].get("result", "")
        if result_text:
            result = {"final_answer": result_text, "needs_summary": False, "skip_reflection": True}
        else:
            result = {"needs_summary": True, "skip_reflection": True}
        if is_retry:
            result["needs_reflexion_retry"] = False
        return result

    all_errors = all(r.get("error") for r in agent_results)
    if all_errors:
        # ── Graceful degradation: use LLM to generate a friendly fallback message ──
        pipeline_llm = cfg.get("pipeline_llm", cfg["llm"])
        attempts_lines = []
        error_lines = []
        for r in agent_results:
            task_desc = r.get("task", "未知任务")[:80]
            attempts_lines.append(f"- {task_desc}")
            err = r.get("error", "未知错误")[:120]
            error_lines.append(f"- {err}")
        attempts_text = "\n".join(attempts_lines) if attempts_lines else "（无记录）"
        errors_text = "\n".join(error_lines) if error_lines else "（无记录）"

        try:
            fallback_msg = pipeline_llm.complete(
                FALLBACK_AGGREGATE_PROMPT.format(
                    question=user_msg,
                    attempts=attempts_text,
                    errors=errors_text,
                ),
                [{"role": "user", "content": "请生成一条友好的回复。"}],
                temperature=0.4,
            )
            fallback_msg = fallback_msg.strip() if isinstance(fallback_msg, str) else ""
        except (LLMError, ConnectionError, TimeoutError, ValueError, OSError) as e:
            logger.warning("aggregate_node fallback LLM failed: %s", type(e).__name__)
            fallback_msg = ""

        if not fallback_msg or len(fallback_msg) < 10:
            # Hard fallback: compose a simple message without internal details
            fallback_msg = (
                "抱歉，暂时无法完成您的请求。\n\n"
                "可能的原因：网络波动或服务暂时不可用。\n\n"
                "建议：\n"
                "- 稍等片刻后重新提问\n"
                "- 尝试换一种方式描述您的问题"
            )

        result = {"final_answer": fallback_msg, "needs_summary": False, "skip_reflection": True}
        if is_retry:
            result["needs_reflexion_retry"] = False
        return result

    # ── Guard: all agent results are empty (no text, no error) ──────────
    # This happens when ReAct loops complete but verification stripping
    # removed all content. Skip LLM call — there's nothing to summarize.
    all_empty = all(
        not r.get("result") and not r.get("error")
        for r in agent_results
    )
    if all_empty:
        logger.warning(
            "aggregate_node: all %d agent results are empty, skipping LLM call",
            len(agent_results),
        )
        result = {
            "final_answer": "抱歉，当前未能获取到相关数据。请检查查询条件后重试，或尝试使用其他关键词搜索。",
            "needs_summary": False,
            "skip_reflection": True,
        }
        if is_retry:
            result["needs_reflexion_retry"] = False
        return result

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

    # Build system prompt, injecting reflection memory on retry
    if is_retry and reflexion_memory:
        reflections_text = "\n".join(f"- {r}" for r in reflexion_memory)
        system_prompt = (
            COMMANDER_AGGREGATE_PROMPT.format(
                today=_today_cn(),
                question=user_msg,
                results=json.dumps(results_summary, ensure_ascii=False, indent=2),
            )
            + "\n\n## Self-Reflection (from previous attempt)\n"
            + "Your previous answer had issues. Here is what you learned:\n"
            + reflections_text
            + "\n\nAddress these issues in your new answer."
        )
    else:
        system_prompt = COMMANDER_AGGREGATE_PROMPT.format(
            today=_today_cn(),
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

        if not final_answer:
            logger.warning("aggregate_node: LLM returned empty response")

        all_tool_results = []
        for r in agent_results:
            all_tool_results.extend(r.get("tool_results", []))
        if all_tool_results and final_answer:
            verification = verify_all_claims(final_answer, all_tool_results, llm)
            if verification.total > 0:
                final_answer = build_verified_output(final_answer, verification)
            else:
                final_answer = _strip_all_verification_tags(final_answer)
                # P1 fallback: heuristic number check when no [VERIFICATION] blocks
                # Use main agent LLM (DeepSeek V4 Flash) for regex-based verification
                final_answer = _heuristic_number_check(
                    final_answer, all_tool_results, cfg["llm"], user_msg,
                )

        # ── Guard: final_answer is empty (LLM returned empty, or
        # verification stripping removed all content) ──
        # Compose a fallback from the raw agent results so the user gets
        # *something* rather than the generic "暂时无法生成回复" error.
        if not final_answer:
            logger.warning(
                "aggregate_node: final_answer empty after processing, "
                "composing fallback from %d agent results", len(agent_results),
            )
            parts = []
            for r in agent_results:
                if r.get("result"):
                    parts.append(r["result"])
            final_answer = "\n\n".join(parts) if parts else (
                "抱歉，当前无法生成汇总回复，请稍后重试或换一种方式提问。"
            )

        result = {"final_answer": final_answer, "needs_summary": False}
        if is_retry:
            result["needs_reflexion_retry"] = False  # clear retry flag for next cycle
        return result
    except LLMError as e:
        logger.error("aggregate_node LLM failed: %s", type(e).__name__)
        parts = []
        for r in agent_results:
            if r.get("result"):
                parts.append(f"### {r['agent_id']}\n{r['result']}")
        result = {"final_answer": "\n\n".join(parts) if parts else "无法汇总结果。", "needs_summary": False}
        if is_retry:
            result["needs_reflexion_retry"] = False
        return result


# ── Reflection (Reflexion loop) ─────────────────────────────────────────────

def reflection_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Reflexion quality gate: evaluate → self-reflect → retry or finalize.

    When the answer is insufficient and retry budget remains, this node
    generates a self-reflection (lessons learned), stores it in state,
    and signals the graph to route back to ``aggregate`` for a retry with
    the reflection as additional context.

    When the answer is good or retries are exhausted, it applies a final
    best-effort revision and returns the cleaned answer.
    """
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

    max_attempts = state.get("reflexion_max", 0)
    current_attempt = state.get("reflexion_attempts", 0)

    try:
        history_context = _build_history_context(cfg.get("history", []))
        data = llm.complete_json(
            REFLECTION_PROMPT.format(question=user_msg, answer=answer),
            [{"role": "user", "content": history_context + "Evaluate the answer."}],
            temperature=0.1,
        )
        if not isinstance(data, dict) or data.get("ok") is not False:
            return {}  # Answer is acceptable

        issues = data.get("issues", [])
        if not issues:
            return {}

        # ── P3: Extract cross-session lesson from failure ──────────────
        _extract_and_store_lesson(cfg, llm, user_msg, answer, issues)

        # ---- Reflexion loop: retry with self-reflection ----
        if current_attempt < max_attempts:
            # Generate self-reflection
            prior = state.get("reflexion_memory", [])
            prior_text = (
                "Previous reflections:\n" + "\n".join(f"- {r}" for r in prior)
                if prior else "No prior reflections."
            )

            reflection = llm.complete(
                REFLEXION_PROMPT.format(
                    question=user_msg,
                    answer=answer,
                    issues="\n".join(f"- {i}" for i in issues),
                    prior_reflections=prior_text,
                ),
                [{"role": "user", "content": "Write your self-reflection."}],
                temperature=0.2,
            ).strip()

            if reflection:
                _push_event(cfg, "reflexion", {
                    "attempt": current_attempt + 1,
                    "issues": issues,
                    "reflection": reflection[:200],
                })
                logger.info(
                    "reflexion_retry: attempt=%d/%d issues=%d",
                    current_attempt + 1, max_attempts, len(issues),
                )
                return {
                    "reflexion_attempts": current_attempt + 1,
                    "reflexion_memory": prior + [reflection],
                    # Signal graph to route back to aggregate for retry
                    "needs_reflexion_retry": True,
                }

        # ---- Max retries exhausted: best-effort revision ----
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
        logger.warning("reflection_node LLM failed: %s", type(e).__name__)

    return {}


def _extract_and_store_lesson(
    cfg: dict[str, Any],
    llm: Any,
    user_msg: str,
    answer: str,
    issues: list[str],
) -> None:
    """P3: Extract a cross-session lesson from a failed answer and persist it.

    Uses the LLM to generate a structured lesson from the failure context,
    then stores it in LessonStore for future sessions. Best-effort: failures
    in extraction or storage are logged but never propagated.
    """
    lesson_store = cfg.get("lesson_store")
    if lesson_store is None:
        return

    try:
        lesson_data = llm.complete_json(
            LESSON_EXTRACTION_PROMPT.format(
                question=user_msg[:500],
                answer=answer[:1000],
                issues="\n".join(f"- {i}" for i in issues[:5]),
            ),
            [{"role": "user", "content": "Extract the lesson."}],
            temperature=0.0,
        )

        if not isinstance(lesson_data, dict):
            return

        task_pattern = str(lesson_data.get("task_pattern", "")).strip()
        failure_type = str(lesson_data.get("failure_type", "incomplete")).strip()
        lesson_text = str(lesson_data.get("lesson_text", "")).strip()
        severity = str(lesson_data.get("severity", "medium")).strip()

        if not lesson_text:
            return

        # Determine agent_id from plan
        plan = cfg.get("delegation_plan", [])
        agent_id = ""
        if plan:
            current_step = cfg.get("current_step", 0)
            if current_step < len(plan):
                agent_id = plan[current_step].get("agent_id", "")

        user_id = cfg.get("user_id", "")

        lesson_store.record_lesson(
            task_pattern=task_pattern or user_msg[:100],
            failure_type=failure_type,
            lesson_text=lesson_text,
            agent_id=agent_id,
            user_id=user_id,
            severity=severity,
        )
        logger.info(
            "lesson_extracted: type=%s agent=%s pattern=%s",
            failure_type, agent_id, task_pattern[:50],
        )
    except (LLMError, json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        logger.debug("lesson_extraction failed: %s", type(e).__name__)
