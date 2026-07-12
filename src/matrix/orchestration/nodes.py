"""Multi-agent orchestration nodes.

Commander + Domain Agents architecture:
  commander_plan → delegate → aggregate → reflection
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langgraph.types import RunnableConfig

from ..agent.registry import AgentRegistry
from ..llm import LLMError, FunctionCallResult
from ..tools import FinanceToolError, ToolRegistry
from .state import AgentState

# ---- Limits ----

MAX_REACT_ITERATIONS = 5
MAX_PLAN_STEPS = 3

# ---- Prompts ----

COMMANDER_PLAN_PROMPT = """你是指挥官 Agent。请制定委派计划来回答用户的问题。

可用的领域专家：
{agents}

用户问题：{question}

请制定执行计划，以 JSON 数组格式返回。每个步骤：
{{"step": 1, "agent_id": "专家ID", "task": "委派给该专家的具体任务（用中文）", "skill_name": "", "purpose": "为什么需要这个专家"}}

规则：
- 绝大多数问题返回空数组 []，由指挥官直接处理（指挥官拥有全部工具，包括联网搜索、网页抓取、图片生成、视频生成等）
- 只有投资/金融/持仓/配置分析类问题才委派给 investment-analyst
- 不要委派给 general-assistant（指挥官自己就是通用助手，拥有相同甚至更多的工具）
- 跨领域问题：投资部分委派给 investment-analyst，其余指挥官自己处理
- 每个专家只委派一次，合并相似任务
- 如果问题匹配某个专家的技能，填写 skill_name 字段

只返回 JSON 数组，不要其他文字。"""

COMMANDER_AGGREGATE_PROMPT = """你是指挥官 Agent。请根据各领域专家的执行结果，汇总回答用户的问题。

用户问题：{question}

专家执行结果：
{results}

请用清晰、结构化的方式汇总回答。要求：
1. 直接回答用户的问题，不要展示执行过程、步骤回顾、专家状态表格
2. 引用专家的关键发现，但不要列出"执行专家""任务目标""执行状态"等元信息
3. 如果某个专家结果不完整或有错误，用一句话说明即可
4. 使用与用户相同的语言
5. 使用 Markdown 格式化：**加粗**关键数字，列表展示要点
6. 如果结果中包含图片 URL，使用 ![描述](URL) 格式展示图片

重要：你的输出是给最终用户看的，不是内部日志。不要包含执行过程回顾。"""

DOMAIN_AGENT_REACT_SYSTEM = """You are {agent_name}, a domain expert with tool access.

{persona}

Current task: {task}

## Tool Usage Rules
- When the user asks to generate an image, draw, or create a picture, you MUST call `agnes.generate_image`
- When the user asks to generate a video or create a video, you MUST call `agnes.generate_video`
- When you need to search for information, call `web_search`
- When you need to fetch a webpage, call `web_fetch`
- When you need investment/holdings data, call the corresponding `finance.*` tool
- If a tool can solve the request, DO NOT ask the user questions — just call the tool
- After the tool returns results, summarize them for the user
- If the tool fails, explain the failure and suggest alternatives

## Image Prompt Optimization
When calling `agnes.generate_image`, you MUST optimize the prompt — do NOT pass the user's raw Chinese text directly. Follow these rules:
- Translate to English (English prompts produce better results)
- Add quality keywords: "photorealistic, highly detailed, 8k, professional photography"
- Add style description: lighting, composition, camera angle, mood, color palette
- Specify what NOT to include: "no text, no watermark, no distortion"
- Be specific about the subject: position, action, expression, environment
- Keep the prompt under 200 words, focused on visual elements
- Example: User says "一只猫" → prompt becomes "A fluffy orange tabby cat sitting on a windowsill, soft morning light streaming through lace curtains, shallow depth of field, photorealistic, highly detailed, 8k, professional photography, warm cozy atmosphere, no text, no watermark"

## Output
- Use the same language as the user
- If the tool generated an image, show it using Markdown image syntax: ![描述](URL)
- If the tool generated a video, show it using: ![描述](URL)
- Never use plain text links [text](url) for images/videos — always use ![](url) format
- Use Markdown formatting: **bold** for key figures, `code` for code, bullet lists for breakdowns
- Do NOT include execution process review, agent status tables, or step-by-step workflow in your output
- Money is CNY unless stated otherwise. Never fabricate data; if tool data is missing, say so."""

REFLECTION_PROMPT = """You are a quality reviewer. Check if the answer below is accurate and complete.

Context: The agent has access to tools including web_search, web_fetch, finance.*, agnes.generate_image (AI image generation), and agnes.generate_video (AI video generation). If the answer mentions generating an image/video with a URL link, that is a REAL tool result — do NOT flag it as hallucination.

User question: {question}
Answer to review: {answer}

Check:
1. Does the answer directly address the question?
2. Are all claims supported by the data (no fabrication)?
3. Is the answer complete (no missing key info)?
4. Is the answer concise and free of hallucination?

Return ONLY a JSON object:
{{"ok": true}} — if the answer is good
{{"ok": false, "issues": ["issue 1", "issue 2"]}} — if there are problems

Do NOT rewrite the answer. Just evaluate."""

REVISE_PROMPT = """You are a helpful AI assistant. Your previous answer had the following issues:

{issues}

Original question: {question}
Original answer: {answer}

Please rewrite the answer to fix these issues. Keep the same language and formatting style.
Return ONLY the corrected answer, no explanations."""


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


def _trace(cfg: dict[str, Any], event: dict[str, Any]) -> None:
    sink = cfg.get("trace")
    if sink is not None:
        sink.record(event)


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _is_refusal(content: str) -> bool:
    """Check if the LLM content is a refusal to use tools."""
    lowered = content.lower()
    refusal_patterns = [
        r"抱歉", r"不能", r"无法", r"做不到", r"目前不",
        r"sorry", r"cannot", r"unable", r"can't", r"don't have",
        r"i apologize", r"i am not able",
    ]
    return any(re.search(pat, lowered) for pat in refusal_patterns)


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
    except (LLMError, ConnectionError, TimeoutError, ValueError, OSError):
        return FunctionCallResult(content="", tool_calls=[])


def _build_tools_for_llm(tools: ToolRegistry) -> list[dict[str, Any]]:
    """Build tool definitions list for LLM function calling."""
    return tools.list_tools()


# ---- Nodes ----


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

    # Build conversation history context for multi-turn awareness
    history = cfg.get("history", [])
    history_context = ""
    if history:
        recent = history[-6:]  # last 3 turns
        lines = []
        for h in recent:
            role_label = "用户" if h["role"] == "user" else "助手"
            lines.append(f"[{role_label}]: {h['content'][:300]}")
        history_context = "对话历史：\n" + "\n".join(lines) + "\n\n"

    try:
        response = llm.complete(
            COMMANDER_PLAN_PROMPT.format(agents=agents_desc, question=user_msg),
            [{"role": "user", "content": history_context + user_msg}],
        )
        plan = _extract_json(response)
        if not isinstance(plan, list):
            plan = []
    except (LLMError, json.JSONDecodeError, ValueError):
        plan = []

    # Filter out any steps that reference non-existent agents (e.g. LLM hallucination)
    valid_ids = {a["id"] for a in agent_registry.agents_for_commander()}
    plan = [s for s in plan if s.get("agent_id", "") in valid_ids]

    # Merge steps for the same agent (e.g. two general-assistant steps → one)
    merged: list[dict[str, Any]] = []
    seen_agents: set[str] = set()
    for s in plan:
        aid = s.get("agent_id", "")
        if aid in seen_agents:
            # Merge task into the existing step
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

    # Limit plan steps
    plan = plan[:MAX_PLAN_STEPS]

    # Empty plan = simple question. Always let Commander handle it via ReAct
    # so that Tool Gate can enforce tool calls when needed.
    if not plan:
        return {
            "delegation_plan": [
                {"step": 1, "agent_id": "commander", "task": user_msg, "purpose": "直接回答"}
            ],
            "current_step": 0,
        }

    # Ensure each step has required fields
    for i, step in enumerate(plan):
        if "step" not in step:
            step["step"] = i + 1
        if "skill_name" not in step:
            step["skill_name"] = ""

    return {
        "delegation_plan": plan,
        "current_step": 0,
    }


def delegate_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Execute one step of the delegation plan.

    Runs a domain agent's ReAct loop to complete the assigned task.
    """
    cfg = _get_configurable(config)
    agent_registry: AgentRegistry = cfg["agent_registry"]
    full_tools: ToolRegistry = cfg["full_tools"]

    plan = state.get("delegation_plan", [])
    current_step = state.get("current_step", 0)
    agent_results = list(state.get("agent_results", []))

    if current_step >= len(plan):
        return {}

    step = plan[current_step]
    agent_id = step.get("agent_id", "")
    task = step.get("task", state["user_message"])
    skill_name = step.get("skill_name", "")

    # Look up agent
    agent_def = agent_registry.get(agent_id)
    if agent_def is None:
        agent_results.append({
            "agent_id": agent_id,
            "task": task,
            "error": f"Agent not found: {agent_id}",
            "findings": [],
        })
        return {
            "agent_results": agent_results,
            "current_step": current_step + 1,
        }

    # Build agent-specific tools and skills
    agent_tools = agent_registry.build_tool_registry(agent_id, full_tools)
    agent_skills = agent_registry.load_skills_for_agent(agent_id)

    # Execute skill if specified
    skill_results = []
    if skill_name:
        skill = next((s for s in agent_skills if getattr(s, "name", "") == skill_name), None)
        if skill is not None:
            from ..skills.executor import execute_skill
            skill_result = execute_skill(skill, agent_tools, cfg.get("trace"))
            skill_results = skill_result.get("results", [])
            if skill_result.get("errors"):
                agent_results.append({
                    "agent_id": agent_id,
                    "task": task,
                    "skill_name": skill_name,
                    "error": "; ".join(skill_result["errors"]),
                    "findings": skill_result.get("findings", []),
                    "tool_results": skill_results,
                })
                return {
                    "agent_results": agent_results,
                    "current_step": current_step + 1,
                }

    # Run domain agent's ReAct loop
    result = _run_domain_agent_react(
        agent_def=agent_def,
        task=task,
        tools=agent_tools,
        skill_results=skill_results,
        cfg=cfg,
    )

    agent_results.append({
        "agent_id": agent_id,
        "task": task,
        "skill_name": skill_name,
        "result": result.get("answer", ""),
        "findings": result.get("findings", []),
        "tool_results": result.get("tool_results", []),
        "error": result.get("error", ""),
    })

    return {
        "agent_results": agent_results,
        "tool_results": result.get("tool_results", []),
        "current_step": current_step + 1,
        "react_iteration": 0,  # reset for next agent
    }


def _run_domain_agent_react(
    agent_def: Any,
    task: str,
    tools: ToolRegistry,
    skill_results: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run a ReAct loop for a domain agent.

    Returns: {"answer": str, "findings": list, "tool_results": list, "error": str}
    """
    llm = cfg["llm"]
    tool_results: list[dict[str, Any]] = list(skill_results)
    iteration = 0

    # Build conversation history context for multi-turn awareness
    history = cfg.get("history", [])
    history_context = ""
    if history:
        recent = history[-6:]  # last 3 turns
        lines = []
        for h in recent:
            role_label = "用户" if h["role"] == "user" else "助手"
            lines.append(f"[{role_label}]: {h['content'][:300]}")
        history_context = "对话历史：\n" + "\n".join(lines) + "\n\n"

    while iteration < MAX_REACT_ITERATIONS:
        iteration += 1

        # Build system prompt
        system_prompt = DOMAIN_AGENT_REACT_SYSTEM.format(
            agent_name=agent_def.name,
            persona=agent_def.persona,
            task=task,
        )

        # Build context
        context = f"Task: {task}\n\n"
        context += history_context
        if tool_results:
            context += (
                "Previous tool results:\n"
                + json.dumps(tool_results, ensure_ascii=False, indent=2)
                + "\n\n"
            )
            # Check for duplicate calls
            seen = set()
            dupes = set()
            failed = set()
            for tr in tool_results:
                key = (tr.get("name"), json.dumps(tr.get("arguments", {}), sort_keys=True))
                if key in seen:
                    dupes.add(key[0])
                seen.add(key)
                if tr.get("error"):
                    failed.add(tr.get("name", ""))
            if dupes:
                context += (
                    "WARNING: The following tools were already called: "
                    + ", ".join(dupes)
                    + ". Do NOT call them again. Use existing results.\n\n"
                )
            if failed:
                context += (
                    "NOTE: The following tools FAILED: "
                    + ", ".join(failed)
                    + ". Do NOT retry them. Explain the failure to the user.\n\n"
                )
        context += "Complete the task based on available data."

        # Try function calling
        try:
            llm_tools = _build_tools_for_llm(tools)
            result: FunctionCallResult = llm.function_call(
                system_prompt, [{"role": "user", "content": context}], llm_tools
            )

            if result.tool_calls:
                executed = _run_tool_calls(result.tool_calls, tool_results, tools, cfg)
                if executed == 0:
                    # All tool calls were duplicates — ask LLM to summarize
                    # existing results (including errors) instead of a hardcoded message.
                    try:
                        summary = llm.complete(
                            system_prompt,
                            [{"role": "user", "content": context + "\nAll tool calls were duplicates. "
                             "Summarize the existing results for the user. If tools failed, explain the failure."}],
                        )
                        return {"answer": summary.strip(), "tool_results": tool_results, "findings": []}
                    except (LLMError, ConnectionError, TimeoutError, ValueError, OSError):
                        return {"answer": "工具调用已完成，但无法生成总结。", "tool_results": tool_results, "findings": []}
                continue

            if result.content:
                # Tool Gate: on first iteration, if the model refuses to call tools
                # but the agent has tools available, force a retry with tool_choice="required".
                if iteration == 1 and _is_refusal(result.content) and llm_tools:
                    retry_result = _force_tool_call(llm, system_prompt, task, llm_tools)
                    if retry_result.tool_calls:
                        _run_tool_calls(retry_result.tool_calls, tool_results, tools, cfg)
                        continue
                    if retry_result.content:
                        return {"answer": retry_result.content.strip(), "tool_results": tool_results, "findings": []}
                return {"answer": result.content.strip(), "tool_results": tool_results, "findings": []}

        except (LLMError, ConnectionError, TimeoutError, ValueError, OSError):
            pass

        # Fallback: regex-based ReAct
        react_result = _domain_react_fallback(agent_def, task, tool_results, tools, llm)
        if react_result.get("answer"):
            return react_result
        if react_result.get("tool_results"):
            tool_results = react_result["tool_results"]
            continue

        # No progress, break
        return {"answer": "无法完成任务，请检查工具和数据。", "tool_results": tool_results, "findings": []}

    # Max iterations reached
    return {
        "answer": "已达到最大分析步数，请基于已有数据回答。",
        "tool_results": tool_results,
        "findings": [],
    }


def _run_tool_calls(
    tool_calls: list,
    tool_results: list[dict],
    tools: ToolRegistry,
    cfg: dict[str, Any],
) -> int:
    """Execute a batch of tool calls. Returns number of NEW calls executed."""
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
                "ok": True, "tool": name, "arguments": args,
                "elapsed_ms": elapsed_ms, "ts": _now_ts(),
            })
            tool_results.append({
                "name": name, "arguments": args,
                "result": result, "elapsed_ms": elapsed_ms,
            })
        except FinanceToolError as err:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            _trace(cfg, {
                "ok": False, "tool": name, "arguments": args,
                "error": str(err), "elapsed_ms": elapsed_ms, "ts": _now_ts(),
            })
            tool_results.append({
                "name": name, "arguments": args,
                "error": str(err), "elapsed_ms": elapsed_ms,
            })

    return executed


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
        response = llm.complete(system_prompt, [{"role": "user", "content": context}])
    except LLMError:
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
    except FinanceToolError as err:
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
    if len(agent_results) == 1 and agent_results[0].get("agent_id") == "commander":
        result_text = agent_results[0].get("result", "")
        if result_text:
            return {"final_answer": result_text, "needs_summary": False}
        return {"needs_summary": True}

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
        summary = {
            "agent": r["agent_id"],
            "task": r.get("task", ""),
            "result": r.get("result", "")[:1000],  # truncate
            "error": r.get("error", ""),
        }
        results_summary.append(summary)

    # Commander aggregates
    system_prompt = COMMANDER_AGGREGATE_PROMPT.format(
        question=user_msg,
        results=json.dumps(results_summary, ensure_ascii=False, indent=2),
    )

    try:
        response = llm.complete(
            system_prompt,
            [{"role": "user", "content": "请汇总回答。"}],
        )
        return {"final_answer": response.strip(), "needs_summary": False}
    except LLMError:
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

    if not answer or not user_msg:
        return {}

    # Skip reflection for very short or skill-execution answers
    if len(answer) < 15 or answer.startswith("技能") or answer.startswith("所有领域专家"):
        return {}

    # Step 1: Evaluate
    try:
        response = llm.complete(
            REFLECTION_PROMPT.format(question=user_msg, answer=answer),
            [{"role": "user", "content": "Evaluate the answer."}],
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
        )
        revised = revise_response.strip()
        if revised and len(revised) > 10:
            return {"final_answer": revised}
    except (LLMError, json.JSONDecodeError, ValueError):
        pass

    return {}