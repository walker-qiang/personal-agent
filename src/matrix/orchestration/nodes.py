"""Multi-agent orchestration nodes.

Commander + Domain Agents architecture:
  commander_plan → delegate → aggregate → reflection
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langgraph.types import RunnableConfig, interrupt

from ..agent.registry import AgentRegistry
from ..llm import LLMError, FunctionCallResult
from ..tools import FinanceToolError, ToolRegistry
from .state import AgentState

logger = logging.getLogger("matrix.orchestration")

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
- 绝大多数问题返回空数组 []，由指挥官直接处理（指挥官拥有联网搜索、网页抓取工具）
- 投资/金融/持仓/配置分析类问题委派给 investment-analyst
- 图片生成、视频生成、图像创作类问题委派给 media-generator
- 跨领域问题：投资部分委派给 investment-analyst，媒体生成委派给 media-generator，其余指挥官自己处理
- 每个专家只委派一次，合并相似任务
- 如果问题匹配某个专家的技能，填写 skill_name 字段

只返回 JSON 数组，不要其他文字。"""

RAG_NEED_PROMPT = """你是一个路由判断器。判断用户问题是否涉及个人知识库中的领域知识。

个人知识库包含：投资笔记、资产配置、财务记录、交易策略、学习笔记、个人文档等。

用户问题：{question}

只回答 YES 或 NO，不要其他文字。"""

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
- Read the tool descriptions carefully and choose the most appropriate tool for the task
- If a tool can solve the request, DO NOT ask the user questions — just call the tool
- After the tool returns results, summarize them for the user
- If the tool fails, explain the failure and suggest alternatives
- DO NOT call the same tool with nearly identical queries — use the first result

## Image & Video Generation Guidelines
When calling `agnes.generate_image` or `agnes.generate_video`, follow these rules:

**What you do (creative description):**
- Translate the user's intent to English
- Describe the visual content: subject, scene, action, pose, expression, environment
- Describe the composition: camera angle, framing, depth of field, lighting
- Describe the mood and atmosphere: warm/cold, tense/calm, bright/dark
- Be specific and concrete — avoid vague terms like "beautiful" or "nice"
- Keep the prompt under 150 words, focused on visual elements

**What the code handles automatically (do NOT include):**
- Quality keywords: photorealistic, 8k, highly detailed, professional photography
- Negative prompts: no text, no watermark, no distortion, no extra limbs
- Technical specs: resolution, format, rendering engine

**Example:**
- User says "一只猫" → your prompt: "A fluffy orange tabby cat sitting on a wooden windowsill, soft morning light streaming through lace curtains, shallow depth of field focusing on the cat's green eyes, warm cozy atmosphere, dust particles dancing in the light"
- User says "老虎捕猎北极熊" → your prompt: "A Siberian tiger in mid-pounce, muscles tensed, mouth open showing sharp teeth, targeting a polar bear on a snowy Arctic ice field, dramatic overcast sky, snow particles in the air, low camera angle, intense action shot, cold blue-white color palette"

**Style guidance:**
- If the user asks for a specific style (artistic, anime, oil-painting, sketch, 3d-render, watercolor), describe it in the prompt — e.g., "anime style illustration of..."
- Default is photorealistic — no need to mention it explicitly
- For videos, use the default settings (1152x768, 121 frames, 24fps ≈ 5 seconds). Only change if the user asks for specific duration or quality.

**Video generation note:**
- Video generation is asynchronous and takes 2-3 minutes. The tool will wait for completion automatically.
- After calling the tool, show the result with: ![描述](video_url)

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

    LLM also judges whether the question needs RAG (personal knowledge base).
    Only queries RAG when the LLM decides it's relevant.
    """
    cfg = _get_configurable(config)
    llm = cfg.get("pipeline_llm", cfg["llm"])
    agent_registry: AgentRegistry = cfg["agent_registry"]
    retriever = cfg.get("retriever")

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

    # Limit plan steps
    plan = plan[:MAX_PLAN_STEPS]

    # Empty plan = simple question. Always let Commander handle it via ReAct.
    if not plan:
        plan = [
            {"step": 1, "agent_id": "commander", "task": user_msg, "purpose": "直接回答"}
        ]

    # Ensure each step has required fields
    for i, step in enumerate(plan):
        if "step" not in step:
            step["step"] = i + 1
        if "skill_name" not in step:
            step["skill_name"] = ""

    # --- RAG: LLM judges whether to query personal knowledge base ---
    if retriever is not None:
        try:
            need_rag_resp = llm.complete(
                RAG_NEED_PROMPT.format(question=user_msg),
                [{"role": "user", "content": user_msg}],
            )
            need_rag = need_rag_resp.strip().upper().startswith("YES")
        except Exception:
            need_rag = False

        if need_rag:
            try:
                docs = retriever.query(user_msg, top_k=5)
                if docs:
                    context_parts = []
                    for d in docs:
                        content = d.get("content", "")
                        if content:
                            title = d.get("title", "")
                            context_parts.append(f"## {title}\n{content}")
                    rag_context = (
                        "\n\n## 相关文档（来自个人知识库）\n"
                        + "\n\n".join(context_parts)
                    )
                    # Inject RAG context into the first task
                    plan[0]["task"] = plan[0]["task"] + rag_context
                    logger.info(
                        "rag: injected %d docs for '%s'",
                        len(docs), user_msg[:50],
                    )
            except Exception as exc:
                logger.warning("rag: query failed: %s", exc)

    return {
        "delegation_plan": plan,
        "current_step": 0,
    }


def delegate_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Execute one step of the delegation plan.

    Runs a domain agent's ReAct loop to complete the assigned task.
    RAG context (if needed) is already injected by commander_plan_node.
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

    logger.debug(
        "delegate: agent=%s answer_len=%d preview=%s tools=%d",
        agent_id, len(result.get("answer", "")),
        result.get("answer", "")[:80],
        len(result.get("tool_results", [])),
    )

    # Check for high-risk tool calls
    pending_actions = []
    for tr in result.get("tool_results", []):
        tool_name = tr.get("name", "")
        if _is_high_risk(tool_name) and not tr.get("error"):
            pending_actions.append({
                "agent": agent_id,
                "tool": tool_name,
                "args": tr.get("arguments", {}),
                "summary": f"{agent_id} 将调用 {tool_name}",
            })

    return {
        "agent_results": agent_results,
        "tool_results": result.get("tool_results", []),
        "current_step": current_step + 1,
        "react_iteration": 0,
        "needs_confirmation": len(pending_actions) > 0,
        "pending_actions": pending_actions,
    }


def _run_domain_agent_react(
    agent_def: Any,
    task: str,
    tools: ToolRegistry,
    skill_results: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run a ReAct loop for a domain agent using standard multi-turn tool calling.

    Uses OpenAI-compatible format:
      assistant(tool_calls[]) → tool(tool_call_id, content) → assistant(...)

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

    # System prompt
    system_prompt = DOMAIN_AGENT_REACT_SYSTEM.format(
        agent_name=agent_def.name,
        persona=agent_def.persona,
        task=task,
    )

    # Build initial messages — standard multi-turn format
    task_content = history_context + f"请完成以下任务：{task}"
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": task_content},
    ]

    while iteration < MAX_REACT_ITERATIONS:
        iteration += 1

        llm_tools = _build_tools_for_llm(tools)

        try:
            result: FunctionCallResult = llm.function_call(
                system_prompt, messages, llm_tools,
            )
        except (LLMError, ConnectionError, TimeoutError, ValueError, OSError):
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

            # Execute tools and append tool messages with tool_call_id
            executed = 0
            for tc in result.tool_calls:
                if tc.name not in tools.tool_names():
                    continue

                # Dedup check
                args_key = json.dumps(tc.arguments, sort_keys=True)
                if any(
                    prev.get("name") == tc.name
                    and json.dumps(prev.get("arguments", {}), sort_keys=True) == args_key
                    for prev in tool_results
                ):
                    continue

                executed += 1
                started = time.perf_counter()
                logger.debug(
                    "tool_call: tool=%s args=%s",
                    tc.name, json.dumps(tc.arguments, ensure_ascii=False)[:200],
                )
                try:
                    tool_result = tools.call(tc.name, tc.arguments)
                    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                    _trace(cfg, {
                        "ok": True, "tool": tc.name, "arguments": tc.arguments,
                        "elapsed_ms": elapsed_ms, "ts": _now_ts(),
                    })
                    tool_results.append({
                        "name": tc.name, "arguments": tc.arguments,
                        "result": tool_result, "elapsed_ms": elapsed_ms,
                    })
                    # Append tool message (standard format)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })
                except FinanceToolError as err:
                    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                    _trace(cfg, {
                        "ok": False, "tool": tc.name, "arguments": tc.arguments,
                        "error": str(err), "elapsed_ms": elapsed_ms, "ts": _now_ts(),
                    })
                    tool_results.append({
                        "name": tc.name, "arguments": tc.arguments,
                        "error": str(err), "elapsed_ms": elapsed_ms,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": str(err)}, ensure_ascii=False),
                    })

            if executed == 0:
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
                except (LLMError, ConnectionError, TimeoutError, ValueError, OSError):
                    pass
                return {
                    "answer": _fix_media_answer("工具调用已完成，但无法生成总结。", tool_results),
                    "tool_results": tool_results,
                    "findings": [],
                }

            continue  # Loop back — LLM sees tool results as proper messages

        if result.content:
            # Refusal check: if the LLM says it cannot answer but we have tool
            # results, the model ignored the tool data. Append a follow-up
            # message forcing it to use the available data and loop again.
            if _is_refusal(result.content) and tool_results:
                if iteration < MAX_REACT_ITERATIONS:
                    messages.append({
                        "role": "user",
                        "content": (
                            "你刚才说无法提供数据，但实际上工具已经返回了结果。"
                            "请基于以上工具返回的真实数据回答用户的问题。"
                            "直接给出天气信息，不要说你无法提供。"
                        ),
                    })
                    continue
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

    # Max iterations reached
    return {
        "answer": _fix_media_answer("已达到最大分析步数，请基于已有数据回答。", tool_results),
        "tool_results": tool_results,
        "findings": [],
    }


def _extract_media_urls(tool_results: list[dict]) -> str:
    """Extract image/video URLs from tool results as Markdown."""
    lines = []
    for tr in tool_results:
        name = tr.get("name", "")
        result = tr.get("result", {})
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(result, dict):
            continue
        if name == "agnes.generate_image" and result.get("images"):
            for img in result["images"]:
                url = img.get("url", "")
                if url:
                    desc = result.get("prompt", "生成的图片")[:50]
                    lines.append(f"![{desc}]({url})")
        elif name == "agnes.generate_video" and result.get("videos"):
            for vid in result["videos"]:
                url = vid.get("url", "")
                if url:
                    desc = result.get("prompt", "生成的视频")[:50]
                    lines.append(f"![{desc}]({url})")
    return "\n".join(lines)


def _fix_media_answer(answer: str, tool_results: list[dict]) -> str:
    """If the model hallucinates 'can't generate' but tools actually succeeded,
    replace the answer with the actual media results."""
    if not tool_results or not answer:
        return answer
    # Detect "can't do" / "unable" / "sorry" type responses
    negative = ["无法", "不能", "can't", "cannot", "can not", "抱歉", "sorry", "unable", "无法直接"]
    is_negative = any(phrase in answer.lower() for phrase in negative)
    if not is_negative:
        return answer
    # Check if any generation tool actually succeeded
    media_urls = _extract_media_urls(tool_results)
    if not media_urls:
        return answer
    # Replace with positive answer showing the actual results
    lang = "zh" if any("\u4e00" <= c <= "\u9fff" for c in answer) else "en"
    if lang == "zh":
        return f"好的，已成功生成！\n\n{media_urls}"
    return f"Done! Generated successfully.\n\n{media_urls}"


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
        logger.debug("tool_call: tool=%s args=%s", name, json.dumps(args, ensure_ascii=False)[:200])
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

# ---- High-risk tool patterns ----

_HIGH_RISK_PATTERNS = [
    "snapshot.create", "snapshot.update", "snapshot.delete",
    "asset.create", "asset.update", "asset.delete",
    "write", "save", "delete", "create", "update",
    "execute", "run", "deploy",
]


def _is_high_risk(tool_name: str) -> bool:
    """Check if a tool call is high-risk based on its name."""
    return any(pattern in tool_name.lower() for pattern in _HIGH_RISK_PATTERNS)


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