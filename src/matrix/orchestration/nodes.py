"""Multi-agent orchestration nodes.

Commander + Domain Agents architecture:
  commander_plan → delegate → aggregate → reflection
"""

from __future__ import annotations

import json
import logging
import queue
import re
import time
from datetime import datetime, timezone
from typing import Any

from langgraph.types import RunnableConfig, interrupt

from ..agent.registry import AgentRegistry
from ..llm import LLMError, FunctionCallResult
from ..tools import FinanceToolError, ToolRegistry
from .anti_hallucination import verify_all_claims, build_verified_output, VerificationResult
from .state import AgentState

logger = logging.getLogger("matrix.orchestration")

# ---- Limits ----

MAX_REACT_ITERATIONS = 20  # Hard safety net; goal-driven stopping should trigger earlier
MAX_SUBTASK_ITERATIONS = 10  # Per-subtask ReAct limit
MAX_SUBTASKS = 5             # Max subtasks in a decomposition
MAX_PLAN_STEPS = 3
MAX_CONSECUTIVE_FAILURES = 2      # Stop if N consecutive tool calls all fail
MAX_CONSECUTIVE_NO_PROGRESS = 3   # Stop if N consecutive steps add no new info
MAX_SAME_TOOL_CALLS = 2           # Stop if same tool called N+ times (same name, regardless of args)
MAX_TOTAL_TOOL_CALLS = 5          # Stop if total tool calls exceed this (across all tools)
EVALUATOR_INTERVAL = 3            # Run evaluator every N iterations

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
    import re as _re
    score = 0
    for pat in _FACTUAL_PATTERNS:
        if _re.search(pat, question, _re.IGNORECASE):
            score += 1
    if score >= 2:
        return 0.1
    elif score >= 1:
        return 0.4
    return 0.7

# ---- Prompts ----

COMMANDER_PLAN_PROMPT = """你是指挥官 Agent。请制定委派计划来回答用户的问题。

可用的领域专家：
{agents}

用户问题：{question}

请制定执行计划，以 JSON 数组格式返回。每个步骤：
{{"step": 1, "agent_id": "专家ID", "task": "委派给该专家的具体任务（用中文）", "skill_name": "", "purpose": "为什么需要这个专家"}}

规则：
- 简单问题（单步即可回答）返回空数组 []，由指挥官直接处理
- 复杂问题（需要多个独立数据源、多步推理、对比分析）拆分为子任务数组，每个子任务独立运行
  - 例如"对比A和B的财报"拆为：查A财报、查B财报、查股价、汇总分析
  - 每个子任务的 task 必须具体、可独立执行，指定要查什么数据
  - 子任务之间尽量独立无依赖，可并行执行
  - 最多 {max_subtasks} 个子任务
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

## Honesty Rules — READ FIRST
**You MUST NOT fabricate data.** If a tool result does not contain the specific information the user asked for, you MUST clearly state that you could not find it. Fabricating plausible-sounding details is the worst possible failure.

Specifically:
- NEVER invent dates, numbers, statistics, prices, model names, event details, or proper nouns
- If a search result only shows analyst ratings, do NOT pretend it shows live stock prices
- If you cannot find the answer, say "抱歉，搜索结果中未找到该信息" — do NOT make up an answer
- Every factual claim MUST be traceable to a tool result you just received
- If a tool returns a page that requires login/is geo-blocked/has no data, report that honestly

## Tool Usage Rules
- **CRITICAL: Call exactly ONE tool per response. Never call the same tool twice in one step.**
- **CRITICAL: After a tool returns results, use those results. Do NOT call the same tool again with a different query for the same information — the results will be nearly identical.**
- **CRITICAL: STOP AND ANSWER when you have enough information. After each tool call, ask yourself: "Can I fully answer the user's question with the data I already have?" If YES, output the answer immediately.**
- **TIME-SENSITIVE QUERIES: When the user asks for 最近/最新/今天/这次/近期, you MUST use `news_search` (NOT `web_search`). You MUST scan ALL returned results and pick the one with the LATEST date. The first result in the list is NOT necessarily the most recent. If the first result mentions 2025 but a later result mentions 2026-07-06, you MUST cite the 2026 one. Do NOT stop until you have found the most recent event.**
- **KEYWORD HINT: "潜射导弹" IS a type of "洲际导弹" — treat them as the same concept. "潜射弹道导弹" = submarine-launched ballistic missile = 洲际弹道导弹.**
- **CRITICAL: web_fetch only works with real article URLs. If a search result has no URL, use the snippet directly.**
- Read the tool descriptions carefully and choose the most appropriate tool for the task
- If a tool can solve the request, DO NOT ask the user questions — just call the tool
- After the tool returns results, summarize them for the user
- If the tool fails, explain the failure and suggest alternatives
- If you need to search for multiple things, call ONE tool at a time, then decide based on the results

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
- Today is {today}. Never invent dates — only cite dates found in search results.
- Use the same language as the user
- **SOURCE CITATION: Every factual claim (number, date, price, event, quote) MUST be followed by a source tag in the format `[来源: tool_name]`. For example: "腾讯今日收盘价 380 港元 [来源: web_search]" or "据央行公告，利率下调 25 个基点 [来源: news_search]"**
- **If you cannot find a source for a claim, do NOT make the claim. Instead say "搜索结果中未找到该信息"**
- If the tool generated an image, show it using Markdown image syntax: ![描述](URL)
- If the tool generated a video, show it using: ![描述](URL)
- Never use plain text links [text](url) for images/videos — always use ![](url) format
- Use Markdown formatting: **bold** for key figures, `code` for code, bullet lists for breakdowns
- Do NOT include execution process review, agent status tables, or step-by-step workflow in your output
- Money is CNY unless stated otherwise.

## 结构化输出要求（反幻觉）

在回答末尾，你必须附加一个验证块。格式如下：

[VERIFICATION]
[CLAIM] 具体的事实陈述1 [/CLAIM]
[EVIDENCE] 工具返回中支持此陈述的原文 [/EVIDENCE]
[SOURCE] tool_name [/SOURCE]

[CLAIM] 具体的事实陈述2 [/CLAIM]
[EVIDENCE] 工具返回中支持此陈述的原文 [/EVIDENCE]
[SOURCE] tool_name [/SOURCE]
[/VERIFICATION]

规则：
- 你的回答中每个事实性陈述（数字、日期、价格、人名、事件名、百分比）都必须对应一个 CLAIM 条目
- EVIDENCE 必须是工具返回结果中的原文（可截取关键句），不得自行编写
- 如果某个陈述无法在工具结果中找到原文支持，不要写 CLAIM，改为在回答中标注"该信息未在搜索结果中确认"
- 主观判断、总结、建议不需要 CLAIM"""

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
        lines.append(f"[{role_label}]: {h['content'][:300]}")
    return "对话历史：\n" + "\n".join(lines) + "\n\n"


# ---- Goal-driven Evaluator ----

EVALUATOR_PROMPT = """你是一个任务完成度评估器。你的唯一工作是判断：当前收集的信息是否已经足够回答用户的问题。

评估标准：
- SUFFICIENT（充分）：所有回答问题的关键数据已获取，agent 的回答直接针对用户问题，不需要更多工具调用
- INSUFFICIENT（不充分）：关键数据缺失、agent 拒绝回答、答案含糊其辞、或重要事实陈述缺乏工具结果支撑

请只输出一个词：SUFFICIENT 或 INSUFFICIENT，然后输出一行简短原因（中文）。"""


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
        verdict = llm.complete(EVALUATOR_PROMPT, [{"role": "user", "content": eval_prompt}], temperature=0.1)
        verdict_clean = verdict.strip().upper()
        # Validate: must be a meaningful response
        if len(verdict_clean) < 5 or (
            "SUFFICIENT" not in verdict_clean and "INSUFFICIENT" not in verdict_clean
        ):
            logger.warning("Evaluator returned invalid verdict: %s", verdict[:100])
            return _evaluate_heuristic(tool_results, llm_response)
        is_sufficient = verdict_clean.startswith("SUFFICIENT")
        # Extract reason (after the keyword)
        reason = verdict.strip()
        if "\n" in reason:
            reason = reason.split("\n", 1)[-1].strip()
        else:
            # Remove the keyword prefix
            for prefix in ("SUFFICIENT", "INSUFFICIENT"):
                if reason.upper().startswith(prefix):
                    reason = reason[len(prefix):].strip(" ,，:：")
                    break
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

    # --- RAG: LLM judges whether to query personal knowledge base ---
    if retriever is not None:
        try:
            need_rag_resp = llm.complete(
                RAG_NEED_PROMPT.format(question=user_msg),
                [{"role": "user", "content": user_msg}],
                temperature=0.0,
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
        "plan_type": plan_type,
    }


def delegate_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Execute one step of the delegation plan.

    Runs a domain agent's ReAct loop to complete the assigned task.
    RAG context (if needed) is already injected by commander_plan_node.

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


def _push_event(cfg: dict[str, Any], evt_type: str, payload: dict[str, Any]) -> None:
    """Push a real-time event to the SSE queue if available."""
    q = cfg.get("event_queue")
    if q is not None:
        try:
            q.put_nowait((evt_type, payload))
        except queue.Full:
            pass


# ---- Split ReAct nodes (top-level graph, single-step plans) ----

def react_prepare_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Prepare the ReAct context for a single-step domain agent execution.

    Sets up the react dict with system prompt, messages, tools, and
    tracking variables.  Subsequent nodes (react_llm, react_tool,
    react_evaluate) use this context to drive the ReAct loop with
    per-node streaming yields.
    """
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

    system_prompt = DOMAIN_AGENT_REACT_SYSTEM.format(
        agent_name=agent_def.name,
        persona=agent_def.persona,
        task=task,
        today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )

    react = {
        "messages": [{"role": "user", "content": task_content}],
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
    """Call the LLM with function calling to decide the next action.

    Returns updated react context with the LLM response appended to messages.
    If the LLM returns tool_calls, the route sends execution to react_tool_node.
    If the LLM returns content without tool_calls, the route sends to
    react_evaluate_node for sufficiency checking.
    """
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

    try:
        temp = _classify_query_factuality(question)
        result: FunctionCallResult = llm.function_call(system_prompt, messages, llm_tools, temperature=temp)
    except (LLMError, ConnectionError, TimeoutError, ValueError, OSError) as e:
        msg_len = sum(len(str(m.get("content", ""))) for m in messages)
        logger.error("ReAct: LLM call failed in react_llm_node: %s: %s (msg_count=%d, total_chars=%d)", type(e).__name__, str(e)[:200], len(messages), msg_len)
        _push_event(cfg, "progress", {"message": f"LLM 调用失败: {type(e).__name__}"})
        return {
            "react": {**react, "answer": f"无法完成任务，LLM 调用失败: {type(e).__name__}。请重试。", "error": f"LLM call failed: {type(e).__name__}"},
            "error": f"LLM call failed: {type(e).__name__}",
        }

    # Push thinking event
    if result.content:
        _push_event(cfg, "thinking", {"content": result.content.strip()})
    elif result.tool_calls:
        tool_names = [tc.name for tc in result.tool_calls if tc.name in agent_tools.tool_names()]
        if tool_names:
            _push_event(cfg, "thinking", {"content": f"正在调用 {'、'.join(tool_names)} 获取数据..."})

    # Append assistant message
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


def _route_after_react_llm(state: AgentState) -> str:
    """Route after react_llm: tool calls → react_tool, otherwise → react_evaluate."""
    react = state.get("react", {})
    messages = react.get("messages", [])
    if not messages:
        return "react_evaluate"
    last_msg = messages[-1]
    if last_msg.get("tool_calls"):
        return "react_tool"
    return "react_evaluate"


# ---- Shared tool execution (used by both ReAct paths) ----

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
) -> dict:
    """Execute tool calls from LLM response with all guards.

    Shared by both ReAct paths:
    - Top-level: react_tool_node (push_events=False, SSE via state emission)
    - Subgraph:   _run_domain_agent_react (push_events=True, direct SSE)

    Guards applied in order:
    1. Batch-dedup: same (name, args) within a single LLM response
    2. Total-calls: MAX_TOTAL_TOOL_CALLS across all iterations
    3. Same-tool:   MAX_SAME_TOOL_CALLS for the same tool name
    4. Args-dedup:  same (name, args) across all accumulated results

    Returns a dict with all updated state values.
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
            continue

        # Guard 1: batch-dedup
        call_key = (name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
        if call_key in _called_in_batch:
            logger.info("ReAct: batch-dedup skip %s (same args in batch)", name)
            _push_event(cfg, "progress", {"message": f"跳过重复调用 {name}（同批次内相同参数）"})
            continue

        # Guard 2: total-calls
        total_calls = len(accumulated) + len(new_tool_results)
        if total_calls >= MAX_TOTAL_TOOL_CALLS:
            logger.info("ReAct: skipping %s (total %d >= %d)", name, total_calls, MAX_TOTAL_TOOL_CALLS)
            _push_event(cfg, "progress", {"message": f"已收集 {total_calls} 条数据，跳过剩余工具调用"})
            continue

        # Guard 3: same-tool
        same_tool_count = (
            sum(1 for tr in accumulated if tr.get("name") == name)
            + sum(1 for tr in new_tool_results if tr.get("name") == name)
        )
        if same_tool_count >= MAX_SAME_TOOL_CALLS:
            logger.info("ReAct: skipping %s (%d >= %d)", name, same_tool_count, MAX_SAME_TOOL_CALLS)
            _push_event(cfg, "progress", {"message": f"已调用 {name} {same_tool_count} 次，跳过重复调用"})
            continue

        # Guard 4: args-dedup across accumulated + new
        args_key = json.dumps(arguments, sort_keys=True)
        dedup_match = False
        for prev in accumulated + new_tool_results:
            prev_name = prev.get("name", "")
            prev_args = prev.get("arguments", {})
            prev_args_key = json.dumps(prev_args, sort_keys=True)
            if prev_name == name and prev_args_key == args_key:
                dedup_match = True
                break
        if dedup_match:
            logger.info("ReAct: dedup skip %s (same args key=%s...)", name, args_key[:80])
            continue
        else:
            acc_keys = [json.dumps(p.get("arguments", {}), sort_keys=True)[:60] for p in accumulated]
            logger.info("ReAct: dedup no-match for %s args_key=%s accumulated_keys=%s",
                         name, args_key[:80], acc_keys)

        _called_in_batch.add(call_key)

        executed += 1
        started = time.perf_counter()

        if push_events:
            _push_event(cfg, "tool_call", {"name": name, "args": arguments})

        try:
            tool_result = agent_tools.call(name, arguments)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            _trace(cfg, {
                "session_id": session_id,
                "event_type": "tool_call",
                "node_name": node_name,
                "agent_id": agent_id,
                "ok": True, "tool_name": name, "arguments": arguments,
                "result": str(tool_result)[:500],
                "elapsed_ms": elapsed_ms, "ts": _now_ts(),
            })
            new_tool_results.append({
                "name": name, "arguments": arguments,
                "result": tool_result, "elapsed_ms": elapsed_ms,
            })
            new_messages.append({
                "role": "tool",
                "tool_call_id": tc_raw.get("id", ""),
                "content": json.dumps(tool_result, ensure_ascii=False),
            })
            if push_events:
                _push_event(cfg, "tool_result", {"name": name, "result": tool_result})
        except (FinanceToolError, TypeError) as err:
            failed += 1
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            _trace(cfg, {
                "session_id": session_id,
                "event_type": "tool_call",
                "node_name": node_name,
                "agent_id": agent_id,
                "ok": False, "tool_name": name, "arguments": arguments,
                "error": str(err), "elapsed_ms": elapsed_ms, "ts": _now_ts(),
            })
            new_tool_results.append({
                "name": name, "arguments": arguments,
                "error": str(err), "elapsed_ms": elapsed_ms,
            })
            new_messages.append({
                "role": "tool",
                "tool_call_id": tc_raw.get("id", ""),
                "content": json.dumps({"error": str(err)}, ensure_ascii=False),
            })
            if push_events:
                _push_event(cfg, "tool_result", {"name": name, "error": str(err)[:200]})

    # ---- Failure / progress tracking ----
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


def react_tool_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Execute tool calls from the latest assistant message.

    Delegates to _react_execute_tool_calls (shared with subgraph path).
    Adds force_summarize hint for the evaluate node when all calls are deduped.
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

    last_msg = messages[-1]
    tool_calls_raw = last_msg.get("tool_calls", [])
    if not tool_calls_raw:
        return {}

    accumulated = list(state.get("tool_results", []))
    logger.info("ReAct: react_tool_node batch_size=%d accumulated_before=%d", len(tool_calls_raw), len(accumulated))

    result = _react_execute_tool_calls(
        tool_calls_raw=tool_calls_raw,
        agent_tools=agent_tools,
        messages=messages,
        accumulated=accumulated,
        agent_id=agent_id,
        session_id=state.get("session_id", ""),
        cfg=cfg,
        node_name="react_tool",
        consecutive_failures=react.get("consecutive_failures", 0),
        consecutive_no_progress=react.get("consecutive_no_progress", 0),
        prev_result_count=react.get("prev_result_count", 0),
        push_events=False,
    )

    new_messages = result["messages"]
    if result["force_summarize"]:
        new_messages.append({
            "role": "user",
            "content": "所有工具调用均为重复，无需再调用工具。请直接基于以上已有的工具返回结果回答用户的问题。",
        })

    return {
        "react": {
            **react,
            "messages": new_messages,
            "consecutive_failures": result["consecutive_failures"],
            "consecutive_no_progress": result["consecutive_no_progress"],
            "prev_result_count": result["prev_result_count"],
            "force_summarize": result["force_summarize"],
        },
        "tool_results": result["new_tool_results"],
        "tool_call_count": len(result["new_tool_results"]),
    }


def react_evaluate_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Evaluate whether the current results are sufficient to answer.

    Checks early stop conditions and runs the evaluator LLM.  If sufficient,
    returns the final answer.  If insufficient, prompts the LLM to improve.
    """
    cfg = _get_configurable(config)
    llm = cfg["llm"]

    react = state.get("react", {})
    if not react:
        return {}

    # If answer is already set (e.g. error from react_llm_node), route to aggregate
    if react.get("answer"):
        return _build_react_final_answer(react, state.get("tool_results", []), llm, react.get("iteration", 0))

    tool_results = state.get("tool_results", [])
    iteration = react.get("iteration", 0) + 1
    question = react.get("question", "")
    messages = react.get("messages", [])
    consecutive_failures = react.get("consecutive_failures", 0)
    consecutive_no_progress = react.get("consecutive_no_progress", 0)

    # Early stop check
    early_reason = _check_early_stop(
        tool_results, iteration, consecutive_failures, consecutive_no_progress,
    )
    if early_reason:
        logger.info("ReAct early stop at iteration %d: %s", iteration, early_reason)
        return _build_react_final_answer(react, tool_results, llm, iteration)

    # Get the latest LLM response
    last_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            last_content = msg["content"]
            break

    if not last_content:
        # No text content yet — normally this loops back to react_llm_node.
        # But if all tool calls were deduped (force_summarize), the LLM has
        # nothing to act on and will loop forever.  Force final answer instead.
        if react.get("force_summarize"):
            return _build_react_final_answer(react, tool_results, llm, iteration)
        return {"react": {**react, "iteration": iteration}}

    # Refusal check
    if _is_refusal(last_content) and tool_results:
        new_messages = list(messages) + [{
            "role": "user",
            "content": "你刚才说无法提供数据，但实际上工具已经返回了结果。请基于以上工具返回的真实数据回答用户的问题。",
        }]
        return {
            "react": {
                **react,
                "messages": new_messages,
                "iteration": iteration,
                "consecutive_no_progress": consecutive_no_progress + 1,
            },
        }

    # Periodic evaluator — only runs every EVALUATOR_INTERVAL iterations.
    # No always-run evaluator: calling a separate LLM per iteration to judge
    # sufficiency introduces its own errors (false "insufficient" verdicts
    # that trigger unnecessary loops, overriding correct answers).
    if tool_results and iteration > 0 and iteration % EVALUATOR_INTERVAL == 0:
        sufficient, reason = _evaluate_sufficiency(question, tool_results, last_content, llm)
        if sufficient:
            react["iteration"] = iteration
            return _build_react_final_answer(react, tool_results, llm, iteration)

    # Sufficient — build final answer
    return _build_react_final_answer(react, tool_results, llm, iteration)


def _llm_summarize_from_results(
    question: str,
    tool_results: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    llm,
) -> str:
    """Call the LLM to summarize from existing tool results.

    Used when all tool calls were deduped and the LLM has no text content
    (only tool_calls) — we need an explicit summarization call to get a
    text answer the user can read.
    """
    # Build a compact summary of tool results
    result_summary_parts = []
    for i, tr in enumerate(tool_results):
        name = tr.get("name", f"tool_{i}")
        result = tr.get("result", "")
        if result:
            result_str = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
            # Truncate very long results
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "..."
            result_summary_parts.append(f"[{name} #{i+1}]\n{result_str}")

    summary_text = "\n\n".join(result_summary_parts)

    summary_msg = [
        {"role": "system", "content": "你是一个有帮助的助手。请基于提供的工具搜索结果，直接回答用户的问题。不要调用工具，直接回答。使用中文。"},
        {"role": "user", "content": f"用户问题：{question}\n\n以下是工具搜索结果：\n\n{summary_text}\n\n请基于以上结果回答用户的问题。"},
    ]

    try:
        response = llm.invoke(summary_msg)
        return response.content.strip() if hasattr(response, "content") else str(response)
    except Exception:
        logger.exception("_llm_summarize_from_results: LLM call failed")
        return "抱歉，系统暂时无法处理您的问题。请稍后重试。"


def _build_react_final_answer(
    react: dict[str, Any],
    tool_results: list[dict[str, Any]],
    llm,
    iteration: int,
) -> dict[str, Any]:
    """Build the final agent result from the react context."""
    agent_id = react.get("agent_id", "")
    messages = react.get("messages", [])
    question = react.get("question", "")

    # Extract the last assistant content as answer; fall back to react["answer"]
    # (which may be set directly by error paths in react_llm_node)
    answer = react.get("answer", "")
    if not answer:
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                answer = msg["content"]
                break

    # If still no answer but we have tool results, ask the LLM to summarize
    if not answer and tool_results:
        answer = _llm_summarize_from_results(question, tool_results, messages, llm)

    answer = _fix_media_answer(answer, tool_results)

    # ── Anti-hallucination verification ──
    verification = verify_all_claims(answer, tool_results, llm)
    if verification.total > 0:
        answer = build_verified_output(answer, verification)

    new_result = {
        "agent_id": agent_id,
        "task": question,
        "result": answer,
        "findings": [],
        "tool_results": tool_results,
        "error": "",
    }

    return {
        "react": {**react, "iteration": iteration, "answer": answer},
        "agent_results": [new_result],
    }


def _route_after_react_evaluate(state: AgentState) -> str:
    """Route after react_evaluate: loop back to react_llm if not done, else aggregate."""
    react = state.get("react", {})
    if react.get("answer"):
        return "aggregate"
    return "react_llm"


# ---- Subgraph ReAct (multi-step plans, compiled inside delegate_node) ----


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