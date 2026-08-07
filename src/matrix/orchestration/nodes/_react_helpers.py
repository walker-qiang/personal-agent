"""ReAct-specific helpers: final answer building, LLM summarization, and media URL handling.

Split from _helpers.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..anti_hallucination import (
    verify_all_claims, build_verified_output, _strip_all_verification_tags,
)
from ._verification import _heuristic_number_check, _is_empty_tool_result

logger = logging.getLogger("matrix.orchestration")


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

    system_prompt = "你是一个有帮助的助手。请基于提供的工具搜索结果，直接回答用户的问题。不要调用工具，直接回答。使用中文。"
    user_content = f"用户问题：{question}\n\n以下是工具搜索结果：\n\n{summary_text}\n\n请基于以上结果回答用户的问题。"

    try:
        response = llm.complete(system_prompt, [{"role": "user", "content": user_content}])
        return response.strip() if isinstance(response, str) else str(response)
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
        # Pass 1: prefer text-only assistant messages (no tool_calls at all)
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                if msg.get("tool_calls"):
                    continue
                answer = msg["content"]
                break

    # ── P0 guard: if ALL tool results are empty/errors, override ANY answer ──
    # This runs UNCONDITIONALLY — even when the LLM already produced text.
    # The LLM may fabricate data when tools return empty results, so we must
    # intercept BEFORE the answer reaches the user.
    all_empty = False
    if tool_results:
        all_empty = all(
            _is_empty_tool_result(tr)
            for tr in tool_results
        )
        if all_empty:
            answer = "抱歉，当前未能获取到相关数据。请检查查询条件后重试，或尝试使用其他关键词搜索。"
        elif not answer:
            # Only call LLM summarization when we have real data but no answer yet
            answer = _llm_summarize_from_results(question, tool_results, messages, llm)

    # ── P1 guard: heuristic number consistency check ──
    # Even when tools return non-empty results, the LLM may fabricate specific
    # numbers that don't appear in the tool results. This lightweight check
    # extracts numbers from the answer and verifies them against tool results.
    _all_empty = all_empty if tool_results else True
    if answer and tool_results and not _all_empty:
        answer = _heuristic_number_check(answer, tool_results, llm, question)

    answer = _fix_media_answer(answer, tool_results)

    # ── Anti-hallucination verification ──
    pre_verification_answer = answer  # preserve for fallback
    verification = verify_all_claims(answer, tool_results, llm)
    if verification.total > 0:
        answer = build_verified_output(answer, verification)
    else:
        # Always strip ALL verification tags from user-facing output,
        # even when parsing found no claims (e.g. LLM formatted it incorrectly,
        # forgot closing tags, or emitted loose [CLAIM]/[EVIDENCE] tags).
        answer = _strip_all_verification_tags(answer)

    # ── Guard: if verification stripping emptied the answer (e.g. the LLM
    # output was entirely [VERIFICATION] tags with no actual content),
    # fall back to the pre-stripping version so the ReAct result is never
    # empty. The aggregate_node will handle final cleanup.
    if not answer and pre_verification_answer:
        logger.warning(
            "_build_react_final_answer: verification stripping emptied answer, "
            "falling back to pre-verification version (agent=%s)", agent_id,
        )
        answer = pre_verification_answer

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
