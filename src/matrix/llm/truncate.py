"""Message truncation utilities for token budget management.

Estimates token count from character count and truncates message lists
to fit within a maximum token budget. Uses a conservative heuristic:
- Chinese characters: ~1.5 tokens per char
- ASCII/English: ~0.25 tokens per char (roughly 4 chars per token)
"""

from __future__ import annotations

import re
from typing import Any


# Conservative token estimation: count Chinese and non-Chinese characters separately
_CHINESE_CHAR = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_OTHER_CHAR = re.compile(r"[^\u4e00-\u9fff\u3400-\u4dbf\s]")

# Smallest suffix kept when the budget cannot fit any message.
_MIN_RECENT_MESSAGES = 6


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count.

    Chinese: ~1.5 tokens per character (conservative, actual is closer to 1.0-1.2)
    Other characters: ~0.25 tokens per character (roughly 4 chars per token)
    """
    chinese = len(_CHINESE_CHAR.findall(text))
    other = len(_OTHER_CHAR.findall(text))
    return int(chinese * 1.5 + other * 0.25)


def _msg_content(msg: dict[str, Any]) -> str:
    """Extract text content from a message dict for token estimation."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multi-modal content blocks: extract text parts, mark images
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "image_url":
                    parts.append("[图片]")
                elif b.get("type") == "image":
                    parts.append("[图片]")
        return " ".join(parts)
    return str(content)


def _msg_tokens(msg: dict[str, Any]) -> int:
    """Estimate tokens for a message dict."""
    return estimate_tokens(_msg_content(msg))


def truncate_messages(
    messages: list[dict[str, Any]],
    system_prompt: str = "",
    max_tokens: int = 8000,
    reserve_tokens: int = 2000,
) -> list[dict[str, Any]]:
    """Truncate message history to fit within a token budget.

    Args:
        messages: List of role/content messages, oldest first.
        system_prompt: System prompt to reserve tokens for.
        max_tokens: Maximum total tokens allowed.
        reserve_tokens: Tokens to reserve for the model's response.

    Returns:
        Truncated list of messages (removes oldest messages first).
        At least the last message is always kept, even if it exceeds budget.
        Tool messages (role="tool") are kept together with their preceding
        assistant message to avoid breaking the tool calling context.
    """
    if not messages:
        return []

    message_budget = max_tokens - reserve_tokens
    if message_budget <= 0:
        return _recent_window(messages)

    # The system prompt is sent in full no matter what this function returns,
    # so charging it entirely against the message budget only starves the
    # conversation. A starved history loses tool results, which makes the model
    # repeat the same call forever and makes strict tool APIs reject the
    # request outright ("No tool output found for tool call"). Reserve at most
    # half of the message budget for the system prompt.
    budget = max(
        message_budget - estimate_tokens(system_prompt),
        message_budget // 2,
    )

    result: list[dict[str, Any]] = []
    used = 0

    # Always preserve the first user message (contains the original question)
    first_user_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            first_user_idx = i
            break
    first_user = messages[first_user_idx] if first_user_idx >= 0 else None
    if first_user is not None:
        first_user_tokens = _msg_tokens(first_user)
        # Reserve space for the first user message
        budget -= first_user_tokens

    # Always keep the last message (most recent user query)
    for msg in reversed(messages[:-1]):
        if first_user is not None and msg is first_user:
            continue  # Will be inserted at the start
        tokens = _msg_tokens(msg)
        if used + tokens > budget:
            break
        # Keep tool messages paired with their preceding assistant message
        result.insert(0, msg)
        used += tokens

    # Insert first user message at the start if it was preserved
    if first_user is not None and first_user not in result:
        result.insert(0, first_user)

    # Append the last message unconditionally
    # Avoid duplicating if the first user IS the last message
    if not result or result[-1] is not messages[-1]:
        result.append(messages[-1])

    # Remove orphan tool messages: DeepSeek and other strict APIs require
    # every tool message to follow an assistant message with tool_calls.
    # Orphans can appear anywhere in the list (not just at the start) when
    # the preceding assistant message was dropped by the budget limit.
    result = _remove_orphan_tool_messages(result)

    # A single surviving message means every tool result was dropped. The model
    # then re-issues the same call, so prefer a short recent window that still
    # carries the latest tool exchange.
    if len(result) < 2 and len(messages) > 1:
        return _recent_window(messages)

    return result


def _recent_window(
    messages: list[dict[str, Any]],
    keep: int = _MIN_RECENT_MESSAGES,
) -> list[dict[str, Any]]:
    """Keep a recent, structurally valid suffix of the history.

    Used when the budget is exhausted before any message is inspected.
    Collapsing the history to a single message dropped every tool result,
    which both destroyed the conversation and produced requests that strict
    tool APIs reject. A short recent window keeps the latest tool exchange
    intact instead.
    """
    window = list(messages[-keep:]) if keep > 0 else list(messages[-1:])
    while window and window[0].get("role") == "tool":
        window.pop(0)
    if not window:
        window = [messages[-1]]
    return _remove_orphan_tool_messages(window)


def _remove_orphan_tool_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove tool messages that lack a preceding assistant with tool_calls.

    DeepSeek's API strictly requires: "Messages with role 'tool' must be a
    response to a preceding message with 'tool_calls'." Consecutive tool
    messages are all responses to the same assistant, so we look backward
    for the most recent non-tool message to verify the pairing.
    """
    clean: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            clean.append(msg)
            continue
        # Look backward for the most recent non-tool message
        paired = False
        for j in range(i - 1, -1, -1):
            prev = messages[j]
            if prev.get("role") == "tool":
                continue  # skip consecutive tool messages
            if prev.get("role") == "assistant" and "tool_calls" in prev:
                paired = True
            break  # found the pairing anchor
        if paired:
            clean.append(msg)

    # Safety net: never return empty list if input was non-empty.
    # If all messages are orphan tool messages, keep at least the
    # most recent non-tool message or the last message as fallback.
    if not clean and messages:
        for msg in reversed(messages):
            if msg.get("role") != "tool":
                return [msg]
        return [messages[-1]]

    return clean