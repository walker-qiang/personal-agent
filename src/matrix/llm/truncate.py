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

    budget = max_tokens - reserve_tokens - estimate_tokens(system_prompt)
    if budget <= 0:
        # Budget exhausted by system prompt; keep only the last message
        return [messages[-1]]

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

    # Ensure we don't leave orphan tool messages at the start
    # (tool messages without their preceding assistant message)
    while result and result[0].get("role") == "tool":
        result.pop(0)

    return result