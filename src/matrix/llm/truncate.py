"""Message truncation utilities for token budget management.

Estimates token count from character count and truncates message lists
to fit within a maximum token budget. Uses a conservative heuristic:
- Chinese characters: ~1.5 tokens per char
- ASCII/English: ~0.25 tokens per char (roughly 4 chars per token)
"""

from __future__ import annotations

import re


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


def truncate_messages(
    messages: list[dict[str, str]],
    system_prompt: str = "",
    max_tokens: int = 8000,
    reserve_tokens: int = 2000,
) -> list[dict[str, str]]:
    """Truncate message history to fit within a token budget.

    Args:
        messages: List of role/content messages, oldest first.
        system_prompt: System prompt to reserve tokens for.
        max_tokens: Maximum total tokens allowed.
        reserve_tokens: Tokens to reserve for the model's response.

    Returns:
        Truncated list of messages (removes oldest messages first).
    """
    budget = max_tokens - reserve_tokens - estimate_tokens(system_prompt)
    if budget <= 0:
        return messages[-2:] if len(messages) >= 2 else messages

    result: list[dict[str, str]] = []
    used = 0
    for msg in reversed(messages):
        tokens = estimate_tokens(msg.get("content", ""))
        if used + tokens > budget:
            break
        result.insert(0, msg)
        used += tokens

    return result