"""Assemble the memory block injected into the system prompt.

Two-tier injection:

- **Policies** are injected in full. They are hard rules, they are few, and
  dropping one because it scored poorly would silently relax a constraint the
  user set deliberately.
- **Preferences** compete for slots via hybrid retrieval, then the whole block
  is truncated to a token budget.

The budget is what stops prompt cost from growing linearly with the memory
store. At 80 memories the old full-dump approach spent the entire budget on
context the current turn probably did not need.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from ..llm.truncate import estimate_tokens
from .retriever import MemoryRetriever, RetrievedMemory
from .temporal import humanize_span, resolve_time_range

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_TOKENS = 800
DEFAULT_TOP_K = 8


@dataclass
class MemoryBlock:
    """The assembled prompt block plus what was spent to build it."""

    text: str
    policy_count: int = 0
    preference_count: int = 0
    tokens: int = 0
    truncated: int = 0
    retrieved: list[RetrievedMemory] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_count": self.policy_count,
            "preference_count": self.preference_count,
            "tokens": self.tokens,
            "truncated": self.truncated,
            "retrieved": [r.to_dict() for r in (self.retrieved or [])],
        }


def _format_line(
    label: str, key: str, value: str, age_hint: str = "",
) -> str:
    return f"- [{label}] {key}: {value}{age_hint}"


def _age_hint(updated_at: float, now: float) -> str:
    if updated_at <= 0:
        return ""
    age_days = (now - updated_at) / 86400
    if age_days > 90:
        return f" [established {int(age_days)}d ago]"
    return ""


def build_memory_block(
    store: Any,
    user_id: str,
    query: str = "",
    retriever: MemoryRetriever | None = None,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    top_k: int = DEFAULT_TOP_K,
    full_dump: bool = False,
    session_id: str = "",
) -> MemoryBlock:
    """Build the memory section for a system prompt.

    Args:
        store: ``SessionStore``.
        query: current user utterance. Empty means "no signal", in which case
            the block falls back to the highest-weight memories.
        retriever: when provided and the query is non-empty, preferences are
            selected by hybrid retrieval instead of being dumped.
        budget_tokens: hard ceiling for the whole block.
        top_k: how many preferences retrieval may contribute.
        full_dump: emergency escape hatch that restores the legacy behaviour.

    Returns a ``MemoryBlock``; ``text`` is empty when there is nothing to say.
    """
    now = time.time()
    parts: list[str] = []
    selected: list[RetrievedMemory] = []
    truncated = 0

    policies = store.get_policies(user_id)
    if policies:
        parts.append("## Hard Rules (DO NOT override these)")
        for key, value in policies.items():
            parts.append(_format_line("HARD RULE", key, value))

    if full_dump or not query or retriever is None:
        # Legacy / no-signal path: everything, ordered by decayed weight.
        preferences = store.get_preferences(user_id)
        if preferences:
            parts.append("\n## User Preferences")
            for key, value in preferences.items():
                parts.append(_format_line("PREFERENCE", key, value))
        text = "\n".join(parts) if parts else ""
        tokens = estimate_tokens(text)
        if tokens > budget_tokens and preferences:
            # Keep every policy; trim preferences from the end.
            text, dropped = _truncate_preferences(parts, budget_tokens)
            truncated = dropped
            tokens = estimate_tokens(text)
        _mark_accessed(store, user_id, list(policies.keys()))
        return MemoryBlock(
            text=text,
            policy_count=len(policies),
            preference_count=len(preferences),
            tokens=tokens,
            truncated=truncated,
        )

    # Temporal scoping: "三个月前我在研究什么" should not surface today's facts.
    time_from, time_to = resolve_time_range(query)
    if time_from or time_to:
        logger.debug(
            "memory_block: temporal scope %s", humanize_span(time_from, time_to),
        )
    # Session-scoped facts: visible only inside the conversation that made them.
    if session_id:
        session_mems = store.get_session_memories(user_id, session_id)
        if session_mems:
            parts.append("\n## This Conversation")
            for mem in session_mems[:top_k]:
                parts.append(_format_line("SESSION", mem["key"], mem["value"]))

    selected = retriever.search(
        user_id, query, top_k=top_k, memory_type="preference",
        time_from=time_from, time_to=time_to,
    )
    if selected:
        parts.append("\n## User Preferences (relevant to this turn)")
        for item in selected:
            hint = _age_hint(item.updated_at, now)
            parts.append(_format_line("PREFERENCE", item.key, item.value, hint))

    text = "\n".join(parts) if parts else ""
    tokens = estimate_tokens(text)
    if tokens > budget_tokens:
        text, dropped = _truncate_preferences(parts, budget_tokens)
        truncated = dropped
        tokens = estimate_tokens(text)

    _mark_accessed(
        store, user_id,
        list(policies.keys()) + [r.key for r in selected[: len(selected) - truncated]],
    )
    logger.debug(
        "memory_block: user=%s policies=%d prefs=%d tokens=%d truncated=%d",
        user_id, len(policies), len(selected), tokens, truncated,
    )
    return MemoryBlock(
        text=text,
        policy_count=len(policies),
        preference_count=max(0, len(selected) - truncated),
        tokens=tokens,
        truncated=truncated,
        retrieved=selected,
    )


def _truncate_preferences(parts: list[str], budget_tokens: int) -> tuple[str, int]:
    """Drop preference lines from the end until the block fits the budget."""
    # Never touch the header or the policy section.
    try:
        split_at = next(
            i for i, line in enumerate(parts)
            if line.lstrip().startswith("## User Preferences")
        )
    except StopIteration:
        return "\n".join(parts), 0

    head = parts[:split_at]
    prefs = parts[split_at:]
    while len(prefs) > 1 and estimate_tokens("\n".join(head + prefs)) > budget_tokens:
        prefs.pop()
    return "\n".join(head + prefs), max(0, len(parts[split_at:]) - len(prefs))


def _mark_accessed(store: Any, user_id: str, keys: list[str]) -> None:
    """Record retrieval so forgetting can be access-aware. Never fatal."""
    try:
        store.mark_accessed(user_id, keys)
    except Exception as exc:  # noqa: BLE001
        logger.debug("memory_block: mark_accessed failed: %s", exc)
