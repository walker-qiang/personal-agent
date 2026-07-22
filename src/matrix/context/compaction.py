"""L3 Compaction: conversation-level compression for context window management.

Triggers when prompt_tokens / context_window >= 85% (COMPACTION_THRESHOLD).
Compresses conversation history into a structured handoff document that
replaces the original messages, targeting ~30% of the window.

Design principles:
- Structured handoff: four fixed sections (goals, history, abandoned, refs)
- Never cut from a tool message (preserves assistant-tool pairing)
- Minimum 6 messages preserved, minimum 2 messages deleted
- Compiled data ref index to prevent ref loss after compression
- Uses pipeline LLM to avoid consuming main LLM's context
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..llm import LLMClient

logger = logging.getLogger("matrix.context")

# Thresholds
COMPACTION_THRESHOLD = 0.85   # Trigger when usage >= 85% of window
COMPACTION_TARGET = 0.30      # Target to compress to ~30% of window
MIN_PRESERVE_MESSAGES = 6     # Minimum messages to keep after compression
MIN_DELETE_MESSAGES = 2       # Minimum messages to delete (avoid trivial compactions)
CONTEXT_WINDOW_TOKENS = 128000  # Default context window


# Compaction prompt: instructs the LLM to produce a structured handoff
COMPACTION_SYSTEM_PROMPT = """You are a conversation summarizer. Your task is to compress a long conversation history into a structured handoff document.

The output MUST be a JSON object with exactly these fields:

{
  "user_goal": "The user's original request in one sentence. Preserve the exact question.",
  "execution_history": [
    {
      "phase": "Phase name (e.g., Search, Data Collection, Analysis)",
      "actions": "What was done: specific tool calls, their results, key values found",
      "outcome": "What was discovered: concrete numbers, names, IDs, decisions"
    }
  ],
  "abandoned_paths": [
    "~~Approach description~~: Reason it was abandoned. Include only if there were failed attempts."
  ],
  "data_references": [
    {"refId": "xxx", "tool": "tool_name", "summary": "What this data contains"}
  ]
}

CRITICAL RULES:
1. Preserve ALL concrete values: numbers, dates, prices, IDs, names — do NOT generalize them
2. For execution_history, group by logical phases, not by individual messages
3. abandoned_paths should only include approaches that were explicitly tried and failed
4. data_references: include every __refId found in tool results
5. Keep the user_goal concise but exact — it's the anchor for future reasoning
6. Output ONLY the JSON object, no other text"""


def build_compaction_messages(
    messages: list[dict[str, Any]],
    user_goal: str,
) -> list[dict[str, str]]:
    """Build a compact prompt for the compaction LLM.

    We extract the most relevant parts of the conversation to give the
    compaction LLM enough context to produce a useful handoff.
    """
    # Build a compact representation of the conversation
    conversation_parts: list[str] = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            conversation_parts.append(f"[USER #{i}] {str(content)[:500]}")
        elif role == "assistant":
            tc = msg.get("tool_calls", [])
            if tc:
                tool_names = [t.get("function", {}).get("name", "?") for t in tc]
                conversation_parts.append(
                    f"[ASSISTANT #{i}] called {', '.join(tool_names)}"
                )
            elif content:
                conversation_parts.append(
                    f"[ASSISTANT #{i}] {str(content)[:300]}"
                )
        elif role == "tool":
            # Extract refId if present
            content_str = str(content)[:500]
            ref_hint = ""
            if "__refId" in content_str or "__stored" in content_str:
                ref_hint = " [CONTAINS REF]"
            conversation_parts.append(f"[TOOL #{i}]{ref_hint} {content_str}")

    conversation_text = "\n".join(conversation_parts)

    return [
        {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User's original goal: {user_goal}\n\n"
                f"Conversation to compress:\n\n{conversation_text}\n\n"
                "Generate the structured handoff JSON."
            ),
        },
    ]


def build_handoff_message(handoff: dict[str, Any]) -> dict[str, Any]:
    """Convert a structured handoff dict into a single system message.

    This replaces the compressed conversation history in the LLM context.
    """
    parts = ["## Conversation Handoff\n"]

    # 1. User's original goal
    parts.append(f"### User Goal\n{handoff.get('user_goal', '')}\n")

    # 2. Execution history
    history = handoff.get("execution_history", [])
    if history:
        parts.append("### What Was Done")
        for phase in history:
            phase_name = phase.get("phase", "Phase")
            actions = phase.get("actions", "")
            outcome = phase.get("outcome", "")
            parts.append(f"\n**{phase_name}**: {actions}")
            if outcome:
                parts.append(f"  → Result: {outcome}")

    # 3. Abandoned paths
    abandoned = handoff.get("abandoned_paths", [])
    if abandoned:
        parts.append("\n\n### Approaches Already Tried (Do Not Retry)")
        for path in abandoned:
            parts.append(f"- {path}")

    # 4. Data references
    refs = handoff.get("data_references", [])
    if refs:
        parts.append("\n\n### External Data References (use get_stored_data)")
        for ref in refs:
            parts.append(
                f"- refId=`{ref.get('refId', '?')}` "
                f"({ref.get('tool', '?')}): {ref.get('summary', '')}"
            )

    parts.append(
        "\n\n---\n"
        "The conversation above has been compressed. "
        "Use this handoff as context. Key data is available via get_stored_data."
    )

    return {"role": "system", "content": "\n".join(parts)}


def compact_messages(
    messages: list[dict[str, Any]],
    user_goal: str,
    llm: LLMClient,
    context_window: int = CONTEXT_WINDOW_TOKENS,
) -> list[dict[str, Any]]:
    """Compress conversation history into a structured handoff.

    Args:
        messages: The full message list to compress.
        user_goal: The user's original request (anchor).
        llm: Pipeline LLM for compaction (not the main LLM).
        context_window: Context window size in tokens.

    Returns:
        A new message list with the handoff replacing compressed messages.
        At least MIN_PRESERVE_MESSAGES remain.
    """
    if len(messages) <= MIN_PRESERVE_MESSAGES + MIN_DELETE_MESSAGES:
        return messages  # Not enough to compress

    # Find a safe cut point: cannot cut from a tool message
    target_keep = MIN_PRESERVE_MESSAGES
    cut_point = len(messages) - target_keep

    # Ensure we don't cut from a tool message
    while cut_point > 0 and messages[cut_point].get("role") == "tool":
        cut_point -= 1
        target_keep = len(messages) - cut_point

    if cut_point <= MIN_DELETE_MESSAGES:
        return messages  # Not enough messages before the safe cut point

    to_compress = messages[:cut_point]
    to_keep = list(messages[cut_point:])

    logger.info(
        "compaction: compressing %d messages, keeping %d, goal=%s...",
        len(to_compress), len(to_keep), user_goal[:80],
    )

    # Build compaction prompt and call pipeline LLM
    compaction_msgs = build_compaction_messages(to_compress, user_goal)

    try:
        handoff = llm.complete_json(
            compaction_msgs[0]["content"], compaction_msgs[1:], temperature=0.3,
        )
        if not isinstance(handoff, dict) or "user_goal" not in handoff:
            logger.warning("Compaction JSON missing user_goal, falling back to truncation")
            return _fallback_truncate(messages, cut_point, to_keep)
    except Exception as e:
        logger.warning("Compaction LLM call failed: %s, falling back to truncation", e)
        return _fallback_truncate(messages, cut_point, to_keep)

    handoff_msg = build_handoff_message(handoff)

    # Build new message list: handoff + preserved messages
    # Insert handoff as a system message at the beginning
    result = [handoff_msg] + to_keep

    logger.info(
        "compaction_complete: %d messages → %d messages (handoff + %d preserved)",
        len(messages), len(result), len(to_keep),
    )

    return result


def _fallback_truncate(
    messages: list[dict[str, Any]],
    cut_point: int,
    to_keep: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fallback compaction: simple truncation with a summary marker."""
    truncated = messages[:cut_point]
    marker = {
        "role": "system",
        "content": (
            f"[{len(truncated)} earlier messages were truncated due to context limits. "
            "Key information should have been recorded in working memory.]"
        ),
    }
    return [marker] + to_keep