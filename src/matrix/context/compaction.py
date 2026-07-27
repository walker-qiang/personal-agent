"""L3 Compaction: conversation-level compression for context window management.

Triggers when prompt_tokens / context_window >= 85% (COMPACTION_THRESHOLD).
Compresses conversation history into a structured handoff document that
replaces the original messages, targeting ~30% of the window.

Design principles:
- Structured handoff: five fixed sections (goal, history, abandoned, critical, refs)
- Token-based backward cut point (keeps ~20K tokens of recent context)
- Never cut from a tool message (preserves assistant-tool pairing)
- Incremental update: merges with previous summary on repeated compactions
- Compiled data ref index to prevent ref loss after compression
- Uses pipeline LLM to avoid consuming main LLM's context
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..llm import LLMClient
from ..llm.truncate import estimate_tokens

logger = logging.getLogger("matrix.context")

# Thresholds
COMPACTION_THRESHOLD = 0.85
COMPACTION_TARGET = 0.30
MIN_PRESERVE_MESSAGES = 6
MIN_DELETE_MESSAGES = 2
CONTEXT_WINDOW_TOKENS = 128000

# Token-based cut point: keep at least this many tokens of recent messages
KEEP_RECENT_TOKENS = 20000


# Compaction prompt: 5 sections (added critical_context)
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
    "Approach description: Reason it was abandoned. Include only if there were failed attempts."
  ],
  "critical_context": "Critical information that must not be forgotten: specific values, constraints, user preferences, configuration details. List as bullet points.",
  "data_references": [
    {"refId": "xxx", "tool": "tool_name", "summary": "What this data contains"}
  ]
}

CRITICAL RULES:
1. Preserve ALL concrete values: numbers, dates, prices, IDs, names — do NOT generalize them
2. For execution_history, group by logical phases, not by individual messages
3. abandoned_paths should only include approaches that were explicitly tried and failed
4. critical_context: anything that losing would cause the agent to repeat work or make wrong assumptions
5. data_references: include every __refId found in tool results
6. Keep the user_goal concise but exact — it's the anchor for future reasoning
7. Output ONLY the JSON object, no other text"""


# Incremental update prompt
UPDATE_SYSTEM_PROMPT = """You are updating an existing conversation summary with new information.

You will receive:
1. A previous summary (JSON)
2. New conversation messages to incorporate

Your task is to produce an updated JSON summary with the same structure:

{
  "user_goal": "Keep the same as previous, unless the user explicitly changed direction.",
  "execution_history": [...],  // Merge: keep existing phases, add new ones
  "abandoned_paths": [...],    // Accumulate: keep existing, add new if any
  "critical_context": "...",   // Merge: keep existing critical info, add new
  "data_references": [...]     // Accumulate: keep existing refs, add new ones
}

CRITICAL RULES:
1. Do NOT remove information from the previous summary unless it's explicitly contradicted
2. Preserve ALL concrete values from both previous summary and new messages
3. For execution_history, append new phases rather than rewriting old ones
4. Output ONLY the JSON object, no other text"""


def build_compaction_messages(
    messages: list[dict[str, Any]],
    user_goal: str,
    previous_summary: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build a compact prompt for the compaction LLM.

    If previous_summary is provided, uses incremental update prompt.
    """
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
            content_str = str(content)[:500]
            ref_hint = ""
            if "__refId" in content_str or "__stored" in content_str:
                ref_hint = " [CONTAINS REF]"
            conversation_parts.append(f"[TOOL #{i}]{ref_hint} {content_str}")

    conversation_text = "\n".join(conversation_parts)

    if previous_summary is not None:
        return [
            {"role": "system", "content": UPDATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Previous summary:\n{json.dumps(previous_summary, ensure_ascii=False, indent=2)}\n\n"
                    f"New conversation to incorporate:\n\n{conversation_text}\n\n"
                    "Generate the updated summary JSON."
                ),
            },
        ]

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
    """Convert a structured handoff dict into a single system message."""
    parts = ["## Conversation Handoff\n"]

    parts.append(f"### User Goal\n{handoff.get('user_goal', '')}\n")

    history = handoff.get("execution_history", [])
    if history:
        parts.append("### What Was Done")
        for phase in history:
            phase_name = phase.get("phase", "Phase")
            actions = phase.get("actions", "")
            outcome = phase.get("outcome", "")
            parts.append(f"\n**{phase_name}**: {actions}")
            if outcome:
                parts.append(f"  -> Result: {outcome}")

    abandoned = handoff.get("abandoned_paths", [])
    if abandoned:
        parts.append("\n\n### Approaches Already Tried (Do Not Retry)")
        for path in abandoned:
            parts.append(f"- {path}")

    critical = handoff.get("critical_context", "")
    if critical:
        parts.append(f"\n\n### Critical Context (Do Not Forget)\n{critical}")

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


def find_cut_point(
    messages: list[dict[str, Any]],
    keep_tokens: int = KEEP_RECENT_TOKENS,
) -> int:
    """Find the cut point using token-based backward traversal.

    Walks from the latest message backward, accumulating tokens until
    reaching keep_tokens. Then finds the nearest valid cut point
    (not a tool message) at or after that position.

    Returns the index of the first message to KEEP (0-based).
    Messages before this index will be compressed.
    """
    if len(messages) <= MIN_PRESERVE_MESSAGES + MIN_DELETE_MESSAGES:
        return -1  # Not enough to compress

    accumulated = 0
    target_idx = len(messages) - 1

    for i in range(len(messages) - 1, -1, -1):
        msg_tokens = estimate_tokens(str(messages[i].get("content", "")))
        accumulated += msg_tokens
        if accumulated >= keep_tokens:
            target_idx = i
            break
        target_idx = i

    # Find nearest valid cut point (not a tool message) at or after target_idx
    cut = target_idx
    while cut < len(messages) and messages[cut].get("role") == "tool":
        cut += 1

    # Ensure minimum delete
    if cut < MIN_DELETE_MESSAGES:
        cut = MIN_DELETE_MESSAGES
        while cut < len(messages) and messages[cut].get("role") == "tool":
            cut += 1

    # Ensure minimum preserve
    if len(messages) - cut < MIN_PRESERVE_MESSAGES:
        return -1  # Can't preserve enough

    return cut


def compact_messages(
    messages: list[dict[str, Any]],
    user_goal: str,
    llm: LLMClient,
    context_window: int = CONTEXT_WINDOW_TOKENS,
    previous_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compress conversation history into a structured handoff.

    Args:
        messages: The full message list to compress.
        user_goal: The user's original request (anchor).
        llm: Pipeline LLM for compaction (not the main LLM).
        context_window: Context window size in tokens.
        previous_summary: If provided, do incremental update instead of from-scratch.

    Returns:
        A new message list with the handoff replacing compressed messages.
    """
    cut_point = find_cut_point(messages)
    if cut_point < 0:
        return messages  # Not enough to compress

    to_compress = messages[:cut_point]
    to_keep = list(messages[cut_point:])

    logger.info(
        "compaction: compressing %d messages, keeping %d, goal=%s...",
        len(to_compress), len(to_keep), user_goal[:80],
    )

    compaction_msgs = build_compaction_messages(to_compress, user_goal, previous_summary)

    try:
        handoff = llm.complete_json(
            compaction_msgs[0]["content"], compaction_msgs[1:], temperature=0.3,
        )
        if not isinstance(handoff, dict) or "user_goal" not in handoff:
            logger.warning("Compaction JSON missing user_goal, falling back to truncation")
            return _fallback_truncate(messages, cut_point, to_keep)

        # Merge data_references from previous summary
        if previous_summary and "data_references" in previous_summary:
            existing_refs = {r.get("refId") for r in handoff.get("data_references", [])}
            for ref in previous_summary["data_references"]:
                if ref.get("refId") not in existing_refs:
                    handoff.setdefault("data_references", []).append(ref)

    except Exception as e:
        logger.warning("Compaction LLM call failed: %s, falling back to truncation", e)
        return _fallback_truncate(messages, cut_point, to_keep)

    handoff_msg = build_handoff_message(handoff)
    result = [handoff_msg] + to_keep

    logger.info(
        "compaction_complete: %d messages -> %d messages (handoff + %d preserved)",
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
