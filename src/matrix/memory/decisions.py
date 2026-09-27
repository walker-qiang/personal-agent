"""Write-time decision engine for long-term memory.

The previous pipeline did ``upsert`` first and hoped a post-hoc evolution pass
would clean up the mess. That has two failure modes:

- **Duplicates pile up.** The vault grew three near-identical entries
  (``investment_research_data_preference``, ``research_method_preference``,
  ``market_research_tool``) because nothing compared a candidate against what
  was already stored.
- **Contradictions persist.** Conflict detection was a boolean word list, so
  "研究对象从贵州茅台换成五粮液" looked like two unrelated facts.

This module inverts the order: before writing, each candidate is compared
against its nearest existing memories and the LLM returns one of four
operations — ``ADD``, ``UPDATE``, ``DELETE`` or ``NONE``. Evolution then
becomes a low-frequency janitor instead of the primary correctness mechanism.

When the LLM is unavailable the engine degrades to ``ADD`` for everything,
which reproduces the old behaviour rather than losing the memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)

VALID_OPS = ("ADD", "UPDATE", "DELETE", "NONE")
VALID_TYPES = ("preference", "policy")

DECISION_SYSTEM_PROMPT = """你是一个记忆写入决策引擎。给定「新候选事实」和「已有记忆」，为每条候选判定写入操作。

可选操作：
- ADD    —— 全新事实，已有记忆中没有对应条目
- UPDATE —— 已有记忆中语义相同或矛盾，用新值替换（key 沿用已有条目的 key）
- DELETE —— 新信息明确否定了已有记忆（如"我不再持有XX"），删除该条目
- NONE   —— 已有记忆已覆盖该信息，或候选不是值得长期记住的事实（闲聊、一次性指令）

判定规则：
1. 先判断是否值得长期记住。用户随口的、一次性的、不会复用的信息 → NONE
2. 语义相同但表述不同 → UPDATE，不要新建条目
3. 新事实与旧事实矛盾 → UPDATE（用新的），不要并存两条
4. 价值不同但都对（如"喜欢咖啡"和"喜欢茶"）→ ADD，两条并存是合理的
5. key 用简洁的中文或英文短语；UPDATE/DELETE 必须使用已有条目的原 key

只输出 JSON，不要解释：
{"decisions": [{"op": "ADD|UPDATE|DELETE|NONE", "key": "...", "value": "...", "type": "preference|policy", "target": "已有条目的 key（UPDATE/DELETE 时必填）"}]}"""

DECISION_USER_TEMPLATE = """对话片段：
用户：{question}
助手：{answer}

新候选事实：
{candidates}

已有记忆（最近的 {existing_count} 条）：
{existing}

请对每条候选事实输出判定。"""


@dataclass
class WriteDecision:
    """One resolved write operation."""

    op: str
    key: str
    value: str = ""
    memory_type: str = "preference"
    target: str = ""

    @property
    def is_write(self) -> bool:
        return self.op in ("ADD", "UPDATE")


@dataclass
class DecisionOutcome:
    """Result of one decision pass."""

    decisions: list[WriteDecision]
    llm_used: bool = False
    reason: str = ""

    def counts(self) -> dict[str, int]:
        out = {op: 0 for op in VALID_OPS}
        for decision in self.decisions:
            out[decision.op] = out.get(decision.op, 0) + 1
        return out


def _normalize_op(raw: Any) -> str:
    op = str(raw or "").strip().upper()
    return op if op in VALID_OPS else "NONE"


def _normalize_type(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    return value if value in VALID_TYPES else "preference"


def format_candidates(candidates: Sequence[dict[str, Any]]) -> str:
    if not candidates:
        return "（无）"
    lines = []
    for i, item in enumerate(candidates, 1):
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        mem_type = _normalize_type(item.get("type"))
        lines.append(f"{i}. [{mem_type}] {key}: {value}")
    return "\n".join(lines)


def format_existing(existing: Sequence[dict[str, Any]], limit: int = 40) -> str:
    if not existing:
        return "（无）"
    lines = []
    for item in existing[:limit]:
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        lines.append(f"- {item.get('memory_type', 'preference')} | {key}: {value}")
    return "\n".join(lines)


class MemoryDecisionEngine:
    """Turns extracted candidate facts into explicit write operations."""

    def __init__(self, llm: Any | None = None, enabled: bool = True) -> None:
        self._llm = llm
        self._enabled = enabled and llm is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def decide(
        self,
        question: str,
        answer: str,
        candidates: list[dict[str, Any]],
        existing: list[dict[str, Any]],
    ) -> DecisionOutcome:
        """Resolve candidates into operations.

        Falls back to ``ADD`` for every candidate when the engine is disabled
        or the LLM call fails — losing a memory is worse than storing a
        redundant one.
        """
        if not candidates:
            return DecisionOutcome(decisions=[], llm_used=False, reason="no candidates")

        if not self._enabled:
            return DecisionOutcome(
                decisions=[
                    WriteDecision(
                        op="ADD",
                        key=str(c.get("key", "")).strip(),
                        value=str(c.get("value", "")).strip(),
                        memory_type=_normalize_type(c.get("type")),
                    )
                    for c in candidates
                ],
                llm_used=False,
                reason="engine disabled",
            )

        prompt = DECISION_USER_TEMPLATE.format(
            question=question[:400],
            answer=answer[:600],
            candidates=format_candidates(candidates),
            existing=format_existing(existing),
            existing_count=len(existing),
        )
        try:
            data = self._llm.complete_json(
                DECISION_SYSTEM_PROMPT,
                [{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory_decision: llm failed (%s), defaulting to ADD", exc)
            return self._fallback(candidates, reason=f"llm error: {type(exc).__name__}")

        raw = data.get("decisions") if isinstance(data, dict) else None
        if not isinstance(raw, list) or not raw:
            logger.warning("memory_decision: empty response, defaulting to ADD")
            return self._fallback(candidates, reason="empty llm response")

        decisions: list[WriteDecision] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            op = _normalize_op(item.get("op"))
            key = str(item.get("key", "")).strip()
            target = str(item.get("target", "")).strip()
            if op == "NONE":
                decisions.append(WriteDecision(op="NONE", key=key or target))
                continue
            if op in ("UPDATE", "DELETE"):
                if not target:
                    # Cannot act without a concrete existing key.
                    decisions.append(WriteDecision(op="NONE", key=key))
                    continue
                decisions.append(
                    WriteDecision(
                        op=op,
                        key=target,
                        value=str(item.get("value", "")).strip(),
                        memory_type=_normalize_type(item.get("type")),
                        target=target,
                    )
                )
                continue
            if not key:
                continue
            decisions.append(
                WriteDecision(
                    op="ADD",
                    key=key,
                    value=str(item.get("value", "")).strip(),
                    memory_type=_normalize_type(item.get("type")),
                )
            )

        outcome = DecisionOutcome(decisions=decisions, llm_used=True)
        logger.info(
            "memory_decision: ops=%s candidates=%d",
            outcome.counts(), len(candidates),
        )
        return outcome

    @staticmethod
    def _fallback(
        candidates: list[dict[str, Any]], reason: str,
    ) -> DecisionOutcome:
        return DecisionOutcome(
            decisions=[
                WriteDecision(
                    op="ADD",
                    key=str(c.get("key", "")).strip(),
                    value=str(c.get("value", "")).strip(),
                    memory_type=_normalize_type(c.get("type")),
                )
                for c in candidates
            ],
            llm_used=False,
            reason=reason,
        )
