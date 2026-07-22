"""Memory Evolution: consolidate, resolve conflicts, and forget stale memories.

After each conversation, the system extracts new memories (key-value pairs)
from the dialogue. Over time, this leads to three problems:

1. **Proliferation**: The memory store grows unbounded with similar entries.
2. **Conflict**: A user's preference changes ("prefers_dark_mode" → "prefers_light_mode"),
   but the old entry remains, sending contradictory signals to the LLM.
3. **Staleness**: Low-value memories from months ago still occupy space and
   dilute the signal-to-noise ratio of the system prompt.

The MemoryEvolution module addresses these through a four-stage pipeline that
runs as a background task after each conversation:

    ┌─────────────────────────────────────────────────────────┐
    │  Stage 1: Importance Scoring                           │
    │  Compute importance = decay_weight × type_weight ×     │
    │  (1 + recency_boost) for each memory                   │
    ├─────────────────────────────────────────────────────────┤
    │  Stage 2: Conflict Detection                           │
    │  Find memories with similar keys but contradictory      │
    │  values. Keep the most recent, delete the stale one.    │
    │  (Uses LLM for semantic conflict detection.)           │
    ├─────────────────────────────────────────────────────────┤
    │  Stage 3: Consolidation                               │
    │  Merge near-duplicate memories into a single, richer   │
    │  entry. (Uses LLM for semantic merging.)                │
    ├─────────────────────────────────────────────────────────┤
    │  Stage 4: Active Forgetting                             │
    │  If memory count exceeds MAX_MEMORIES (default 80),     │
    │  remove the lowest-importance entries.                  │
    └─────────────────────────────────────────────────────────┘

The module is designed to work with SessionStore but operates on the store's
public API, making it testable in isolation with a mock store.
"""

from __future__ import annotations

import logging
import time
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..store import SessionStore
    from ..llm.protocol import LLMClient

logger = logging.getLogger(__name__)

# ---- Configuration -----------------------------------------------------------

@dataclass
class EvolutionConfig:
    """Configuration for memory evolution."""
    max_memories: int = 80              # trigger forgetting above this count
    forget_batch_size: int = 10         # how many to forget in one pass
    similarity_threshold: float = 0.75  # above this = near-duplicate
    conflict_recency_window: float = 7 * 86400  # 7 days: if newer memory contradicts older
    enable_llm_consolidation: bool = True  # use LLM for semantic merging
    enable_active_forgetting: bool = True
    # Importance weights
    type_weight_policy: float = 2.0     # policies are 2x more important
    type_weight_preference: float = 1.0
    recency_boost_days: float = 3.0     # memories < 3 days old get a boost
    recency_boost_factor: float = 1.5   # boost multiplier


# ---- Data structures ---------------------------------------------------------

@dataclass
class ScoredMemory:
    """A memory with its computed importance score."""
    key: str
    value: str
    memory_type: str
    created_at: float
    updated_at: float
    importance: float = 0.0


@dataclass
class EvolutionReport:
    """Result of a single evolution pass."""
    total_before: int = 0
    total_after: int = 0
    conflicts_resolved: int = 0
    memories_consolidated: int = 0
    memories_forgotten: int = 0
    details: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Evolution: {self.total_before}→{self.total_after} "
            f"(conflicts={self.conflicts_resolved}, "
            f"consolidated={self.memories_consolidated}, "
            f"forgotten={self.memories_forgotten})"
        )


# ---- Text similarity utilities ----------------------------------------------

def _normalize_key(key: str) -> str:
    """Normalize a memory key for similarity comparison."""
    # Lowercase, strip, replace separators with underscores
    k = key.lower().strip()
    k = re.sub(r"[\s\-]+", "_", k)
    # Remove common prefixes (iteratively for stacked prefixes)
    k = re.sub(r"^(user_|prefers?_|likes?_|favorite_|fav_)", "", k)
    # Remove common suffixes iteratively (e.g. "dark_mode_pref" → "dark_mode" → "dark")
    while True:
        new_k = re.sub(r"(_mode|_preference|_pref|_setting|_config)$", "", k)
        if new_k == k:
            break
        k = new_k
    return k


def _tokenize(text: str) -> set[str]:
    """Simple tokenization for Jaccard similarity."""
    return set(re.findall(r"\w+", text.lower()))


def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two strings based on token overlap."""
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    intersection = ta & tb
    union = ta | tb
    return len(intersection) / len(union)


def _key_similarity(key1: str, key2: str) -> float:
    """Similarity between two memory keys, using normalized forms."""
    nk1 = _normalize_key(key1)
    nk2 = _normalize_key(key2)
    if nk1 == nk2:
        return 1.0
    return _jaccard_similarity(nk1, nk2)


def _values_conflict(v1: str, v2: str) -> bool:
    """Heuristic: check if two values are contradictory.

    Simple approach: if they share the same key topic but have opposite
    boolean values (yes/no, true/false, etc.) or very different content.
    """
    # Normalize to lowercase for comparison
    a = v1.lower().strip()
    b = v2.lower().strip()
    if a == b:
        return False  # Same value, not a conflict

    # Check for boolean contradictions
    positive = {"yes", "true", "on", "enabled", "active", "喜欢", "是", "开", "用"}
    negative = {"no", "false", "off", "disabled", "inactive", "不喜欢", "否", "关", "不用"}
    a_pos = a in positive or any(p in a for p in positive)
    a_neg = a in negative or any(n in a for n in negative)
    b_pos = b in positive or any(p in b for p in positive)
    b_neg = b in negative or any(n in b for n in negative)

    if (a_pos and b_neg) or (a_neg and b_pos):
        return True

    # Check for direct negation patterns
    if f"不{a}" in b or f"not {a}" in b or f"don't {a}" in b:
        return True
    if f"不{b}" in a or f"not {b}" in a or f"don't {b}" in a:
        return True

    return False


# ---- Main evolution class ---------------------------------------------------

class MemoryEvolution:
    """Evolution pipeline for user memories.

    Usage::

        evolution = MemoryEvolution(store, config)
        report = evolution.evolve(user_id)
        print(report)  # "Evolution: 85→72 (conflicts=3, consolidated=5, forgotten=5)"
    """

    def __init__(
        self,
        store: SessionStore,
        config: EvolutionConfig | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.store = store
        self.config = config or EvolutionConfig()
        self.llm = llm

    def evolve(self, user_id: str) -> EvolutionReport:
        """Run the full evolution pipeline for a user.

        This is the main entry point, called after each conversation.
        Returns an EvolutionReport with statistics.
        """
        report = EvolutionReport()

        memories = self.store.get_all_memories(user_id)
        report.total_before = len(memories)

        if not memories:
            return report

        # Stage 1: Score importance
        scored = self._score_importance(memories)

        # Stage 2: Detect and resolve conflicts
        scored, conflicts = self._resolve_conflicts(user_id, scored)
        report.conflicts_resolved = conflicts

        # Stage 3: Consolidate near-duplicates
        scored, consolidated = self._consolidate(user_id, scored)
        report.memories_consolidated = consolidated

        # Stage 4: Active forgetting
        forgotten = self._forget(user_id, scored)
        report.memories_forgotten = forgotten

        report.total_after = self.store.count_memories(user_id)

        if report.total_after != report.total_before:
            logger.info(
                "memory_evolution: user=%s %s",
                user_id, str(report),
            )

        return report

    # ---- Stage 1: Importance Scoring ----

    def _score_importance(self, memories: list[dict]) -> list[ScoredMemory]:
        """Compute importance score for each memory.

        Formula: importance = decay_weight × type_weight × recency_boost

        - decay_weight: 2^(-age / half_life), 0~1, from SessionStore
        - type_weight: policy=2.0, preference=1.0
        - recency_boost: 1.5 if updated < 3 days ago, else 1.0
        """
        now = time.time()
        half_life = self.store.MEMORY_HALF_LIFE
        scored: list[ScoredMemory] = []

        for mem in memories:
            age = now - mem["updated_at"]
            # Time decay
            if mem["memory_type"] == "policy":
                decay = 1.0
            else:
                decay = 2.0 ** (-age / half_life)

            # Type weight
            type_weight = (
                self.config.type_weight_policy
                if mem["memory_type"] == "policy"
                else self.config.type_weight_preference
            )

            # Recency boost
            recency_boost = 1.0
            if age < self.config.recency_boost_days * 86400:
                recency_boost = self.config.recency_boost_factor

            importance = decay * type_weight * recency_boost

            scored.append(ScoredMemory(
                key=mem["key"],
                value=mem["value"],
                memory_type=mem["memory_type"],
                created_at=mem["created_at"],
                updated_at=mem["updated_at"],
                importance=round(importance, 4),
            ))

        # Sort by importance descending
        scored.sort(key=lambda x: x.importance, reverse=True)
        return scored

    # ---- Stage 2: Conflict Resolution ----

    def _resolve_conflicts(
        self, user_id: str, scored: list[ScoredMemory],
    ) -> tuple[list[ScoredMemory], int]:
        """Detect and resolve conflicts between memories.

        A conflict occurs when two memories have similar keys but
        contradictory values. The more recently updated one wins.
        """
        conflicts = 0
        to_remove: set[str] = set()

        for i in range(len(scored)):
            if scored[i].key in to_remove:
                continue
            for j in range(i + 1, len(scored)):
                if scored[j].key in to_remove:
                    continue

                # Check if keys are similar enough to be related
                key_sim = _key_similarity(scored[i].key, scored[j].key)
                if key_sim < self.config.similarity_threshold:
                    continue

                # Check if values conflict
                if not _values_conflict(scored[i].value, scored[j].value):
                    continue

                # Conflict detected — keep the more recent one
                older = scored[i] if scored[i].updated_at < scored[j].updated_at else scored[j]
                to_remove.add(older.key)
                conflicts += 1
                logger.debug(
                    "memory_conflict: key1=%s key2=%s, removing older=%s",
                    scored[i].key, scored[j].key, older.key,
                )

        # Delete conflicting memories from store
        for key in to_remove:
            self.store.delete_profile_key(user_id, key)

        # Filter out removed memories
        scored = [s for s in scored if s.key not in to_remove]
        return scored, conflicts

    # ---- Stage 3: Consolidation ----

    def _consolidate(
        self, user_id: str, scored: list[ScoredMemory],
    ) -> tuple[list[ScoredMemory], int]:
        """Merge near-duplicate memories.

        Two memories are candidates for merging if:
        - Their keys have high similarity (≥ threshold)
        - Their values are NOT in conflict (different values, same topic)
        - They are both of the same memory_type

        The merged memory keeps the older created_at and newer updated_at,
        with a combined value.
        """
        if not self.config.enable_llm_consolidation or not self.llm:
            # Without LLM, use simple text-based consolidation
            return self._consolidate_text(user_id, scored)

        consolidated = 0
        to_remove: set[str] = set()
        to_upsert: list[tuple[str, str, str, float, float]] = []  # key, value, type, created, updated

        for i in range(len(scored)):
            if scored[i].key in to_remove:
                continue
            for j in range(i + 1, len(scored)):
                if scored[j].key in to_remove:
                    continue

                key_sim = _key_similarity(scored[i].key, scored[j].key)
                if key_sim < self.config.similarity_threshold:
                    continue

                # Skip if values conflict (already handled in Stage 2)
                if _values_conflict(scored[i].value, scored[j].value):
                    continue

                # Skip if different types
                if scored[i].memory_type != scored[j].memory_type:
                    continue

                # Skip if values are identical
                if scored[i].value.strip() == scored[j].value.strip():
                    # Exact duplicate — just remove the older one
                    older = scored[i] if scored[i].updated_at < scored[j].updated_at else scored[j]
                    to_remove.add(older.key)
                    consolidated += 1
                    continue

                # Use LLM to merge the two memories
                merged_value = self._llm_merge_memories(
                    scored[i].key, scored[i].value,
                    scored[j].key, scored[j].value,
                )

                if merged_value and merged_value != scored[i].value:
                    # Keep the first (higher importance) key, merge values
                    created = min(scored[i].created_at, scored[j].created_at)
                    updated = max(scored[i].updated_at, scored[j].updated_at)
                    to_upsert.append((
                        scored[i].key, merged_value,
                        scored[i].memory_type, created, updated,
                    ))
                    to_remove.add(scored[j].key)
                    consolidated += 1

        # Apply changes
        for key in to_remove:
            self.store.delete_profile_key(user_id, key)

        for key, value, mem_type, created, updated in to_upsert:
            self.store.upsert_profile(user_id, key, value, memory_type=mem_type)
            # Update timestamps if the store supports it
            self._update_timestamps(user_id, key, created, updated)

        # Filter out removed memories from the scored list
        scored = [s for s in scored if s.key not in to_remove]
        return scored, consolidated

    def _consolidate_text(
        self, user_id: str, scored: list[ScoredMemory],
    ) -> tuple[list[ScoredMemory], int]:
        """Simple text-based consolidation without LLM."""
        consolidated = 0
        to_remove: set[str] = set()

        for i in range(len(scored)):
            if scored[i].key in to_remove:
                continue
            for j in range(i + 1, len(scored)):
                if scored[j].key in to_remove:
                    continue

                key_sim = _key_similarity(scored[i].key, scored[j].key)
                if key_sim < self.config.similarity_threshold:
                    continue

                if scored[i].memory_type != scored[j].memory_type:
                    continue

                # Check for exact or near-exact duplicate values
                val_sim = _jaccard_similarity(scored[i].value, scored[j].value)
                if val_sim >= self.config.similarity_threshold:
                    # Remove the older, less important one
                    older = scored[i] if scored[i].updated_at < scored[j].updated_at else scored[j]
                    to_remove.add(older.key)
                    consolidated += 1

        for key in to_remove:
            self.store.delete_profile_key(user_id, key)

        scored = [s for s in scored if s.key not in to_remove]
        return scored, consolidated

    def _llm_merge_memories(
        self, key1: str, val1: str, key2: str, val2: str,
    ) -> str | None:
        """Use LLM to semantically merge two related memories.

        Returns the merged value, or None if merging is not beneficial.
        """
        if not self.llm:
            return None

        system = (
            "You are a memory consolidation engine. Merge two related memory "
            "entries into a single concise value.\n"
            "Rules:\n"
            "- Combine complementary information, don't just concatenate\n"
            "- Keep it concise (under 200 chars)\n"
            "- If one is more specific than the other, keep the specific one\n"
            "- If they are about different aspects, combine them\n"
            "- Output ONLY the merged value, no explanation, no prefix"
        )
        user_msg = (
            f"Memory A — {key1}: {val1}\n"
            f"Memory B — {key2}: {val2}\n\n"
            f"Merged value:"
        )

        try:
            merged = self.llm.complete(
                system,
                [{"role": "user", "content": user_msg}],
            ).strip()
            # Reject obviously bad responses
            if not merged or len(merged) > 500:
                return None
            return merged
        except Exception as exc:
            logger.debug("memory_merge_llm_failed: %s", exc)
            return None

    # ---- Stage 4: Active Forgetting ----

    def _forget(self, user_id: str, scored: list[ScoredMemory]) -> int:
        """Remove lowest-importance memories if count exceeds threshold.

        Policies are never forgotten (they have importance >= 2.0).
        Only preference-type memories with low importance are candidates.
        """
        if not self.config.enable_active_forgetting:
            return 0

        count = self.store.count_memories(user_id)
        if count <= self.config.max_memories:
            return 0

        # How many to forget
        excess = count - self.config.max_memories
        to_forget = min(excess, self.config.forget_batch_size)

        # Sort by importance ascending (lowest first)
        scored_sorted = sorted(scored, key=lambda x: x.importance)

        forgotten = 0
        for mem in scored_sorted:
            if forgotten >= to_forget:
                break
            # Never forget policies
            if mem.memory_type == "policy":
                continue
            # Never forget very recent memories (< 1 day old)
            if time.time() - mem.updated_at < 86400:
                continue

            self.store.delete_profile_key(user_id, mem.key)
            forgotten += 1
            logger.debug(
                "memory_forgotten: user=%s key=%s importance=%.3f",
                user_id, mem.key, mem.importance,
            )

        return forgotten

    # ---- Helpers ----

    def _update_timestamps(
        self, user_id: str, key: str, created_at: float, updated_at: float,
    ) -> None:
        """Update created_at and updated_at for a memory entry."""
        try:
            with self.store._lock:
                self.store._get_conn().execute(
                    "UPDATE user_profile SET created_at=?, updated_at=? "
                    "WHERE user_id=? AND key=?",
                    (created_at, updated_at, user_id, key),
                )
                self.store._get_conn().commit()
        except Exception as exc:
            logger.debug("update_timestamps_failed: %s", exc)
