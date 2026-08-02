"""Semantic skill router: L1 embedding-based matching.

Architecture (industry-standard tiered routing):

    L0: Keyword match  (SkillDefinition.matches)     → O(n) string ops
    L1: Semantic match (this module, cosine sim)     → O(n) vector ops
    L2: LLM plan       (commander_plan_node)          → 1 LLM call

L0 is tried first (zero cost). If L0 misses, L1 uses pre-computed
skill-description embeddings to find the closest match by cosine
similarity. If the top score exceeds a threshold, the skill is
selected. Otherwise, the query falls through to L2 (LLM planning).

The LocalEmbedder from rag/embedder.py is reused — it loads
BAAI/bge-small-zh-v1.5 when sentence-transformers is available,
and degrades to hash-based pseudo-vectors for dev/test.
"""

from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from ..rag.embedder import LocalEmbedder
from .loader import SkillDefinition

logger = logging.getLogger("matrix.skills.router")

# ── Config ───────────────────────────────────────────────────────────────────

# Environment variable names
ENV_SEMANTIC_THRESHOLD = "MATRIX_SEMANTIC_THRESHOLD"
ENV_SEMANTIC_CACHE_SIZE = "MATRIX_SEMANTIC_CACHE_SIZE"

# Minimum cosine similarity to accept a semantic match.
# Below this, the query falls through to LLM planning.
DEFAULT_SEMANTIC_THRESHOLD = 0.65

# How many top candidates to return from semantic search.
DEFAULT_TOP_K = 3

# Max number of query results to cache (LRU eviction).
DEFAULT_CACHE_SIZE = 128


def _get_threshold_from_env() -> float:
    """Read semantic threshold from env var, with validation."""
    raw = os.environ.get(ENV_SEMANTIC_THRESHOLD, "").strip()
    if not raw:
        return DEFAULT_SEMANTIC_THRESHOLD
    try:
        val = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %.2f", ENV_SEMANTIC_THRESHOLD, raw, DEFAULT_SEMANTIC_THRESHOLD)
        return DEFAULT_SEMANTIC_THRESHOLD
    if not 0.0 <= val <= 1.0:
        logger.warning("%s=%.3f out of [0,1], clamping", ENV_SEMANTIC_THRESHOLD, val)
        val = max(0.0, min(1.0, val))
    return val


def _get_cache_size_from_env() -> int:
    """Read cache size from env var, with validation."""
    raw = os.environ.get(ENV_SEMANTIC_CACHE_SIZE, "").strip()
    if not raw:
        return DEFAULT_CACHE_SIZE
    try:
        val = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %d", ENV_SEMANTIC_CACHE_SIZE, raw, DEFAULT_CACHE_SIZE)
        return DEFAULT_CACHE_SIZE
    if val < 0:
        return 0  # 0 = disable caching
    return val


@dataclass
class SkillMatch:
    """A single skill match result."""

    skill_name: str
    score: float
    skill: SkillDefinition


class SemanticRouter:
    """Pre-compute skill embeddings and route queries by cosine similarity.

    Usage::

        router = SemanticRouter(skills, embedder)
        result = router.route("帮我看看配置偏离")
        if result and result[0].score >= threshold:
            # use result[0].skill
        else:
            # fall through to LLM planning

    Configuration:
        - Threshold: ``MATRIX_SEMANTIC_THRESHOLD`` env var (default 0.65)
        - Cache size: ``MATRIX_SEMANTIC_CACHE_SIZE`` env var (default 128, 0=disable)
        - Both can also be overridden via constructor params.
    """

    def __init__(
        self,
        skills: list[SkillDefinition],
        embedder: LocalEmbedder | None = None,
        threshold: float | None = None,
        cache_size: int | None = None,
    ) -> None:
        self._threshold = threshold if threshold is not None else _get_threshold_from_env()
        self._embedder = embedder

        # Query cache: query_str → list[SkillMatch] (LRU eviction)
        self._cache_size = cache_size if cache_size is not None else _get_cache_size_from_env()
        self._query_cache: OrderedDict[str, list[SkillMatch]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        # name → SkillDefinition
        self._skills: dict[str, SkillDefinition] = {s.name: s for s in skills}

        # Pre-compute embeddings: name → vector
        self._embeddings: dict[str, list[float]] = {}
        if skills and embedder is not None:
            self._build_index(skills, embedder)

    def _build_index(self, skills: list[SkillDefinition], embedder: LocalEmbedder) -> None:
        """Encode all skill descriptions into vectors."""
        texts: list[str] = []
        names: list[str] = []
        for s in skills:
            # Combine name, title, and description for richer context
            text = f"{s.title} {s.description}".strip()
            if not text:
                text = s.name
            texts.append(text)
            names.append(s.name)

        try:
            vectors = embedder.encode(texts)
            for name, vec in zip(names, vectors):
                self._embeddings[name] = vec
            logger.info(
                "SemanticRouter: indexed %d skills (dim=%d)",
                len(self._embeddings),
                len(vectors[0]) if vectors else 0,
            )
        except Exception as exc:
            logger.error("SemanticRouter: failed to build index: %s", exc)
            # Degrade gracefully — route() will return empty list
            self._embeddings = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def route(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SkillMatch]:
        """Return ranked skill matches for a user query.

        Returns up to ``top_k`` matches sorted by descending score.
        If no embedder or empty index, returns an empty list (fall through).

        Results are cached per query (LRU, up to ``cache_size`` entries).
        """
        if not self._embeddings or self._embedder is None:
            return []

        # ── Check cache ──
        cache_key = f"{query}#{top_k}"
        if self._cache_size > 0 and cache_key in self._query_cache:
            self._cache_hits += 1
            # Move to end (most recently used)
            self._query_cache.move_to_end(cache_key)
            return list(self._query_cache[cache_key])  # return a copy

        self._cache_misses += 1

        try:
            query_vec = self._embedder.encode_single(query)
        except Exception as exc:
            logger.warning("SemanticRouter: query encoding failed: %s", exc)
            return []

        scored: list[SkillMatch] = []
        for name, skill_vec in self._embeddings.items():
            score = _cosine_similarity(query_vec, skill_vec)
            skill = self._skills.get(name)
            if skill is None:
                continue
            scored.append(SkillMatch(
                skill_name=name,
                score=score,
                skill=skill,
            ))

        scored.sort(key=lambda m: m.score, reverse=True)
        result = scored[:top_k]

        # ── Store in cache ──
        if self._cache_size > 0:
            self._query_cache[cache_key] = list(result)  # store a copy
            # Evict oldest if over capacity
            while len(self._query_cache) > self._cache_size:
                self._query_cache.popitem(last=False)

        return result

    def best_match(self, query: str) -> Optional[SkillMatch]:
        """Return the single best match, or None if no skills indexed."""
        results = self.route(query, top_k=1)
        return results[0] if results else None

    def should_accept(self, match: SkillMatch) -> bool:
        """Check if a match's score is above the acceptance threshold."""
        return match.score >= self._threshold

    @property
    def is_available(self) -> bool:
        """True if the router has a working embedding index."""
        return bool(self._embeddings)

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    @property
    def cache_size(self) -> int:
        return self._cache_size

    @property
    def cache_stats(self) -> dict[str, int]:
        """Return cache statistics: hits, misses, entries, size."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "entries": len(self._query_cache),
            "max_size": self._cache_size,
        }

    def clear_cache(self) -> None:
        """Clear the query cache and reset hit/miss counters."""
        self._query_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


# ── Helpers ──────────────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero magnitude (avoiding division by zero).
    """
    if len(a) != len(b):
        logger.warning(
            "cosine_similarity: dimension mismatch %d vs %d, returning 0", len(a), len(b),
        )
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
