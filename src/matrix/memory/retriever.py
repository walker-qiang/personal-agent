"""Hybrid retrieval over long-term semantic memory.

Replaces the "dump every memory into the system prompt" strategy that
``SessionStore.get_profile_formatted`` used. Three things change:

1. **Selective**: only memories relevant to the current turn are injected.
   Policies stay mandatory (they are hard rules and few in number), while
   preferences compete for a limited number of slots.
2. **Hybrid scoring**: dense cosine similarity (when a real embedder is
   available) fused with BM25 over character bigrams, which handles Chinese
   without requiring jieba. Either signal alone is enough when the other is
   unavailable, so the retriever degrades instead of disappearing.
3. **Budgeted**: the assembled block is truncated to a token budget, so prompt
   cost stops growing linearly with the size of the memory store.

The index lives in the ``memory_embeddings`` table owned by ``SessionStore``,
so there is no second database and no external service.
"""

from __future__ import annotations

import logging
import math
import re
import struct
import time
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Fusion weights. Semantic carries more weight, but BM25 rescues exact-token
# matches ("贵州茅台", "600519") that embeddings blur.
_W_SEMANTIC = 0.65
_W_LEXICAL = 0.35

# BM25 parameters (standard defaults).
_BM25_K1 = 1.5
_BM25_B = 0.75

_CJK = r"[\u4e00-\u9fff]"
_TOKEN_RE = re.compile(r"[a-zA-Z_]{2,}|[0-9]{2,}|[\u4e00-\u9fff]")


@dataclass
class RetrievedMemory:
    """A memory plus the score that put it in the result set."""

    key: str
    value: str
    memory_type: str
    score: float = 0.0
    semantic: float = 0.0
    lexical: float = 0.0
    updated_at: float = 0.0
    fact_time: float = 0.0
    source_session_id: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "memory_type": self.memory_type,
            "score": round(self.score, 4),
            "semantic": round(self.semantic, 4),
            "lexical": round(self.lexical, 4),
            "updated_at": self.updated_at,
            "fact_time": self.fact_time,
            "source_session_id": self.source_session_id,
            "confidence": self.confidence,
        }


# ---- tokenization ----------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Tokenize mixed CN/EN text.

    Chinese uses character bigrams (no jieba dependency); ASCII runs and
    digit runs are kept whole so tickers and codes stay matchable.
    """
    text = (text or "").lower()
    tokens: list[str] = []
    # ASCII / digit runs first
    tokens.extend(re.findall(r"[a-z_]{2,}", text))
    tokens.extend(re.findall(r"[0-9]{2,}", text))
    # CJK bigrams plus a trailing unigram so single-character queries match
    chars = re.findall(_CJK, text)
    for i in range(len(chars) - 1):
        tokens.append(chars[i] + chars[i + 1])
    if len(chars) == 1:
        tokens.append(chars[0])
    return tokens


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


# ---- BM25 over a tiny in-memory corpus -------------------------------------


class _BM25:
    """BM25 over a corpus rebuilt per query.

    Rebuilding is O(n) which is irrelevant at memory-store scale (tens to
    low hundreds of entries) and avoids keeping a stale index in sync.
    """

    def __init__(self, docs: list[list[str]]) -> None:
        self.docs = docs
        self.lengths = [len(d) for d in docs]
        self.avg_len = (sum(self.lengths) / len(docs)) if docs else 0.0
        self.df: dict[str, int] = {}
        for doc in docs:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1
        self.n = len(docs)

    def score(self, query_tokens: list[str]) -> list[float]:
        if not self.n:
            return []
        scores = [0.0] * self.n
        for token in query_tokens:
            df = self.df.get(token, 0)
            if df == 0:
                continue
            # Robertson/Sparck-Jones idf, floored at 0 to avoid negatives
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(self.docs):
                tf = doc.count(token)
                if not tf:
                    continue
                length = self.lengths[i] or 1
                denom = tf + _BM25_K1 * (
                    1.0 - _BM25_B + _BM25_B * length / (self.avg_len or 1.0)
                )
                scores[i] += idf * (tf * (_BM25_K1 + 1.0)) / denom
        return scores


# ---- retriever -------------------------------------------------------------


class MemoryRetriever:
    """Semantic + lexical retrieval over ``user_profile``."""

    def __init__(
        self,
        store: Any,
        embedder: Any | None = None,
        enable_semantic: bool = True,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._enable_semantic = enable_semantic and embedder is not None
        self._dim: int = 0
        self._model_name: str = ""
        if self._enable_semantic:
            try:
                probe = self._embedder.encode(["dimension probe"])
                if probe and probe[0]:
                    self._dim = len(probe[0])
                    self._model_name = str(
                        getattr(self._embedder, "model_name", "unknown"),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory_retriever: embedder probe failed: %s", exc)
                self._enable_semantic = False
        logger.info(
            "memory_retriever: ready semantic=%s dim=%d model=%s",
            self._enable_semantic, self._dim, self._model_name,
        )

    # ---- index maintenance -------------------------------------------------

    @property
    def semantic_enabled(self) -> bool:
        return self._enable_semantic

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not self._enable_semantic or not texts:
            return []
        try:
            return self._embedder.encode(texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory_retriever: encode failed: %s", exc)
            return []

    @staticmethod
    def _memory_text(key: str, value: str) -> str:
        """Text that gets indexed — key and value, both matter."""
        return f"{key}：{value}" if key else str(value)

    def index_user(self, user_id: str) -> int:
        """(Re)build the vector index for every active memory of a user."""
        if not self._enable_semantic:
            return 0
        memories = self._store.get_all_memories(user_id)
        if not memories:
            return 0
        texts = [self._memory_text(m["key"], m["value"]) for m in memories]
        vectors = self._embed(texts)
        if len(vectors) != len(memories):
            return 0
        now = time.time()
        with self._store._lock:
            conn = self._store._get_conn()
            for mem, vec in zip(memories, vectors):
                conn.execute(
                    "INSERT INTO memory_embeddings "
                    "(user_id, key, dim, vector, model, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(user_id, key) DO UPDATE SET "
                    "dim=excluded.dim, vector=excluded.vector, "
                    "model=excluded.model, updated_at=excluded.updated_at",
                    (
                        user_id, mem["key"], len(vec), _pack(vec),
                        self._model_name, now,
                    ),
                )
            conn.commit()
        logger.info(
            "memory_retriever: indexed user=%s entries=%d", user_id, len(memories),
        )
        return len(memories)

    def index_keys(self, user_id: str, keys_and_values: list[tuple[str, str]]) -> int:
        """Index (or re-index) a handful of memories after a write."""
        if not self._enable_semantic or not keys_and_values:
            return 0
        texts = [self._memory_text(k, v) for k, v in keys_and_values]
        vectors = self._embed(texts)
        if len(vectors) != len(keys_and_values):
            return 0
        now = time.time()
        with self._store._lock:
            conn = self._store._get_conn()
            for (key, _value), vec in zip(keys_and_values, vectors):
                conn.execute(
                    "INSERT INTO memory_embeddings "
                    "(user_id, key, dim, vector, model, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(user_id, key) DO UPDATE SET "
                    "dim=excluded.dim, vector=excluded.vector, "
                    "model=excluded.model, updated_at=excluded.updated_at",
                    (user_id, key, len(vec), _pack(vec), self._model_name, now),
                )
            conn.commit()
        return len(keys_and_values)

    def drop_keys(self, user_id: str, keys: Sequence[str]) -> None:
        if not keys:
            return
        with self._store._lock:
            conn = self._store._get_conn()
            conn.executemany(
                "DELETE FROM memory_embeddings WHERE user_id=? AND key=?",
                [(user_id, key) for key in keys],
            )
            conn.commit()

    def _load_vectors(self, user_id: str) -> dict[str, list[float]]:
        if not self._enable_semantic:
            return {}
        with self._store._lock:
            rows = self._store._get_conn().execute(
                "SELECT key, dim, vector FROM memory_embeddings WHERE user_id=?",
                (user_id,),
            ).fetchall()
        out: dict[str, list[float]] = {}
        for key, dim, blob in rows:
            if dim != self._dim:
                continue  # stored by a different model, ignore
            try:
                out[key] = _unpack(blob)
            except struct.error:
                continue
        return out

    # ---- search ------------------------------------------------------------

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 8,
        memory_type: str | None = None,
        time_from: float = 0.0,
        time_to: float = 0.0,
        min_score: float = 0.02,
    ) -> list[RetrievedMemory]:
        """Return memories ranked by fused semantic + lexical score.

        Args:
            query: the current user utterance (or a rewritten query).
            top_k: maximum number of memories to return.
            memory_type: restrict to ``"policy"`` or ``"preference"``.
            time_from / time_to: filter on the effective fact time —
                ``fact_time`` when present, otherwise ``updated_at``.
            min_score: drop anything below this fused score.
        """
        memories = self._store.get_all_memories(user_id)
        if memory_type:
            memories = [m for m in memories if m["memory_type"] == memory_type]
        if time_from or time_to:
            filtered = []
            for m in memories:
                effective = m["fact_time"] or m["updated_at"]
                if time_from and effective < time_from:
                    continue
                if time_to and effective > time_to:
                    continue
                filtered.append(m)
            memories = filtered
        if not memories:
            return []

        query_tokens = tokenize(query)
        corpus = [tokenize(self._memory_text(m["key"], m["value"])) for m in memories]
        bm25 = _BM25(corpus)
        lexical_scores = bm25.score(query_tokens)

        vectors: dict[str, list[float]] = {}
        query_vec: list[float] = []
        if self._enable_semantic and query_tokens:
            vectors = self._load_vectors(user_id)
            embedded = self._embed([query])
            query_vec = embedded[0] if embedded else []

        # Normalise each channel to 0..1 before fusing.
        max_lex = max(lexical_scores) if lexical_scores else 0.0
        results: list[RetrievedMemory] = []
        for i, mem in enumerate(memories):
            lex = (lexical_scores[i] / max_lex) if max_lex > 0 else 0.0
            sem = 0.0
            if query_vec:
                vec = vectors.get(mem["key"])
                if vec:
                    # cosine can be slightly negative; clamp to 0..1
                    sem = max(0.0, min(1.0, _cosine(query_vec, vec)))
            score = _W_SEMANTIC * sem + _W_LEXICAL * lex
            if score < min_score:
                continue
            results.append(
                RetrievedMemory(
                    key=mem["key"],
                    value=mem["value"],
                    memory_type=mem["memory_type"],
                    score=score,
                    semantic=sem,
                    lexical=lex,
                    updated_at=mem["updated_at"],
                    fact_time=mem["fact_time"],
                    source_session_id=mem["source_session_id"],
                    confidence=mem["confidence"],
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
