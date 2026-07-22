"""Tests for the memory evolution pipeline (importance, conflict, consolidation, forgetting)."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from matrix.store import SessionStore
from matrix.memory import EvolutionConfig, MemoryEvolution, EvolutionReport
from matrix.memory.evolution import (
    ScoredMemory,
    _normalize_key,
    _jaccard_similarity,
    _key_similarity,
    _values_conflict,
)


# ---- Fixtures ---------------------------------------------------------------

@pytest.fixture
def store():
    with Path("/tmp") as _tmp:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "evolution_test.db"
            s = SessionStore(str(db_path))
            yield s


@pytest.fixture
def evolution(store):
    """Evolution without LLM (uses text-based consolidation)."""
    return MemoryEvolution(store, config=EvolutionConfig(
        enable_llm_consolidation=False,
        max_memories=5,
        forget_batch_size=3,
    ))


@pytest.fixture
def evolution_with_llm(store):
    """Evolution with mock LLM."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "merged: combined value from A and B"
    return MemoryEvolution(store, config=EvolutionConfig(
        enable_llm_consolidation=True,
        max_memories=5,
    ), llm=mock_llm), mock_llm


# ---- Utility function tests -------------------------------------------------

class TestNormalizeKey:
    def test_lowercase(self):
        assert _normalize_key("Dark_Mode") == "dark"

    def test_strip_prefixes(self):
        assert _normalize_key("user_language") == "language"
        assert _normalize_key("prefers_theme") == "theme"
        assert _normalize_key("favorite_color") == "color"

    def test_strip_suffixes(self):
        assert _normalize_key("dark_mode") == "dark"
        assert _normalize_key("language_preference") == "language"
        assert _normalize_key("lang_pref") == "lang"

    def test_strip_stacked_suffixes(self):
        # "dark_mode_pref" → strip "_pref" → "dark_mode" → strip "_mode" → "dark"
        assert _normalize_key("dark_mode_pref") == "dark"

    def test_separators(self):
        assert _normalize_key("dark-mode") == "dark"
        assert _normalize_key("dark mode") == "dark"


class TestJaccardSimilarity:
    def test_identical(self):
        assert _jaccard_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _jaccard_similarity("cat", "dog") == 0.0

    def test_partial_overlap(self):
        sim = _jaccard_similarity("hello world", "hello there")
        assert 0.0 < sim < 1.0

    def test_empty_strings(self):
        assert _jaccard_similarity("", "") == 1.0
        assert _jaccard_similarity("a", "") == 0.0


class TestKeySimilarity:
    def test_exact_match(self):
        assert _key_similarity("dark_mode", "dark_mode") == 1.0

    def test_normalized_match(self):
        assert _key_similarity("user_dark_mode", "dark_mode_preference") == 1.0

    def test_no_match(self):
        assert _key_similarity("language", "color") == 0.0


class TestValuesConflict:
    def test_same_value(self):
        assert not _values_conflict("yes", "yes")

    def test_boolean_conflict(self):
        assert _values_conflict("yes", "no")
        assert _values_conflict("true", "false")
        assert _values_conflict("on", "off")

    def test_chinese_boolean(self):
        assert _values_conflict("是", "否")
        assert _values_conflict("开", "关")

    def test_no_conflict_different_topics(self):
        assert not _values_conflict("red", "blue")
        assert not _values_conflict("Chinese", "English")

    def test_negation_pattern(self):
        assert _values_conflict("喜欢", "不喜欢")
        assert _values_conflict("like", "not like")


# ---- Stage 1: Importance Scoring --------------------------------------------

class TestImportanceScoring:
    def test_policy_always_max(self, evolution):
        now = time.time()
        memories = [{
            "key": "hard_rule", "value": "never sell",
            "memory_type": "policy",
            "created_at": now - 100 * 86400,
            "updated_at": now - 100 * 86400,
        }]
        scored = evolution._score_importance(memories)
        assert scored[0].importance == pytest.approx(2.0)  # policy weight * no decay

    def test_preference_decays(self, evolution):
        now = time.time()
        old_mem = {
            "key": "old_pref", "value": "val",
            "memory_type": "preference",
            "created_at": now - 60 * 86400,
            "updated_at": now - 60 * 86400,
        }
        new_mem = {
            "key": "new_pref", "value": "val",
            "memory_type": "preference",
            "created_at": now,
            "updated_at": now,
        }
        scored = evolution._score_importance([old_mem, new_mem])
        # New memory should have higher importance
        assert scored[0].key == "new_pref"
        assert scored[0].importance > scored[1].importance

    def test_recency_boost(self, evolution):
        now = time.time()
        recent_mem = {
            "key": "recent", "value": "val",
            "memory_type": "preference",
            "created_at": now,
            "updated_at": now,
        }
        scored = evolution._score_importance([recent_mem])
        # Recent memory should have boost factor applied
        # importance = decay(1.0) * type_weight(1.0) * recency_boost(1.5) = 1.5
        assert scored[0].importance == pytest.approx(1.5, abs=0.01)

    def test_sorted_descending(self, evolution):
        now = time.time()
        memories = [
            {"key": f"k{i}", "value": "v", "memory_type": "preference",
             "created_at": now - i * 86400, "updated_at": now - i * 86400}
            for i in range(5)
        ]
        scored = evolution._score_importance(memories)
        for i in range(len(scored) - 1):
            assert scored[i].importance >= scored[i + 1].importance


# ---- Stage 2: Conflict Resolution -------------------------------------------

class TestConflictResolution:
    def test_boolean_conflict_resolved(self, store, evolution):
        now = time.time()
        store.upsert_profile("u1", "dark_mode", "yes", memory_type="preference")
        # Manually set older timestamp
        with store._lock:
            store._get_conn().execute(
                "UPDATE user_profile SET updated_at=? WHERE user_id=? AND key=?",
                (now - 86400, "u1", "dark_mode"),
            )
            store._get_conn().commit()
        store.upsert_profile("u1", "dark_mode_preference", "no", memory_type="preference")

        memories = store.get_all_memories("u1")
        scored = evolution._score_importance(memories)
        scored, conflicts = evolution._resolve_conflicts("u1", scored)

        assert conflicts == 1
        assert len(scored) == 1
        # The newer one (dark_mode_preference=no) should survive
        remaining = store.get_profile("u1")
        assert len(remaining) == 1
        assert "no" in remaining.values()

    def test_no_conflict_different_keys(self, store, evolution):
        store.upsert_profile("u1", "language", "Chinese")
        store.upsert_profile("u1", "color", "red")

        memories = store.get_all_memories("u1")
        scored = evolution._score_importance(memories)
        scored, conflicts = evolution._resolve_conflicts("u1", scored)

        assert conflicts == 0
        assert len(scored) == 2

    def test_no_conflict_same_value(self, store, evolution):
        store.upsert_profile("u1", "dark_mode", "yes")
        store.upsert_profile("u1", "dark_mode_pref", "yes")

        memories = store.get_all_memories("u1")
        scored = evolution._score_importance(memories)
        scored, conflicts = evolution._resolve_conflicts("u1", scored)

        assert conflicts == 0  # Same value, not a conflict


# ---- Stage 3: Consolidation -------------------------------------------------

class TestConsolidation:
    def test_text_consolidation_removes_exact_dup(self, store, evolution):
        now = time.time()
        store.upsert_profile("u1", "language", "Chinese")
        with store._lock:
            store._get_conn().execute(
                "UPDATE user_profile SET updated_at=? WHERE user_id=? AND key=?",
                (now - 86400, "u1", "language"),
            )
            store._get_conn().commit()
        store.upsert_profile("u1", "language_pref", "Chinese")

        memories = store.get_all_memories("u1")
        scored = evolution._score_importance(memories)
        scored, consolidated = evolution._consolidate("u1", scored)

        assert consolidated == 1
        assert len(scored) == 1

    def test_text_consolidation_keeps_different_values(self, store, evolution):
        store.upsert_profile("u1", "language", "Chinese")
        store.upsert_profile("u1", "language_pref", "English")

        memories = store.get_all_memories("u1")
        scored = evolution._score_importance(memories)
        scored, consolidated = evolution._consolidate("u1", scored)

        # Values are different but not conflicting (not boolean), so no consolidation
        assert consolidated == 0
        assert len(scored) == 2

    def test_llm_consolidation_merges(self, store, evolution_with_llm):
        evo, mock_llm = evolution_with_llm
        now = time.time()
        store.upsert_profile("u1", "investment_style", "conservative, focus on bonds")
        with store._lock:
            store._get_conn().execute(
                "UPDATE user_profile SET updated_at=? WHERE user_id=? AND key=?",
                (now - 86400, "u1", "investment_style"),
            )
            store._get_conn().commit()
        store.upsert_profile("u1", "investment_style_pref", "also likes index funds")

        memories = store.get_all_memories("u1")
        scored = evo._score_importance(memories)
        scored, consolidated = evo._consolidate("u1", scored)

        assert consolidated == 1
        assert len(scored) == 1
        mock_llm.complete.assert_called_once()

    def test_llm_consolidation_falls_back_on_error(self, store):
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = RuntimeError("LLM error")
        evo = MemoryEvolution(store, config=EvolutionConfig(
            enable_llm_consolidation=True,
        ), llm=mock_llm)

        now = time.time()
        store.upsert_profile("u1", "topic_a", "value one two three")
        with store._lock:
            store._get_conn().execute(
                "UPDATE user_profile SET updated_at=? WHERE user_id=? AND key=?",
                (now - 86400, "u1", "topic_a"),
            )
            store._get_conn().commit()
        store.upsert_profile("u1", "topic_a_pref", "value one two three")

        memories = store.get_all_memories("u1")
        scored = evo._score_importance(memories)
        # Should fall back to text consolidation (high Jaccard similarity = exact dup)
        scored, consolidated = evo._consolidate("u1", scored)
        assert consolidated == 1


# ---- Stage 4: Active Forgetting ---------------------------------------------

class TestActiveForgetting:
    def test_forgetting_triggered(self, store):
        evo = MemoryEvolution(store, config=EvolutionConfig(
            enable_llm_consolidation=False,
            max_memories=3,
            forget_batch_size=2,
        ))
        now = time.time()
        # Add 5 memories, all preferences, varying age
        for i in range(5):
            store.upsert_profile("u1", f"pref_{i}", f"value_{i}")
            with store._lock:
                store._get_conn().execute(
                    "UPDATE user_profile SET updated_at=? WHERE user_id=? AND key=?",
                    (now - (i + 2) * 86400, "u1", f"pref_{i}"),
                )
                store._get_conn().commit()

        memories = store.get_all_memories("u1")
        scored = evo._score_importance(memories)
        forgotten = evo._forget("u1", scored)

        assert forgotten == 2
        assert store.count_memories("u1") == 3

    def test_policy_never_forgotten(self, store):
        evo = MemoryEvolution(store, config=EvolutionConfig(
            enable_llm_consolidation=False,
            max_memories=2,
            forget_batch_size=5,
        ))
        now = time.time()
        # Add 3 policies + 2 preferences = 5 total, max=2
        for i in range(3):
            store.upsert_profile("u1", f"policy_{i}", f"rule_{i}", memory_type="policy")
            with store._lock:
                store._get_conn().execute(
                    "UPDATE user_profile SET updated_at=? WHERE user_id=? AND key=?",
                    (now - (i + 10) * 86400, "u1", f"policy_{i}"),
                )
                store._get_conn().commit()
        for i in range(2):
            store.upsert_profile("u1", f"pref_{i}", f"value_{i}")
            with store._lock:
                store._get_conn().execute(
                    "UPDATE user_profile SET updated_at=? WHERE user_id=? AND key=?",
                    (now - (i + 10) * 86400, "u1", f"pref_{i}"),
                )
                store._get_conn().commit()

        memories = store.get_all_memories("u1")
        scored = evo._score_importance(memories)
        forgotten = evo._forget("u1", scored)

        # Only preferences can be forgotten; max=2, total=5, excess=3
        # But only 2 preferences exist, so at most 2 forgotten
        assert forgotten <= 2
        # All 3 policies must survive
        remaining = store.get_profile("u1")
        policy_keys = [k for k in remaining if k.startswith("policy_")]
        assert len(policy_keys) == 3

    def test_recent_memories_protected(self, store):
        evo = MemoryEvolution(store, config=EvolutionConfig(
            enable_llm_consolidation=False,
            max_memories=1,
            forget_batch_size=5,
        ))
        now = time.time()
        # Add 3 very recent memories
        for i in range(3):
            store.upsert_profile("u1", f"recent_{i}", f"val_{i}")

        memories = store.get_all_memories("u1")
        scored = evo._score_importance(memories)
        forgotten = evo._forget("u1", scored)

        # All are <1 day old, so none should be forgotten
        assert forgotten == 0

    def test_no_forgetting_below_threshold(self, store):
        evo = MemoryEvolution(store, config=EvolutionConfig(
            enable_llm_consolidation=False,
            max_memories=10,
        ))
        for i in range(5):
            store.upsert_profile("u1", f"pref_{i}", f"val_{i}")

        memories = store.get_all_memories("u1")
        scored = evo._score_importance(memories)
        forgotten = evo._forget("u1", scored)

        assert forgotten == 0
        assert store.count_memories("u1") == 5

    def test_forgetting_disabled(self, store):
        evo = MemoryEvolution(store, config=EvolutionConfig(
            enable_llm_consolidation=False,
            enable_active_forgetting=False,
            max_memories=1,
        ))
        for i in range(5):
            store.upsert_profile("u1", f"pref_{i}", f"val_{i}")

        memories = store.get_all_memories("u1")
        scored = evo._score_importance(memories)
        forgotten = evo._forget("u1", scored)

        assert forgotten == 0
        assert store.count_memories("u1") == 5


# ---- Full pipeline (evolve) -------------------------------------------------

class TestEvolvePipeline:
    def test_empty_store(self, evolution):
        report = evolution.evolve("nobody")
        assert report.total_before == 0
        assert report.total_after == 0
        assert report.conflicts_resolved == 0
        assert report.memories_consolidated == 0
        assert report.memories_forgotten == 0

    def test_no_changes_needed(self, store, evolution):
        """Few memories, no conflicts, no duplicates."""
        store.upsert_profile("u1", "language", "Chinese")
        store.upsert_profile("u1", "city", "Shanghai")

        report = evolution.evolve("u1")
        assert report.total_before == 2
        assert report.total_after == 2
        assert report.conflicts_resolved == 0
        assert report.memories_consolidated == 0
        assert report.memories_forgotten == 0

    def test_full_pipeline_with_conflicts_and_forgetting(self, store):
        evo = MemoryEvolution(store, config=EvolutionConfig(
            enable_llm_consolidation=False,
            max_memories=2,
            forget_batch_size=3,
        ))
        now = time.time()

        # Conflict pair
        store.upsert_profile("u1", "dark_mode", "yes")
        with store._lock:
            store._get_conn().execute(
                "UPDATE user_profile SET updated_at=? WHERE user_id=? AND key=?",
                (now - 86400, "u1", "dark_mode"),
            )
            store._get_conn().commit()
        store.upsert_profile("u1", "dark_mode_pref", "no")

        # Extra memories to trigger forgetting
        for i in range(3):
            store.upsert_profile("u1", f"old_pref_{i}", f"val_{i}")
            with store._lock:
                store._get_conn().execute(
                    "UPDATE user_profile SET updated_at=? WHERE user_id=? AND key=?",
                    (now - (i + 5) * 86400, "u1", f"old_pref_{i}"),
                )
                store._get_conn().commit()

        report = evo.evolve("u1")

        assert report.total_before == 5
        assert report.conflicts_resolved >= 1
        assert report.total_after <= 2  # max_memories=2


# ---- Report formatting ------------------------------------------------------

class TestEvolutionReport:
    def test_str_representation(self):
        report = EvolutionReport(
            total_before=10,
            total_after=7,
            conflicts_resolved=1,
            memories_consolidated=1,
            memories_forgotten=1,
        )
        s = str(report)
        assert "10→7" in s
        assert "conflicts=1" in s
        assert "consolidated=1" in s
        assert "forgotten=1" in s

    def test_empty_report(self):
        report = EvolutionReport()
        s = str(report)
        assert "0→0" in s
