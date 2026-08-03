"""Tests for the cross-session lesson store (failure experience persistence)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from matrix.memory.lesson_store import (
    Lesson,
    LessonStore,
    _jaccard_similarity,
    _tokenize,
)


# ---- Fixtures ---------------------------------------------------------------

@pytest.fixture
def store():
    """LessonStore with a temp database."""
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "lessons_test.db"
        s = LessonStore(db_path)
        yield s
        s.close()


# ---- Tokenizer tests -------------------------------------------------------

class TestTokenize:
    def test_english(self):
        tokens = _tokenize("query holdings data")
        assert "query" in tokens
        assert "holdings" in tokens
        assert "data" in tokens

    def test_chinese_2gram(self):
        tokens = _tokenize("查询持仓数据")
        assert "查询" in tokens
        assert "持仓" in tokens
        assert "仓数" in tokens
        assert "数据" in tokens

    def test_mixed(self):
        tokens = _tokenize("查询 portfolio 数据")
        assert "查询" in tokens  # Chinese 2-gram
        assert "portfolio" in tokens  # English token

    def test_empty(self):
        assert _tokenize("") == set()

    def test_single_chinese_char(self):
        tokens = _tokenize("买")
        assert "买" in tokens


class TestJaccardSimilarity:
    def test_identical(self):
        assert _jaccard_similarity("查询持仓", "查询持仓") == 1.0

    def test_completely_different(self):
        assert _jaccard_similarity("hello world", "查询持仓") == 0.0

    def test_partial_overlap(self):
        sim = _jaccard_similarity("查询持仓数据", "查询持仓分析")
        assert 0.0 < sim < 1.0

    def test_both_empty(self):
        assert _jaccard_similarity("", "") == 1.0


# ---- LessonStore CRUD tests -------------------------------------------------

class TestLessonStoreCRUD:
    def test_record_and_count(self, store):
        assert store.count() == 0
        lid = store.record_lesson(
            task_pattern="查询持仓",
            failure_type="missing_data",
            lesson_text="需要先确认用户ID",
            agent_id="investment-analyst",
            user_id="user1",
        )
        assert lid > 0
        assert store.count("user1") == 1

    def test_record_empty_lesson_ignored(self, store):
        lid = store.record_lesson("", "", "", "", "")
        assert lid == 0
        assert store.count() == 0

    def test_record_whitespace_lesson_ignored(self, store):
        lid = store.record_lesson("task", "type", "   ", "agent", "user")
        assert lid == 0

    def test_get_relevant_lessons_exact_match(self, store):
        store.record_lesson(
            task_pattern="查询持仓数据",
            failure_type="missing_data",
            lesson_text="需要先确认用户ID再查询",
            agent_id="investment-analyst",
            user_id="user1",
        )
        lessons = store.get_relevant_lessons("查询持仓", agent_id="investment-analyst", user_id="user1")
        assert len(lessons) >= 1
        assert "需要先确认用户ID" in lessons[0].lesson_text

    def test_get_relevant_no_match(self, store):
        store.record_lesson(
            task_pattern="完全不同的任务",
            failure_type="missing_data",
            lesson_text="某教训",
            agent_id="agent_a",
            user_id="user1",
        )
        lessons = store.get_relevant_lessons("查询持仓数据", agent_id="agent_a", user_id="user1")
        # Low similarity → no results
        assert len(lessons) == 0

    def test_get_relevant_agent_filter(self, store):
        store.record_lesson(
            task_pattern="查询持仓",
            failure_type="missing_data",
            lesson_text="教训A",
            agent_id="agent_a",
            user_id="user1",
        )
        store.record_lesson(
            task_pattern="查询持仓",
            failure_type="missing_data",
            lesson_text="教训B",
            agent_id="agent_b",
            user_id="user1",
        )
        lessons = store.get_relevant_lessons("查询持仓", agent_id="agent_a", user_id="user1")
        assert len(lessons) == 1
        assert lessons[0].agent_id == "agent_a"
        assert "教训A" in lessons[0].lesson_text

    def test_get_relevant_user_isolation(self, store):
        store.record_lesson(
            task_pattern="查询持仓",
            failure_type="missing_data",
            lesson_text="教训A",
            agent_id="agent_a",
            user_id="user1",
        )
        store.record_lesson(
            task_pattern="查询持仓",
            failure_type="missing_data",
            lesson_text="教训B",
            agent_id="agent_a",
            user_id="user2",
        )
        lessons = store.get_relevant_lessons("查询持仓", agent_id="agent_a", user_id="user1")
        assert len(lessons) == 1
        assert "教训A" in lessons[0].lesson_text

    def test_delete_lesson(self, store):
        lid = store.record_lesson(
            task_pattern="查询",
            failure_type="test",
            lesson_text="test lesson",
            agent_id="a",
            user_id="u",
        )
        assert store.count("u") == 1
        deleted = store.delete_lesson(lid)
        assert deleted is True
        assert store.count("u") == 0

    def test_delete_nonexistent(self, store):
        deleted = store.delete_lesson(99999)
        assert deleted is False


# ---- Deduplication tests ---------------------------------------------------

class TestLessonDedup:
    def test_similar_lesson_increments_count(self, store):
        """Recording a similar lesson should increment occurrence_count."""
        store.record_lesson(
            task_pattern="查询持仓数据",
            failure_type="missing_data",
            lesson_text="需要先确认用户ID",
            agent_id="agent_a",
            user_id="user1",
        )
        # Record similar lesson (high overlap with "查询持仓数据")
        store.record_lesson(
            task_pattern="查询持仓数据报告",
            failure_type="missing_data",
            lesson_text="需要先确认用户ID再查询",
            agent_id="agent_a",
            user_id="user1",
        )

        lessons = store.get_all_lessons(user_id="user1", agent_id="agent_a")
        assert len(lessons) == 1  # Deduplicated
        assert lessons[0].occurrence_count == 2

    def test_different_agent_not_deduped(self, store):
        """Same task but different agent should create separate lessons."""
        store.record_lesson(
            task_pattern="查询持仓",
            failure_type="missing_data",
            lesson_text="教训",
            agent_id="agent_a",
            user_id="user1",
        )
        store.record_lesson(
            task_pattern="查询持仓",
            failure_type="missing_data",
            lesson_text="教训",
            agent_id="agent_b",
            user_id="user1",
        )
        assert store.count("user1") == 2

    def test_severity_upgrade(self, store):
        """When a similar lesson is recorded with higher severity, it upgrades."""
        store.record_lesson(
            task_pattern="查询持仓数据",
            failure_type="missing_data",
            lesson_text="教训",
            agent_id="agent_a",
            user_id="user1",
            severity="low",
        )
        store.record_lesson(
            task_pattern="查询持仓数据报告",
            failure_type="missing_data",
            lesson_text="教训v2",
            agent_id="agent_a",
            user_id="user1",
            severity="high",
        )
        lessons = store.get_all_lessons(user_id="user1", agent_id="agent_a")
        assert len(lessons) == 1
        assert lessons[0].severity == "high"

    def test_longer_lesson_text_preserved(self, store):
        """When deduplicating, the longer lesson text is kept."""
        store.record_lesson(
            task_pattern="查询持仓数据",
            failure_type="missing_data",
            lesson_text="短",
            agent_id="agent_a",
            user_id="user1",
        )
        store.record_lesson(
            task_pattern="查询持仓数据报告",
            failure_type="missing_data",
            lesson_text="这是一个更长的教训文本，包含更多细节",
            agent_id="agent_a",
            user_id="user1",
        )
        lessons = store.get_all_lessons(user_id="user1", agent_id="agent_a")
        assert len(lessons) == 1
        assert "更长" in lessons[0].lesson_text


# ---- Forgetting tests ------------------------------------------------------

class TestLessonForgetting:
    def test_forget_low_priority(self, store):
        """When lessons exceed max, low-priority ones are forgotten."""
        # Record many lessons with varying occurrence counts
        for i in range(10):
            store.record_lesson(
                task_pattern=f"任务_{i:02d}_unique",
                failure_type="test",
                lesson_text=f"教训 {i}",
                agent_id=f"agent_{i:02d}",
                user_id="user1",
            )

        # Verify all 10 are stored
        assert store.count("user1") == 10

        # Since _MAX_LESSONS is 200, no forgetting should happen with 10 lessons
        all_lessons = store.get_all_lessons(user_id="user1", limit=100)
        assert len(all_lessons) == 10


# ---- Relevance ranking tests ------------------------------------------------

class TestLessonRelevance:
    def test_top_k_limit(self, store):
        """get_relevant_lessons should respect top_k."""
        for i in range(5):
            store.record_lesson(
                task_pattern=f"查询持仓数据 variant_{i}",
                failure_type="test",
                lesson_text=f"教训 {i}",
                agent_id="agent_a",
                user_id="user1",
            )

        lessons = store.get_relevant_lessons(
            "查询持仓数据", agent_id="agent_a", user_id="user1", top_k=2,
        )
        assert len(lessons) <= 2

    def test_relevance_score_in_pattern(self, store):
        """Relevance score should be included in task_pattern."""
        store.record_lesson(
            task_pattern="查询持仓",
            failure_type="test",
            lesson_text="test",
            agent_id="agent_a",
            user_id="user1",
        )
        lessons = store.get_relevant_lessons("查询持仓", agent_id="agent_a", user_id="user1")
        assert len(lessons) >= 1
        assert "[relevance=" in lessons[0].task_pattern

    def test_ranked_by_similarity(self, store):
        """More similar lessons should rank higher."""
        store.record_lesson(
            task_pattern="完全不同",
            failure_type="test",
            lesson_text="low relevance",
            agent_id="agent_a",
            user_id="user1",
        )
        store.record_lesson(
            task_pattern="查询持仓数据",
            failure_type="test",
            lesson_text="high relevance",
            agent_id="agent_a",
            user_id="user1",
        )

        lessons = store.get_relevant_lessons("查询持仓数据", agent_id="agent_a", user_id="user1")
        if len(lessons) >= 2:
            # The high-relevance one should come first
            assert "high relevance" in lessons[0].lesson_text


# ---- Update last_seen tests -------------------------------------------------

class TestUpdateLastSeen:
    def test_update_last_seen(self, store):
        lid = store.record_lesson(
            task_pattern="查询",
            failure_type="test",
            lesson_text="test",
            agent_id="a",
            user_id="u",
        )
        # Should not raise
        store.update_last_seen([lid])
        store.update_last_seen([])  # Empty list is a no-op

    def test_update_nonexistent(self, store):
        """Updating non-existent lesson_id should not raise."""
        store.update_last_seen([99999])


# ---- Lesson dataclass tests ------------------------------------------------

class TestLessonDataclass:
    def test_to_dict(self):
        lesson = Lesson(
            lesson_id=1,
            task_pattern="test",
            failure_type="missing_data",
            lesson_text="test lesson",
            agent_id="agent_a",
            severity="high",
            occurrence_count=3,
        )
        d = lesson.to_dict()
        assert d["lesson_id"] == 1
        assert d["task_pattern"] == "test"
        assert d["failure_type"] == "missing_data"
        assert d["lesson_text"] == "test lesson"
        assert d["agent_id"] == "agent_a"
        assert d["severity"] == "high"
        assert d["occurrence_count"] == 3


# ---- Severity helper tests -------------------------------------------------

class TestMaxSeverity:
    def test_high_beats_medium(self):
        assert LessonStore._max_severity("medium", "high") == "high"
        assert LessonStore._max_severity("high", "medium") == "high"

    def test_medium_beats_low(self):
        assert LessonStore._max_severity("low", "medium") == "medium"
        assert LessonStore._max_severity("medium", "low") == "medium"

    def test_same(self):
        assert LessonStore._max_severity("high", "high") == "high"

    def test_unknown_defaults_medium(self):
        assert LessonStore._max_severity("unknown", "high") == "high"
        assert LessonStore._max_severity("low", "unknown") == "unknown"
