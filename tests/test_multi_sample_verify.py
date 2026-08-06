"""Tests for multi-sampling number verification (_multi_sample_verify).

Tests cover:
- 2-sample agreement (no 3rd sample needed)
- 2-sample disagreement → tiebreaker
- Cache hit (no LLM call needed)
- FABRICATED → full correction path
- All inconclusive → returns None
- Error handling (LLM raises → INCONCLUSIVE)
"""

import pytest
from unittest.mock import MagicMock, patch

from matrix.orchestration.nodes._helpers import (
    _multi_sample_verify,
    _single_verify,
    _cache_key,
    _cached_verdict,
    _cache_verdict,
    _VERIFY_CACHE,
    _full_verify_and_correct,
)


class FakeLLM:
    """Fake LLM that returns pre-configured responses for complete_json."""

    def __init__(self, responses: list[dict]):
        self._responses = responses
        self._call_count = 0
        self.call_history: list[tuple[str, float]] = []

    def complete_json(self, system, messages, schema=None, temperature=None):
        self.call_history.append((system[:50], temperature or 0.0))
        if self._call_count >= len(self._responses):
            return {"verdict": "INCONCLUSIVE"}
        resp = self._responses[self._call_count]
        self._call_count += 1
        if isinstance(resp, Exception):
            raise resp
        return resp


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the verification cache before each test."""
    _VERIFY_CACHE.clear()
    yield
    _VERIFY_CACHE.clear()


class TestCacheKey:
    def test_same_numbers_same_source_produces_same_key(self):
        numbers = ["17.74", "0.06"]
        source = "price=17.74 change=0.06"
        key1 = _cache_key(numbers, source)
        key2 = _cache_key(numbers, source)
        assert key1 == key2

    def test_different_numbers_produces_different_key(self):
        source = "price=17.74"
        key1 = _cache_key(["17.74"], source)
        key2 = _cache_key(["18.00"], source)
        assert key1 != key2

    def test_different_source_produces_different_key(self):
        numbers = ["17.74"]
        key1 = _cache_key(numbers, "source A")
        key2 = _cache_key(numbers, "source B")
        assert key1 != key2

    def test_number_order_does_not_matter(self):
        source = "price=17.74 change=0.06"
        key1 = _cache_key(["17.74", "0.06"], source)
        key2 = _cache_key(["0.06", "17.74"], source)
        assert key1 == key2


class TestCacheOperations:
    def test_cache_miss_returns_none(self):
        assert _cached_verdict("nonexistent") is None

    def test_cache_store_and_retrieve(self):
        _cache_verdict("key1", "SUPPORTED")
        assert _cached_verdict("key1") == "SUPPORTED"

    def test_cache_evicts_oldest_when_full(self):
        # Fill cache to max capacity
        for i in range(128):
            _cache_verdict(f"key{i}", "SUPPORTED")
        # Add one more — should evict key0
        _cache_verdict("key_new", "FABRICATED")
        assert _cached_verdict("key0") is None
        assert _cached_verdict("key_new") == "FABRICATED"

    def test_cache_lru_order(self):
        """LRU: accessing a key moves it to the end, so it survives eviction."""
        # Fill to capacity
        for i in range(128):
            _cache_verdict(f"key{i}", "SUPPORTED")
        # Access key0 — should move to end (most recently used)
        _cached_verdict("key0")
        # Add key_new — should evict key1 (now the least recently used), not key0
        _cache_verdict("key_new", "FABRICATED")
        assert _cached_verdict("key0") == "SUPPORTED"  # Was accessed, not evicted
        assert _cached_verdict("key1") is None  # LRU, evicted
        assert _cached_verdict("key_new") == "FABRICATED"


class TestSingleVerify:
    def test_returns_supported(self):
        llm = FakeLLM([{"verdict": "SUPPORTED", "reason": "all found"}])
        result = _single_verify("answer", ["17.74"], "source", llm, "question", 0.3)
        assert result == "SUPPORTED"

    def test_returns_fabricated(self):
        llm = FakeLLM([{"verdict": "FABRICATED", "reason": "not found"}])
        result = _single_verify("answer", ["999.99"], "source", llm, "question", 0.3)
        assert result == "FABRICATED"

    def test_returns_partial(self):
        llm = FakeLLM([{"verdict": "PARTIAL", "reason": "some found"}])
        result = _single_verify("answer", ["17.74", "999.99"], "source", llm, "question", 0.3)
        assert result == "PARTIAL"

    def test_returns_inconclusive_for_bad_response(self):
        llm = FakeLLM([{"unexpected": "format"}])
        result = _single_verify("answer", ["17.74"], "source", llm, "question", 0.3)
        assert result == "INCONCLUSIVE"

    def test_returns_inconclusive_for_invalid_verdict(self):
        llm = FakeLLM([{"verdict": "UNKNOWN"}])
        result = _single_verify("answer", ["17.74"], "source", llm, "question", 0.3)
        assert result == "INCONCLUSIVE"

    def test_returns_inconclusive_on_exception(self):
        llm = FakeLLM([RuntimeError("LLM error")])
        result = _single_verify("answer", ["17.74"], "source", llm, "question", 0.3)
        assert result == "INCONCLUSIVE"


class TestMultiSampleVerify:
    def test_two_samples_agree_supported_returns_original_answer(self):
        """When 2 samples agree on SUPPORTED, return original answer (no 3rd sample)."""
        llm = FakeLLM([
            {"verdict": "SUPPORTED", "reason": "found"},
            {"verdict": "SUPPORTED", "reason": "found"},
        ])
        answer = "海螺水泥当前价格 17.74 元"
        result = _multi_sample_verify(answer, ["17.74"], "price=17.74", llm, "question")
        assert result == answer
        assert llm._call_count == 2  # No 3rd sample needed

    def test_two_samples_agree_fabricated_returns_corrected(self):
        """When 2 samples agree on FABRICATED, call _full_verify_and_correct."""
        llm = FakeLLM([
            {"verdict": "FABRICATED", "reason": "not found"},
            {"verdict": "FABRICATED", "reason": "not found"},
            # 3rd call for _full_verify_and_correct
            {"verdict": "FABRICATED", "reason": "corrected", "corrected_answer": "数据无法获取"},
        ])
        result = _multi_sample_verify("价格 999.99", ["999.99"], "no data", llm, "question")
        assert result == "数据无法获取"
        assert llm._call_count == 3

    def test_two_samples_disagree_uses_tiebreaker(self):
        """When 2 samples disagree, 3rd sample breaks the tie."""
        llm = FakeLLM([
            {"verdict": "SUPPORTED", "reason": "found"},
            {"verdict": "FABRICATED", "reason": "not found"},
            {"verdict": "SUPPORTED", "reason": "found"},  # Tiebreaker
        ])
        answer = "价格 17.74"
        result = _multi_sample_verify(answer, ["17.74"], "price=17.74", llm, "question")
        assert result == answer  # SUPPORTED wins by majority
        assert llm._call_count == 3

    def test_all_inconclusive_returns_none(self):
        """When all 3 samples are INCONCLUSIVE, return None."""
        llm = FakeLLM([
            {"verdict": "BAD"},
            {"verdict": "BAD"},
            {"verdict": "BAD"},
        ])
        result = _multi_sample_verify("answer", ["17.74"], "source", llm, "question")
        assert result is None

    def test_cache_hit_skips_llm_calls(self):
        """When verdict is cached, no LLM calls should be made."""
        # Pre-populate cache
        numbers = ["17.74"]
        source = "price=17.74"
        key = _cache_key(numbers, source)
        _cache_verdict(key, "SUPPORTED")

        llm = FakeLLM([])  # No responses configured — should not be called
        result = _multi_sample_verify("answer", numbers, source, llm, "question")
        assert result == "answer"
        assert llm._call_count == 0

    def test_cache_hit_fabricated_still_calls_for_correction(self):
        """When FABRICATED is cached, still call LLM for corrected answer."""
        numbers = ["999.99"]
        source = "no matching data"
        key = _cache_key(numbers, source)
        _cache_verdict(key, "FABRICATED")

        llm = FakeLLM([
            # First two samples still run (cache only stores after first verification)
            # Actually no — cache hit means we skip to correction...
            # Wait: the code checks cache, if FABRICATED, it falls through
            # to sampling + correction. Let me re-read the code.
            # Actually: the code says "if cached == FABRICATED: pass (fall through)"
            # So it still does multi-sampling. The cache is for SUPPORTED/PARTIAL.
            {"verdict": "FABRICATED", "reason": "not found"},
            {"verdict": "FABRICATED", "reason": "not found"},
            {"corrected_answer": "corrected text"},
        ])
        result = _multi_sample_verify("answer", numbers, source, llm, "question")
        assert result == "corrected text"

    def test_fabricated_without_corrected_answer_returns_none(self):
        """When FABRICATED but no corrected_answer in full verify, return None."""
        llm = FakeLLM([
            {"verdict": "FABRICATED", "reason": "not found"},
            {"verdict": "FABRICATED", "reason": "not found"},
            {"verdict": "FABRICATED", "reason": "no correction"},  # No corrected_answer
        ])
        result = _multi_sample_verify("answer", ["999.99"], "no data", llm, "question")
        assert result is None

    def test_partial_verdict_returns_original_answer(self):
        """PARTIAL verdict means some numbers found — return original answer."""
        llm = FakeLLM([
            {"verdict": "PARTIAL", "reason": "some found"},
            {"verdict": "PARTIAL", "reason": "some found"},
        ])
        answer = "价格 17.74, 变化 999.99"
        result = _multi_sample_verify(answer, ["17.74", "999.99"], "price=17.74", llm, "question")
        assert result == answer

    def test_verdict_is_cached_after_verification(self):
        """After verification, the verdict should be in the cache."""
        llm = FakeLLM([
            {"verdict": "SUPPORTED", "reason": "found"},
            {"verdict": "SUPPORTED", "reason": "found"},
        ])
        numbers = ["17.74"]
        source = "price=17.74"
        _multi_sample_verify("answer", numbers, source, llm, "question")

        key = _cache_key(numbers, source)
        assert _cached_verdict(key) == "SUPPORTED"


class TestFullVerifyAndCorrect:
    def test_returns_corrected_answer(self):
        llm = MagicMock()
        llm.complete_json.return_value = {
            "verdict": "FABRICATED",
            "reason": "corrected",
            "corrected_answer": "修正后的回答",
        }
        result = _full_verify_and_correct("original", ["999.99"], "no data", llm, "question")
        assert result == "修正后的回答"

    def test_returns_none_when_no_corrected_answer(self):
        llm = MagicMock()
        llm.complete_json.return_value = {
            "verdict": "FABRICATED",
            "reason": "no correction available",
        }
        result = _full_verify_and_correct("original", ["999.99"], "no data", llm, "question")
        assert result is None

    def test_returns_none_on_exception(self):
        llm = MagicMock()
        llm.complete_json.side_effect = RuntimeError("LLM error")
        result = _full_verify_and_correct("original", ["999.99"], "no data", llm, "question")
        assert result is None

    def test_returns_none_for_non_dict_response(self):
        llm = MagicMock()
        llm.complete_json.return_value = "not a dict"
        result = _full_verify_and_correct("original", ["999.99"], "no data", llm, "question")
        assert result is None


class TestTemperatureDiversity:
    def test_first_two_samples_use_03_temperature(self):
        """Verify that the first two samples use temperature=0.3 for diversity."""
        llm = FakeLLM([
            {"verdict": "SUPPORTED", "reason": "found"},
            {"verdict": "SUPPORTED", "reason": "found"},
        ])
        _multi_sample_verify("answer", ["17.74"], "price=17.74", llm, "question")
        assert llm.call_history[0][1] == 0.3
        assert llm.call_history[1][1] == 0.3

    def test_tiebreaker_uses_00_temperature(self):
        """Verify that the tiebreaker uses temperature=0.0 for determinism."""
        llm = FakeLLM([
            {"verdict": "SUPPORTED", "reason": "found"},
            {"verdict": "FABRICATED", "reason": "not found"},
            {"verdict": "SUPPORTED", "reason": "found"},
        ])
        _multi_sample_verify("answer", ["17.74"], "price=17.74", llm, "question")
        assert llm.call_history[2][1] == 0.0
