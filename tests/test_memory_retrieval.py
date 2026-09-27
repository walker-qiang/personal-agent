"""Tests for Phase 1: hybrid retrieval, budgeted injection, temporal scoping."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from matrix.memory.prompting import build_memory_block
from matrix.memory.retriever import MemoryRetriever, tokenize
from matrix.memory.temporal import parse_fact_time, resolve_time_range
from matrix.store import SessionStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as d:
        s = SessionStore(str(Path(d) / "test.db"))
        yield s


@pytest.fixture
def lexical(store):
    """Retriever without an embedder: lexical-only, no model dependency."""
    return MemoryRetriever(store, embedder=None)


class TestTokenize:
    def test_chinese_bigrams(self):
        tokens = tokenize("贵州茅台")
        assert "贵州" in tokens and "州茅" in tokens and "茅台" in tokens

    def test_single_char(self):
        assert "酒" in tokenize("酒")

    def test_ascii_runs(self):
        assert "maotai" in tokenize("MAOTAI 600519")

    def test_empty(self):
        assert tokenize("") == []


class TestLexicalRetrieval:
    def test_exact_token_match_ranks_first(self, store, lexical):
        store.upsert_profile("u", "研究对象", "贵州茅台 600519")
        store.upsert_profile("u", "语言偏好", "使用中文回答")
        store.upsert_profile("u", "投资风格", "偏好稳健配置")

        hits = lexical.search("u", "茅台")
        assert hits, "lexical search must find an exact token match"
        assert hits[0].key == "研究对象"

    def test_no_match_returns_empty(self, store, lexical):
        store.upsert_profile("u", "研究对象", "贵州茅台 600519")
        assert lexical.search("u", "完全不相关的查询") == []

    def test_type_filter(self, store, lexical):
        store.upsert_profile("u", "硬约束", "不买亏损股", memory_type="policy")
        store.upsert_profile("u", "软偏好", "喜欢简洁回答", memory_type="preference")
        assert [h.key for h in lexical.search("u", "亏损股", memory_type="policy")] == ["硬约束"]
        assert lexical.search("u", "亏损股", memory_type="preference") == []

    def test_top_k_respected(self, store, lexical):
        for i in range(10):
            store.upsert_profile("u", f"偏好{i}", f"关于咖啡的第{i}条偏好")
        assert len(lexical.search("u", "咖啡", top_k=3)) <= 3

    def test_retired_memories_excluded(self, store, lexical):
        store.upsert_profile("u", "旧偏好", "曾经喜欢咖啡")
        store.delete_profile_key("u", "旧偏好")
        assert lexical.search("u", "咖啡") == []


class TestFactTime:
    def test_iso_date(self):
        assert parse_fact_time("研究日期为 2026-08-16") > 0

    def test_cn_date(self):
        assert parse_fact_time("2026年8月16日 进行研究") > 0

    def test_no_date(self):
        assert parse_fact_time("用户喜欢简洁回答") == 0.0

    def test_relative_range(self):
        start, end = resolve_time_range("三个月前我在研究什么")
        assert start > 0 and end > start

    def test_non_temporal_query(self):
        assert resolve_time_range("我的持仓情况") == (0.0, 0.0)

    def test_dated_fact_survives_time_filter(self, store, lexical):
        import time
        stamped = parse_fact_time("研究于 2026-01-10")
        store.upsert_profile("u", "研究记录", "研究于 2026-01-10 的茅台", fact_time=stamped)
        start, end = resolve_time_range("2026年1月我在研究什么")
        hits = lexical.search("u", "研究", time_from=start, time_to=end)
        assert [h.key for h in hits] == ["研究记录"]
        assert (time.time() - stamped) > 0  # fact_time is in the past


class TestPromptBudget:
    def test_policies_always_injected(self, store):
        store.upsert_profile("u", "硬约束", "不买亏损股", memory_type="policy")
        block = build_memory_block(store, "u", query="完全无关的提问")
        assert "不买亏损股" in block.text
        assert block.policy_count == 1

    def test_selective_injection_beats_full_dump(self, store, lexical):
        for i in range(60):
            store.upsert_profile("u", f"偏好{i}", f"第{i}条与咖啡无关的偏好描述内容")
        store.upsert_profile("u", "研究对象", "贵州茅台 600519")

        full = build_memory_block(store, "u", query="茅台", full_dump=True)
        selective = build_memory_block(store, "u", query="茅台", retriever=lexical)
        assert selective.tokens < full.tokens, (
            f"selective {selective.tokens} should beat dump {full.tokens}"
        )

    def test_budget_is_enforced(self, store, lexical):
        for i in range(40):
            store.upsert_profile("u", f"偏好{i}", f"关于投资风格偏好描述内容第{i}条")
        block = build_memory_block(store, "u", query="投资", retriever=lexical,
                                   budget_tokens=200)
        assert block.tokens <= 200
        assert block.truncated > 0

    def test_empty_store_yields_empty_block(self, store):
        assert build_memory_block(store, "u", query="anything").text == ""

    def test_no_query_falls_back_to_dump(self, store, lexical):
        store.upsert_profile("u", "软偏好", "喜欢简洁回答")
        block = build_memory_block(store, "u", query="", retriever=lexical)
        assert "喜欢简洁回答" in block.text


class TestAccessTracking:
    def test_injection_bumps_access_count(self, store, lexical):
        store.upsert_profile("u", "研究对象", "贵州茅台 600519")
        build_memory_block(store, "u", query="茅台", retriever=lexical)
        memories = store.get_all_memories("u")
        assert memories[0]["access_count"] == 1
