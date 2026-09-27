"""Memory retrieval quality regression suite.

Not a benchmark entry — a guard rail. It seeds a synthetic memory store, runs
queries with known relevant answers, and asserts recall stays above a floor.
Any change that degrades retrieval (fusion weights, tokenizer, scoring) should
fail here before it reaches a user.

Recall is measured as: did the expected memory appear in the top-k results?
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from matrix.memory.retriever import MemoryRetriever
from matrix.store import SessionStore

# (key, value, memory_type) — deliberately includes near-miss distractors so
# lexical-only matching and semantic-only matching fail differently.
CORPUS: list[tuple[str, str, str]] = [
    ("研究对象", "当前研究对象为贵州茅台，代码 600519", "preference"),
    ("历史研究对象", "曾研究五粮液，代码 000858", "preference"),
    ("沟通语言", "用户沟通语言为中文", "preference"),
    ("回答风格", "用户偏好简洁、结论先行的回答", "preference"),
    ("数据来源", "投资研究优先使用 personal-os 市场数据工具", "preference"),
    ("数据标注", "数据不足时必须明确标注，不得推测", "preference"),
    ("持仓_现金", "持有 Sample Cash，5月2日市值 150.00 元", "preference"),
    ("持仓_基金", "持有 Sample Fund，5月1日市值 200.00 元", "preference"),
    ("研究日期", "该次研究日期为 2026-08-16", "preference"),
    ("风险偏好", "投资风格偏稳健，不接受高波动标的", "preference"),
    ("行业关注", "重点关注白酒行业与消费板块", "preference"),
    ("报告格式", "研究报告使用 Markdown，含数据来源标注", "preference"),
    ("单只仓位上限", "单只股票占投资组合比例不超过 30%", "policy"),
    ("禁止标的", "禁止投资加密货币及相关衍生品", "policy"),
]

# (query, expected_key) — each query has exactly one intended answer.
QUERIES: list[tuple[str, str]] = [
    ("我现在在研究哪只股票", "研究对象"),
    ("我之前研究过五粮液吗", "历史研究对象"),
    ("请用英文回答", "沟通语言"),
    ("回答能不能简短一点", "回答风格"),
    ("研究数据从哪里取", "数据来源"),
    ("我的现金持仓是多少", "持仓_现金"),
    ("基金持仓市值", "持仓_基金"),
    ("单只股票最多能买多少", "单只仓位上限"),
    ("能不能买比特币", "禁止标的"),
    ("我关注哪些行业", "行业关注"),
]

RECALL_FLOOR = 0.7  # 7 of 10 queries must retrieve the intended memory at k=3


@pytest.fixture
def seeded():
    with tempfile.TemporaryDirectory() as d:
        store = SessionStore(str(Path(d) / "eval.db"))
        for key, value, memory_type in CORPUS:
            store.upsert_profile("eval-user", key, value, memory_type=memory_type)
        retriever = MemoryRetriever(store, embedder=None)  # lexical-only, no model
        yield store, retriever


def _recall_at_k(retriever, k: int = 3) -> tuple[float, list[str]]:
    hits = 0
    misses: list[str] = []
    for query, expected in QUERIES:
        results = retriever.search("eval-user", query, top_k=k)
        keys = [r.key for r in results]
        if expected in keys:
            hits += 1
        else:
            misses.append(f"{query!r} -> got {keys}, wanted {expected!r}")
    return hits / len(QUERIES), misses


class TestRetrievalQuality:
    def test_recall_at_3_meets_floor(self, seeded):
        _store, retriever = seeded
        recall, misses = _recall_at_k(retriever, k=3)
        assert recall >= RECALL_FLOOR, (
            f"recall@3 = {recall:.2f} below floor {RECALL_FLOOR}\n"
            + "\n".join(misses)
        )

    def test_recall_at_1_is_reasonable(self, seeded):
        _store, retriever = seeded
        recall, _ = _recall_at_k(retriever, k=1)
        assert recall >= 0.4, f"recall@1 = {recall:.2f} is too low"

    def test_policies_retrievable_by_type_filter(self, seeded):
        _store, retriever = seeded
        hits = retriever.search("eval-user", "投资限制", top_k=5,
                                memory_type="policy")
        assert {h.key for h in hits} <= {"单只仓位上限", "禁止标的"}

    def test_recall_improves_with_k(self, seeded):
        _store, retriever = seeded
        at_1, _ = _recall_at_k(retriever, k=1)
        at_5, _ = _recall_at_k(retriever, k=5)
        assert at_5 >= at_1


class TestPromptEfficiency:
    """The reason Phase 1 exists: stop paying for memories that do not matter."""

    def test_injection_is_smaller_than_full_dump(self, seeded):
        from matrix.memory.prompting import build_memory_block

        store, retriever = seeded
        full = build_memory_block(store, "eval-user", query="茅台",
                                  full_dump=True)
        selective = build_memory_block(store, "eval-user", query="茅台",
                                       retriever=retriever, top_k=4)
        assert selective.tokens < full.tokens
        # Policies must survive the cut.
        assert selective.policy_count == 2

    def test_budget_holds_at_scale(self):
        from matrix.memory.prompting import build_memory_block

        with tempfile.TemporaryDirectory() as d:
            store = SessionStore(str(Path(d) / "scale.db"))
            for i in range(80):
                store.upsert_profile("u", f"偏好{i}", f"第{i}条关于投资风格的偏好描述")
            retriever = MemoryRetriever(store, embedder=None)
            block = build_memory_block(store, "u", query="投资风格",
                                       retriever=retriever, budget_tokens=800)
        assert block.tokens <= 800
