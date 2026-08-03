"""Tests for Knowledge Graph memory module."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from matrix.rag.knowledge_graph import (
    Entity,
    Relation,
    ExtractionResult,
    EntityExtractor,
    KnowledgeGraph,
    GraphRetriever,
    _normalize_entity_name,
)


# ---- Entity Name Normalization ----

class TestNormalizeEntityName:
    def test_basic_lowercase(self):
        assert _normalize_entity_name("Apple Inc") == "appleinc"

    def test_chinese_no_change(self):
        assert _normalize_entity_name("贵州茅台") == "贵州茅台"

    def test_remove_suffix(self):
        result = _normalize_entity_name("贵州茅台股份有限公司")
        assert "股份有限公司" not in result
        assert "茅台" in result

    def test_remove_prefix(self):
        result = _normalize_entity_name("中国平安")
        assert result == "平安"

    def test_empty_string(self):
        assert _normalize_entity_name("") == ""

    def test_id_property(self):
        e1 = Entity(name="贵州茅台股份有限公司")
        e2 = Entity(name="贵州茅台")
        assert e1.id == e2.id  # Should normalize to same ID


# ---- EntityExtractor ----

class FakeLLM:
    """Fake LLM for testing entity extraction."""

    def __init__(self, responses: list):
        self.responses = responses
        self.calls = []

    def complete(self, system, messages, **kwargs):
        self.calls.append(("complete", messages))
        return self.responses.pop(0) if self.responses else "{}"

    def complete_json(self, system, messages, schema=None, **kwargs):
        self.calls.append(("complete_json", messages))
        if not self.responses:
            return {"entities": [], "relations": []}
        resp = self.responses.pop(0)
        if isinstance(resp, str):
            return json.loads(resp)
        return resp


class TestEntityExtractor:
    def test_extract_with_llm(self):
        llm = FakeLLM([{
            "entities": [
                {"name": "贵州茅台", "type": "stock", "description": "白酒龙头"},
                {"name": "白酒", "type": "concept", "description": "消费板块"},
            ],
            "relations": [
                {"source": "贵州茅台", "target": "白酒", "relation": "belongs_to"},
            ],
        }])
        extractor = EntityExtractor(llm=llm)
        result = extractor.extract("贵州茅台是白酒行业的龙头公司", source_file="test.md")

        assert len(result.entities) == 2
        assert result.entities[0].name == "贵州茅台"
        assert result.entities[0].type == "stock"
        assert len(result.relations) == 1
        assert result.relations[0].source == "贵州茅台"
        assert result.relations[0].target == "白酒"
        assert result.relations[0].relation == "belongs_to"

    def test_extract_no_llm_fallback(self):
        """Without LLM, should fall back to rule-based extraction."""
        extractor = EntityExtractor(llm=None)
        result = extractor.extract("贵州茅台是白酒行业的龙头公司", source_file="test.md")

        assert len(result.entities) > 0
        # Rule-based extracts 2-4 char Chinese sequences
        names = [e.name for e in result.entities]
        # Should contain at least some Chinese word fragments
        assert any("茅台" in n for n in names)

    def test_extract_empty_text(self):
        extractor = EntityExtractor(llm=None)
        result = extractor.extract("", source_file="test.md")
        assert len(result.entities) == 0
        assert len(result.relations) == 0

    def test_extract_llm_error_fallback(self):
        """When LLM raises, should fall back to rules."""
        class ErrorLLM:
            def complete_json(self, *args, **kwargs):
                raise RuntimeError("LLM error")

        extractor = EntityExtractor(llm=ErrorLLM())
        result = extractor.extract("贵州茅台是白酒龙头", source_file="test.md")
        assert len(result.entities) > 0  # Should fall back to rules

    def test_extract_dedup_entities(self):
        """Duplicate entity names should be deduplicated."""
        llm = FakeLLM([{
            "entities": [
                {"name": "茅台", "type": "stock", "description": "白酒"},
                {"name": "茅台", "type": "stock", "description": "重复"},
            ],
            "relations": [],
        }])
        extractor = EntityExtractor(llm=llm)
        result = extractor.extract("茅台 茅台", source_file="test.md")
        assert len(result.entities) == 1  # Deduplicated


# ---- KnowledgeGraph ----

class TestKnowledgeGraph:
    @pytest.fixture
    def kg(self):
        return KnowledgeGraph()

    def test_add_extraction(self, kg):
        result = ExtractionResult(
            entities=[
                Entity(name="贵州茅台", type="stock", description="白酒龙头"),
                Entity(name="白酒", type="concept", description="消费板块"),
            ],
            relations=[
                Relation(source="贵州茅台", target="白酒", relation="belongs_to"),
            ],
        )
        changed = kg.add_extraction(result)
        assert changed == 2  # 2 new nodes
        assert kg.stats["nodes"] == 2
        assert kg.stats["edges"] == 1

    def test_add_duplicate_entity_merges(self, kg):
        """Adding same entity twice should merge, not duplicate."""
        result1 = ExtractionResult(
            entities=[Entity(name="茅台", type="stock", description="白酒")],
        )
        result2 = ExtractionResult(
            entities=[Entity(name="茅台", type="stock", description="白酒龙头")],
        )
        kg.add_extraction(result1)
        kg.add_extraction(result2)
        assert kg.stats["nodes"] == 1  # Still 1 node
        # Mentions should be 2
        node = kg._graph.nodes["茅台"]
        assert node["mentions"] == 2

    def test_search_exact_match(self, kg):
        """Search should find entities by exact name."""
        kg.add_extraction(ExtractionResult(
            entities=[
                Entity(name="茅台", type="stock", description="白酒龙头"),
                Entity(name="白酒", type="concept", description="消费板块"),
            ],
            relations=[Relation(source="茅台", target="白酒", relation="belongs_to")],
        ))

        result = kg.search("茅台怎么样")
        assert len(result.entities) >= 1
        names = [e["name"] for e in result.entities]
        assert "茅台" in names
        # Should also find connected entity via BFS
        assert "白酒" in names
        assert result.context_text  # Non-empty context

    def test_search_no_match(self, kg):
        """Search with no matching entities should return empty."""
        kg.add_extraction(ExtractionResult(
            entities=[Entity(name="茅台", type="stock")],
        ))
        result = kg.search("完全不相关的查询xyz")
        assert len(result.entities) == 0
        assert result.context_text == ""

    def test_search_empty_graph(self):
        kg = KnowledgeGraph()
        result = kg.search("任何查询")
        assert len(result.entities) == 0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kg.json")
            kg1 = KnowledgeGraph(persist_path=path)
            kg1.add_extraction(ExtractionResult(
                entities=[
                    Entity(name="茅台", type="stock", description="白酒"),
                    Entity(name="五粮液", type="stock", description="白酒"),
                ],
                relations=[Relation(source="茅台", target="五粮液", relation="competitor")],
            ))
            assert kg1.save()

            # Load into new instance
            kg2 = KnowledgeGraph(persist_path=path)
            assert kg2.stats["nodes"] == 2
            assert kg2.stats["edges"] == 1
            result = kg2.search("茅台")
            assert len(result.entities) >= 1

    def test_bfs_traversal(self, kg):
        """BFS should traverse connected entities within max_depth."""
        kg.add_extraction(ExtractionResult(
            entities=[
                Entity(name="AAA", type="other"),
                Entity(name="BBB", type="other"),
                Entity(name="CCC", type="other"),
                Entity(name="DDD", type="other"),  # Isolated
            ],
            relations=[
                Relation(source="AAA", target="BBB"),
                Relation(source="BBB", target="CCC"),
                # DDD has no connections
            ],
        ))

        # Search from AAA should find AAA, BBB, CCC (within 2 hops) but not DDD
        result = kg.search("AAA", max_depth=2)
        names = {e["name"] for e in result.entities}
        assert "AAA" in names
        assert "BBB" in names
        assert "CCC" in names
        assert "DDD" not in names

    def test_max_nodes_limit(self, kg):
        """Search should respect max_nodes limit."""
        entities = [Entity(name=f"E{i}", type="other") for i in range(20)]
        relations = [Relation(source=f"E{i}", target=f"E{i+1}") for i in range(19)]
        kg.add_extraction(ExtractionResult(entities=entities, relations=relations))

        result = kg.search("E0", max_nodes=5)
        assert len(result.entities) <= 5

    def test_stats(self, kg):
        kg.add_extraction(ExtractionResult(
            entities=[
                Entity(name="茅台", type="stock"),
                Entity(name="白酒", type="concept"),
            ],
            relations=[Relation(source="茅台", target="白酒")],
        ))
        stats = kg.stats
        assert stats["nodes"] == 2
        assert stats["edges"] == 1
        assert "stock" in stats["types"]
        assert stats["types"]["stock"] == 1

    def test_is_empty(self, kg):
        assert kg.is_empty is True
        kg.add_extraction(ExtractionResult(
            entities=[Entity(name="test", type="other")],
        ))
        assert kg.is_empty is False


# ---- GraphRetriever ----

class TestGraphRetriever:
    @pytest.fixture
    def graph_retriever(self):
        kg = KnowledgeGraph()
        kg.add_extraction(ExtractionResult(
            entities=[
                Entity(name="茅台", type="stock", description="白酒龙头股"),
                Entity(name="白酒", type="concept", description="消费板块"),
                Entity(name="消费", type="concept", description="大消费概念"),
            ],
            relations=[
                Relation(source="茅台", target="白酒", relation="belongs_to"),
                Relation(source="白酒", target="消费", relation="part_of"),
            ],
        ))
        return GraphRetriever(kg)

    def test_retrieve(self, graph_retriever):
        result = graph_retriever.retrieve("茅台怎么样")
        assert len(result.entities) >= 1
        assert result.context_text  # Non-empty

    def test_augment_results_with_graph(self, graph_retriever):
        """Augment should prepend graph context to vector docs."""
        vector_docs = [
            {"title": "doc1", "content": "some content", "score": 0.8},
        ]
        augmented = graph_retriever.augment_results("茅台", vector_docs)
        # Should have graph doc + original docs
        assert len(augmented) >= 2
        assert augmented[0]["source"] == "knowledge_graph"
        assert "茅台" in augmented[0]["content"]

    def test_augment_results_no_match(self, graph_retriever):
        """When graph has no match, should return original docs unchanged."""
        vector_docs = [{"title": "doc1", "content": "content", "score": 0.8}]
        augmented = graph_retriever.augment_results("完全无关的xyz查询", vector_docs)
        assert len(augmented) == 1  # No graph doc added
        assert augmented == vector_docs

    def test_augment_results_empty_graph(self):
        """Empty graph should return original docs."""
        kg = KnowledgeGraph()
        gr = GraphRetriever(kg)
        docs = [{"title": "doc1", "content": "content", "score": 0.8}]
        result = gr.augment_results("anything", docs)
        assert result == docs
