"""KnowledgeGraph: 实体抽取 + NetworkX 图存储 + 图谱检索.

模块设计:
    ┌──────────────────────────────────────────────────┐
    │  EntityExtractor                                  │
    │  输入: 文本 → 输出: 实体列表 + 关系列表 (JSON)     │
    ├──────────────────────────────────────────────────┤
    │  KnowledgeGraph (NetworkX)                        │
    │  节点: 实体 (name, type, description, source)     │
    │  边: 关系 (source → target, relation, weight)     │
    │  持久化: JSON (node-link format)                   │
    │  增量更新: 合并同实体, 累积 mentions               │
    ├──────────────────────────────────────────────────┤
    │  GraphRetriever                                   │
    │  输入: 查询 → 实体匹配 → BFS遍历(1-2跳) → 上下文   │
    └──────────────────────────────────────────────────┘

与 RAG 管道的集成:
    - Indexer: 文档入库时抽取实体, 写入图谱
    - AgenticSearch: 向量检索后补充图谱上下文
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.protocol import LLMClient

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────

_MAX_ENTITIES_PER_DOC = 30
_MAX_RELATIONS_PER_DOC = 40
_BFS_MAX_DEPTH = 2
_BFS_MAX_NODES = 15
_ENTITY_NAME_MAX_LEN = 50

# ── 实体抽取 Prompt ──────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """你是一个实体关系抽取引擎。从给定文本中抽取实体和实体间关系。

规则:
1. 只抽取明确出现在文本中的实体, 不要推测
2. 实体类型包括: stock(股票), fund(基金), concept(概念/板块), person(人物), org(机构/公司), indicator(指标/数据), asset(资产), event(事件), other
3. 关系类型包括: belongs_to(属于), related_to(相关), affects(影响), part_of(组成部分), competitor(竞争), holds(持有)
4. 实体名称用规范名称 (如公司全称或常用简称)
5. 每个实体附带简短描述 (来自文本上下文, 不超过100字)

返回 JSON:
{
  "entities": [
    {"name": "贵州茅台", "type": "stock", "description": "白酒龙头"},
    {"name": "白酒", "type": "concept", "description": "消费板块"}
  ],
  "relations": [
    {"source": "贵州茅台", "target": "白酒", "relation": "belongs_to"}
  ]
}

如果没有实体, 返回 {"entities": [], "relations": []}
只返回 JSON, 不要其他文字。"""

# ── 实体名称归一化 ────────────────────────────────────────────────────────

# 中文实体常见后缀, 归一化时移除
_CN_SUFFIXES = [
    "股份有限公司", "有限责任公司", "有限公司", "股份公司",
    "集团", "控股", "投资", "基金", "ETF",
    "股份", "科技", "信息", "通信",
]

# 中文实体常见前缀
_CN_PREFIXES = ["中国", "中华"]


def _normalize_entity_name(name: str) -> str:
    """归一化实体名称用于去重.

    策略: 去空格 → 转小写 → 移除常见后缀/前缀 → 去标点
    """
    n = name.strip().lower()
    # 移除常见后缀 (从长到短)
    for suffix in sorted(_CN_SUFFIXES, key=len, reverse=True):
        if n.endswith(suffix.lower()) and len(n) > len(suffix) + 1:
            n = n[: -len(suffix)]
            break
    # 移除常见前缀
    for prefix in sorted(_CN_PREFIXES, key=len, reverse=True):
        if n.startswith(prefix.lower()) and len(n) > len(prefix) + 1:
            n = n[len(prefix):]
            break
    # 去标点
    n = re.sub(r"[^\w\u4e00-\u9fff]", "", n)
    return n


# ── 数据结构 ─────────────────────────────────────────────────────────────


@dataclass
class Entity:
    """知识图谱中的实体节点."""
    name: str
    type: str = "other"
    description: str = ""
    source_file: str = ""
    mentions: int = 1

    @property
    def id(self) -> str:
        return _normalize_entity_name(self.name)


@dataclass
class Relation:
    """知识图谱中的关系边."""
    source: str  # entity name
    target: str  # entity name
    relation: str = "related_to"
    weight: float = 1.0
    source_file: str = ""


@dataclass
class ExtractionResult:
    """实体抽取结果."""
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


@dataclass
class GraphSearchResult:
    """图谱检索结果."""
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    context_text: str = ""


# ── EntityExtractor ──────────────────────────────────────────────────────


class EntityExtractor:
    """使用 LLM 从文本中抽取实体和关系.

    当 LLM 不可用时, 退化为简单的关键词提取 (基于规则).
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def extract(self, text: str, source_file: str = "") -> ExtractionResult:
        """从文本中抽取实体和关系.

        Args:
            text: 待抽取的文本 (通常是文档的一个 chunk).
            source_file: 来源文件路径, 用于溯源.

        Returns:
            ExtractionResult 包含 entities 和 relations.
        """
        if not text or not text.strip():
            return ExtractionResult()

        if self._llm is None:
            return self._extract_rule_based(text, source_file)

        try:
            result = self._llm.complete_json(
                _EXTRACT_SYSTEM,
                [{"role": "user", "content": text[:2000]}],  # 截断超长文本
                temperature=0.0,
            )

            raw_entities = result.get("entities", [])
            raw_relations = result.get("relations", [])

            entities: list[Entity] = []
            seen_ids: set[str] = set()
            for e in raw_entities[:_MAX_ENTITIES_PER_DOC]:
                name = str(e.get("name", "")).strip()
                if not name or len(name) > _ENTITY_NAME_MAX_LEN:
                    continue
                eid = _normalize_entity_name(name)
                if not eid or eid in seen_ids:
                    continue
                seen_ids.add(eid)
                entities.append(Entity(
                    name=name,
                    type=str(e.get("type", "other")).strip().lower() or "other",
                    description=str(e.get("description", "")).strip()[:200],
                    source_file=source_file,
                ))

            relations: list[Relation] = []
            for r in raw_relations[:_MAX_RELATIONS_PER_DOC]:
                src = str(r.get("source", "")).strip()
                tgt = str(r.get("target", "")).strip()
                rel = str(r.get("relation", "related_to")).strip().lower()
                if not src or not tgt:
                    continue
                # 确保端点在实体列表中
                src_id = _normalize_entity_name(src)
                tgt_id = _normalize_entity_name(tgt)
                if src_id not in seen_ids or tgt_id not in seen_ids:
                    continue
                relations.append(Relation(
                    source=src,
                    target=tgt,
                    relation=rel or "related_to",
                    source_file=source_file,
                ))

            if entities:
                logger.debug(
                    "entity_extract: %s entities, %d relations from %s",
                    len(entities), len(relations), source_file or "text",
                )

            return ExtractionResult(entities=entities, relations=relations)

        except Exception as exc:
            logger.warning("entity_extract: LLM failed, falling back to rules: %s", exc)
            return self._extract_rule_based(text, source_file)

    def _extract_rule_based(self, text: str, source_file: str = "") -> ExtractionResult:
        """规则兜底: 提取中文大写词组和数字+单位模式."""
        entities: list[Entity] = []
        seen: set[str] = set()

        # 匹配中文连续词组 (2-4个汉字, 非贪婪分块)
        cn_pattern = re.compile(r"[\u4e00-\u9fff]{2,4}")
        for m in cn_pattern.finditer(text):
            name = m.group()
            eid = _normalize_entity_name(name)
            if eid in seen:
                continue
            seen.add(eid)
            entities.append(Entity(
                name=name,
                type="other",
                description="",
                source_file=source_file,
            ))
            if len(entities) >= _MAX_ENTITIES_PER_DOC:
                break

        return ExtractionResult(entities=entities)


# ── KnowledgeGraph ───────────────────────────────────────────────────────


class KnowledgeGraph:
    """基于 NetworkX 的知识图谱存储.

    功能:
        - add_extraction: 批量添加实体和关系
        - search: 按实体名称搜索, 返回邻居子图
        - save / load: JSON 持久化 (node-link format)
        - stats: 图谱统计信息

    线程安全: 内部使用 threading.Lock 保护所有写操作.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        try:
            import networkx as nx
            self._nx = nx
        except ImportError:
            raise ImportError(
                "networkx is required for KnowledgeGraph. "
                "Install with: pip install 'matrix[rag]'"
            )

        self._graph = nx.Graph()
        self._persist_path = persist_path
        self._lock = threading.Lock()
        self._dirty = False

        # 启动时加载已有图谱
        if persist_path and os.path.exists(persist_path):
            self.load(persist_path)
            logger.info(
                "KnowledgeGraph loaded: %d nodes, %d edges from %s",
                self._graph.number_of_nodes(),
                self._graph.number_of_edges(),
                persist_path,
            )

    # ── 写操作 ──────────────────────────────────────────────────────

    def add_extraction(self, result: ExtractionResult) -> int:
        """将抽取结果添加到图谱中.

        对于已存在的实体: 累积 mentions, 合并描述.
        对于已存在的边: 累积 weight.

        Returns: 新增/更新的节点数.
        """
        if not result.entities and not result.relations:
            return 0

        changed = 0
        modified = False
        with self._lock:
            # 添加实体节点
            for entity in result.entities:
                eid = entity.id
                if eid in self._graph:
                    # 合并: 累积 mentions, 更新描述
                    node = self._graph.nodes[eid]
                    node["mentions"] = node.get("mentions", 0) + 1
                    if entity.description and not node.get("description"):
                        node["description"] = entity.description
                    if entity.type != "other" and node.get("type", "other") == "other":
                        node["type"] = entity.type
                    # 记录来源文件
                    sources = node.get("source_files", "")
                    if entity.source_file and entity.source_file not in sources:
                        node["source_files"] = (sources + ", " + entity.source_file).strip(", ")
                    modified = True
                else:
                    self._graph.add_node(eid, **{
                        "name": entity.name,
                        "type": entity.type,
                        "description": entity.description,
                        "source_files": entity.source_file,
                        "mentions": entity.mentions,
                    })
                    changed += 1
                    modified = True

            # 添加关系边
            for rel in result.relations:
                src_id = _normalize_entity_name(rel.source)
                tgt_id = _normalize_entity_name(rel.target)
                if src_id not in self._graph or tgt_id not in self._graph:
                    continue
                if self._graph.has_edge(src_id, tgt_id):
                    edge = self._graph.edges[src_id, tgt_id]
                    edge["weight"] = edge.get("weight", 1.0) + rel.weight
                    # 合并关系类型
                    existing_rel = edge.get("relation", "")
                    if rel.relation not in existing_rel:
                        edge["relation"] = f"{existing_rel}, {rel.relation}".strip(", ")
                    modified = True
                else:
                    self._graph.add_edge(src_id, tgt_id, **{
                        "relation": rel.relation,
                        "weight": rel.weight,
                        "source_file": rel.source_file,
                    })
                    modified = True

        if modified:
            self._dirty = True

        return changed

    # ── 读操作 ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        max_depth: int = _BFS_MAX_DEPTH,
        max_nodes: int = _BFS_MAX_NODES,
    ) -> GraphSearchResult:
        """根据查询在图谱中检索相关实体和关系.

        策略:
            1. 从查询中提取候选实体名 (分词 + 模糊匹配)
            2. 在图中找到匹配的节点
            3. BFS 遍历邻居 (max_depth 跳)
            4. 返回子图作为上下文

        Args:
            query: 用户查询.
            max_depth: BFS 遍历深度.
            max_nodes: 返回的最大节点数.

        Returns:
            GraphSearchResult 包含实体、关系和拼接的上下文文本.
        """
        if self._graph.number_of_nodes() == 0:
            return GraphSearchResult()

        # Step 1: 从查询中提取候选实体
        candidates = self._extract_query_entities(query)
        if not candidates:
            return GraphSearchResult()

        # Step 2: 在图中匹配节点
        matched_nodes: set[str] = set()
        for candidate in candidates:
            cid = _normalize_entity_name(candidate)
            # 精确匹配
            if cid in self._graph:
                matched_nodes.add(cid)
                continue
            # 子串匹配 (候选词是图中节点名的一部分, 或反之)
            for node_id in self._graph.nodes:
                if cid and (cid in node_id or node_id in cid):
                    matched_nodes.add(node_id)

        if not matched_nodes:
            return GraphSearchResult()

        # Step 3: BFS 遍历
        result_nodes: set[str] = set()
        with self._lock:
            for start in matched_nodes:
                if len(result_nodes) >= max_nodes:
                    break
                bfs_nodes = self._bfs(start, max_depth, max_nodes - len(result_nodes))
                result_nodes.update(bfs_nodes)

        # Step 4: 构建结果
        entities = []
        relations = []
        context_parts: list[str] = []

        with self._lock:
            for node_id in result_nodes:
                data = self._graph.nodes[node_id]
                entity = {
                    "name": data.get("name", node_id),
                    "type": data.get("type", "other"),
                    "description": data.get("description", ""),
                    "mentions": data.get("mentions", 1),
                }
                entities.append(entity)
                if entity["description"]:
                    context_parts.append(f"{entity['name']}: {entity['description']}")

            for u, v in self._graph.edges(result_nodes):
                if u in result_nodes and v in result_nodes:
                    edge_data = self._graph.edges[u, v]
                    u_name = self._graph.nodes[u].get("name", u)
                    v_name = self._graph.nodes[v].get("name", v)
                    relation = edge_data.get("relation", "related_to")
                    relations.append({
                        "source": u_name,
                        "target": v_name,
                        "relation": relation,
                        "weight": edge_data.get("weight", 1.0),
                    })
                    context_parts.append(f"{u_name} --[{relation}]--> {v_name}")

        context_text = "\n".join(context_parts) if context_parts else ""

        return GraphSearchResult(
            entities=entities,
            relations=relations,
            context_text=context_text,
        )

    def _bfs(self, start: str, max_depth: int, max_nodes: int) -> set[str]:
        """从起始节点 BFS 遍历, 返回子图节点集合."""
        result = {start}
        if max_nodes <= 1:
            return result

        current_level = {start}
        for depth in range(max_depth):
            if not current_level or len(result) >= max_nodes:
                break
            next_level: set[str] = set()
            for node in current_level:
                for neighbor in self._graph.neighbors(node):
                    if neighbor not in result and len(result) < max_nodes:
                        next_level.add(neighbor)
                        result.add(neighbor)
            current_level = next_level

        return result

    def _extract_query_entities(self, query: str) -> list[str]:
        """从查询中提取候选实体名."""
        candidates: list[str] = []

        # 中文连续词组 (2-8个汉字)
        for m in re.finditer(r"[\u4e00-\u9fff]{2,8}", query):
            candidates.append(m.group())

        # 英文词组 (2+ chars)
        for m in re.finditer(r"[A-Za-z]{2,20}", query):
            candidates.append(m.group())

        # 数字+单位模式 (如 "3000点", "50ETF")
        for m in re.finditer(r"\d+[A-Za-z\u4e00-\u9fff]{1,4}", query):
            candidates.append(m.group())

        return candidates

    # ── 持久化 ──────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> bool:
        """将图谱保存为 JSON (node-link format)."""
        save_path = path or self._persist_path
        if not save_path:
            logger.warning("KnowledgeGraph: no persist path, skipping save")
            return False

        try:
            with self._lock:
                data = self._nx.node_link_data(self._graph, edges="links")

            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self._dirty = False

            logger.info(
                "KnowledgeGraph saved: %d nodes, %d edges to %s",
                self._graph.number_of_nodes(),
                self._graph.number_of_edges(),
                save_path,
            )
            return True
        except Exception as exc:
            logger.error("KnowledgeGraph save failed: %s", exc)
            return False

    def load(self, path: str) -> bool:
        """从 JSON 加载图谱."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self._graph = self._nx.node_link_graph(data, edges="links")
                self._dirty = False
            return True
        except Exception as exc:
            logger.warning("KnowledgeGraph load failed: %s", exc)
            return False

    # ── 统计 ────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """返回图谱统计信息."""
        with self._lock:
            type_counts: dict[str, int] = {}
            for node_id in self._graph.nodes:
                t = self._graph.nodes[node_id].get("type", "other")
                type_counts[t] = type_counts.get(t, 0) + 1
            return {
                "nodes": self._graph.number_of_nodes(),
                "edges": self._graph.number_of_edges(),
                "types": type_counts,
            }

    @property
    def is_empty(self) -> bool:
        return self._graph.number_of_nodes() == 0

    @property
    def dirty(self) -> bool:
        """图谱自上次成功持久化后是否发生变化。"""
        return self._dirty


# ── GraphRetriever: 图谱检索增强 ─────────────────────────────────────────


class GraphRetriever:
    """将知识图谱检索结果融合到 RAG 管道中.

    使用方式:
        graph_result = graph_retriever.retrieve(query)
        if graph_result.context_text:
            # 将图谱上下文拼接到向量检索结果前
            context = graph_result.context_text + "\n\n" + vector_context
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    def retrieve(self, query: str) -> GraphSearchResult:
        """检索图谱中与查询相关的上下文."""
        return self._graph.search(query)

    def augment_results(
        self,
        query: str,
        vector_docs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """用图谱上下文增强向量检索结果.

        如果图谱中有与查询相关的实体, 在向量检索结果前插入一个图谱上下文文档.
        """
        graph_result = self._graph.search(query)
        if not graph_result.entities:
            return vector_docs

        # 构建图谱上下文文档
        if graph_result.context_text:
            graph_doc = {
                "title": "知识图谱上下文",
                "content": graph_result.context_text,
                "score": 1.0,  # 图谱匹配高权重
                "source": "knowledge_graph",
                "entities": [e["name"] for e in graph_result.entities],
                "relations": [
                    f"{r['source']} --{r['relation']}--> {r['target']}"
                    for r in graph_result.relations
                ],
            }
            return [graph_doc] + vector_docs

        return vector_docs
