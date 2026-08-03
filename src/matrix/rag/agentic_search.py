"""AgenticSearch: Agentic RAG pipeline with query rewriting, retrieval grading,
multi-step retrieval, and CRAG-style fallback.

Pipeline:
1. Query rewriting — transform user query for better retrieval
2. Initial retrieval — use HybridRetriever with rewritten query
3. Retrieval grading — LLM grades each document's relevance
4. Decision: if sufficient → return; if insufficient → multi-step retrieval or fallback
5. Multi-step retrieval — use sub-queries to find complementary info
6. CRAG fallback — if still insufficient, return assessment for web search fallback

Graceful degradation: every LLM step falls back to no-op when LLM is unavailable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from .retriever import HybridRetriever
from .query_rewriter import QueryRewriter
from .retrieval_grader import RetrievalGrader

if TYPE_CHECKING:
    from .knowledge_graph import GraphRetriever

logger = logging.getLogger(__name__)

# Max sub-query retrievals (to limit total retrieval rounds)
_MAX_SUB_QUERY_ROUNDS = 2

# Min documents to consider retrieval "sufficient" even without grading
_MIN_DOCS_FOR_SUFFICIENT = 2


class AgenticSearch:
    """Agentic RAG pipeline orchestrating query rewriting, retrieval, and grading.

    Usage::

        searcher = AgenticSearch(retriever=retriever, llm=pipeline_llm)
        result = searcher.search("我的持仓中科技股占比多少")
        # result = {
        #     "results": [...],           # relevant documents
        #     "query": "original query",
        #     "rewritten_query": "...",
        #     "sub_queries_used": [...],
        #     "assessment": "sufficient",
        #     "missing_info": "",
        #     "needs_web_fallback": False,
        # }
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: Any | None = None,
        web_search_fn: Callable[[str], dict] | None = None,
        graph_retriever: GraphRetriever | None = None,
    ) -> None:
        """Initialize the agentic search pipeline.

        Args:
            retriever: HybridRetriever instance for document retrieval.
            llm: LLM client for query rewriting and grading. If None,
                 degrades to simple retrieval (no rewriting/grading).
            web_search_fn: Optional callable for CRAG web search fallback.
                           Takes a query string, returns dict with "results" key.
            graph_retriever: Optional GraphRetriever for knowledge graph augmentation.
        """
        self._retriever = retriever
        self._rewriter = QueryRewriter(llm=llm)
        self._grader = RetrievalGrader(llm=llm)
        self._web_search_fn = web_search_fn
        self._graph_retriever = graph_retriever

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> dict[str, Any]:
        """Execute the agentic RAG pipeline.

        Args:
            query: User's natural language query.
            top_k: Target number of relevant documents to return.
            min_similarity: Minimum vector similarity threshold.

        Returns:
            Dict with keys:
                - "results": list[dict] — relevant documents
                - "query": str — original query
                - "rewritten_query": str — rewritten query
                - "sub_queries_used": list[str] — sub-queries that were executed
                - "assessment": str — "sufficient" | "insufficient" | "partial"
                - "missing_info": str — what's missing if insufficient
                - "needs_web_fallback": bool — whether web search should be triggered
                - "web_fallback_results": list[dict] — web search results if triggered
        """
        # Step 1: Query rewriting
        rewrite_result = self._rewriter.rewrite(query)
        rewritten_query = rewrite_result["rewritten"]
        sub_queries = rewrite_result["sub_queries"]

        logger.info(
            "agentic_search: query rewritten '%s' -> '%s', sub_queries=%d",
            query[:50], rewritten_query[:50], len(sub_queries),
        )

        # Step 2: Initial retrieval with rewritten query
        docs = self._retriever.query(
            rewritten_query, top_k=top_k, min_similarity=min_similarity
        )

        # Step 3: Retrieval grading
        grade_result = self._grader.grade(query, docs)
        relevant_docs = grade_result["relevant_docs"]
        assessment = grade_result["assessment"]
        missing_info = grade_result["missing_info"]

        sub_queries_used: list[str] = []

        # Step 4: Multi-step retrieval if insufficient
        if assessment == "insufficient" and sub_queries:
            for sq in sub_queries[:_MAX_SUB_QUERY_ROUNDS]:
                if len(relevant_docs) >= top_k:
                    break

                logger.info("agentic_search: sub-query retrieval '%s'", sq[:50])
                sub_docs = self._retriever.query(
                    sq, top_k=max(1, top_k - len(relevant_docs)),
                    min_similarity=min_similarity,
                )

                # Grade sub-query results
                sub_grade = self._grader.grade(query, sub_docs)
                relevant_docs.extend(sub_grade["relevant_docs"])
                sub_queries_used.append(sq)

                if sub_grade["assessment"] == "sufficient":
                    assessment = "sufficient"
                    missing_info = ""
                    break

        # Step 5: CRAG fallback — trigger web search if still insufficient
        web_fallback_results: list[dict] = []
        needs_web_fallback = False

        if (
            assessment == "insufficient"
            and not relevant_docs
            and self._web_search_fn is not None
        ):
            needs_web_fallback = True
            logger.info("agentic_search: CRAG fallback to web search for '%s'", query[:50])
            try:
                web_result = self._web_search_fn(rewritten_query)
                web_fallback_results = web_result.get("results", [])
                if web_fallback_results:
                    assessment = "sufficient"
                    missing_info = ""
            except Exception as exc:
                logger.warning("agentic_search: web fallback failed: %s", exc)

        # Deduplicate by content hash
        seen_content: set[str] = set()
        unique_docs: list[dict] = []
        for doc in relevant_docs:
            content = doc.get("content", "")
            content_key = content[:200] if content else ""
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_docs.append(doc)

        # Knowledge Graph augmentation: 用图谱上下文增强检索结果
        if self._graph_retriever is not None:
            try:
                unique_docs = self._graph_retriever.augment_results(query, unique_docs)
            except Exception as exc:
                logger.warning("agentic_search: graph augmentation failed: %s", exc)

        return {
            "results": unique_docs[:top_k],
            "query": query,
            "rewritten_query": rewritten_query,
            "sub_queries_used": sub_queries_used,
            "assessment": assessment,
            "missing_info": missing_info,
            "needs_web_fallback": needs_web_fallback,
            "web_fallback_results": web_fallback_results,
        }
