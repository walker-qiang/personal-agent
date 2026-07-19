"""news_search — news-only search using 360 News.

For general web search, use web_search instead.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from ..base import ToolDefinition

tool_definition = ToolDefinition(
    name="news_search",
    description='搜索新闻，返回最新新闻的标题、摘要和链接。当用户问「最近/最新/今天」发生的事时必须用此工具。',
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，英文或中文均可",
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数，默认 5，最大 10",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    handler=None,
)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_SO360_NEWS_URL = "https://news.so.com/ns"

_STRIP_HTML = re.compile(r"<[^>]+>")
_STRIP_ENTITY = re.compile(r"&[a-z]+;")
_COLLAPSE_SPACE = re.compile(r"\s+")


def _clean_html(text: str) -> str:
    text = _STRIP_HTML.sub(" ", text)
    text = _STRIP_ENTITY.sub(" ", text)
    text = _COLLAPSE_SPACE.sub(" ", text)
    return text.strip()


def _fetch(url: str, timeout_sec: float) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _parse_so360_news(html: str, limit: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    all_h3 = re.findall(r"<h3[^>]*>(.*?)</h3>", html, re.DOTALL)

    for h3_content in all_h3:
        if len(results) >= limit:
            break
        title = _clean_html(h3_content)
        if not title:
            continue
        link_match = re.search(r'href="([^"]+)"', h3_content)
        url = link_match.group(1) if link_match else ""
        results.append({"title": title, "url": url, "snippet": ""})

    all_snippets = re.findall(
        r'<p[^>]*class="[^"]*(?:res-desc|news-desc|summary|abstract)[^"]*"[^>]*>(.*?)</p>',
        html, re.DOTALL,
    )
    if not all_snippets:
        all_snippets = re.findall(r"<p[^>]*>(.*?)</p>", html, re.DOTALL)
        all_snippets = [s for s in all_snippets if 30 < len(_clean_html(s)) < 500]

    for i, snippet_html in enumerate(all_snippets):
        if i < len(results):
            results[i]["snippet"] = _clean_html(snippet_html)

    return results


# ---- URL filter: strip search-engine redirect links ----

# 360 redirect URLs (so.com/link?m=...) are not real web pages
_REDIRECT_URL_RE = re.compile(r"^https?://(?:www\.)?so\.com/link\?", re.IGNORECASE)


def _filter_redirect_urls(results: list[dict[str, str]]) -> list[dict[str, str]]:
    """Replace redirect/synthetic URLs with empty string so LLM won't try to web_fetch them."""
    for r in results:
        url = r.get("url", "")
        if url and _REDIRECT_URL_RE.search(url):
            r["url"] = ""
    return results


# Time-sensitive keywords that indicate the query needs the current year
_TIME_SENSITIVE_RE = re.compile(
    r"最近|最新|今天|今日|近期|当前|现在|近日|今年|本(?:周|月|年)",
)

# Multi-year patterns that dilute search results (e.g. "2025 2026" → "2026")
_MULTI_YEAR_RE = re.compile(r"\b(?:20\d{2}[-\s]*)+20\d{2}\b|\b20\d{2}\s+20\d{2}\b")

# Query expansion: when a query contains a key term but not its synonym,
# append the synonym to improve recall.  360 search often returns poor
# results for "洲际导弹" alone (biased toward old land-based articles)
# while the latest events use "潜射导弹" in headlines.
_EXPAND_KEYWORDS: dict[str, str] = {
    "洲际导弹": "潜射",
    "洲际弹道": "潜射",
    "战略导弹": "潜射",
}


def _expand_query(query: str) -> str:
    """Append missing synonyms to improve search recall."""
    for key, synonym in _EXPAND_KEYWORDS.items():
        if key in query and synonym not in query:
            return f"{query} {synonym}"
    return query


def _inject_current_year(query: str) -> str:
    """If the query is time-sensitive, always strip time-sensitive words (they bias
    search engines toward old articles), and append the current year if missing.
    Multi-year queries like "2025 2026" are also cleaned to just the current year."""
    if not _TIME_SENSITIVE_RE.search(query) and not _MULTI_YEAR_RE.search(query):
        return query
    current_year = str(datetime.now(timezone.utc).year)
    # Always strip time-sensitive keywords
    cleaned = _TIME_SENSITIVE_RE.sub("", query)
    # Normalize multi-year patterns: "2025 2026" → "2026", "2024-2026" → "2026"
    cleaned = _MULTI_YEAR_RE.sub(current_year, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if current_year not in cleaned:
        cleaned = f"{cleaned} {current_year}"
    return cleaned


# Year extraction from snippet text (same as search.py)
_YEAR_IN_SNIPPET_RE = re.compile(r"\b(20\d{2})\b")


def _result_year_score(snippet: str, current_year: int) -> int:
    """Score a result by recency: current year = 100, each year older = -100."""
    years = _YEAR_IN_SNIPPET_RE.findall(snippet)
    if not years:
        return 0
    best = max(int(y) for y in years)
    return 100 - (current_year - best) * 100


def _boost_recent_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    """Reorder results so current-year results appear first."""
    current_year = datetime.now(timezone.utc).year
    scored = [(r, _result_year_score(r.get("snippet", ""), current_year)) for r in results]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored]


def news_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search news using 360 News."""
    max_results = min(max(max_results, 1), 10)
    original_query = query
    query = _inject_current_year(query)
    query = _expand_query(query)
    is_time_sensitive = _TIME_SENSITIVE_RE.search(original_query) is not None

    news_url = _SO360_NEWS_URL + "?" + urllib.parse.urlencode({"q": query})
    html = _fetch(news_url, timeout_sec=10)
    if html:
        results = _parse_so360_news(html, max_results)
        if results:
            results = _filter_redirect_urls(results)
            if is_time_sensitive:
                results = _boost_recent_results(results)
            return {"results": results, "query": query, "engine": "so360-news"}

    return {"results": [], "message": "未找到相关新闻，请尝试其他关键词"}