"""news_search — news search using Bing News (primary) with 360 News fallback.

For general web search, use web_search instead.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

from ..base import ToolDefinition
from ._common import (
    SKIP_TITLES,
    boost_recent_results,
    cache_get,
    cache_key,
    cache_set,
    clean_html,
    fetch,
    filter_redirect_urls,
    inject_current_year,
    is_blocked_page,
    is_time_sensitive,
    race_engines,
)

tool_definition = ToolDefinition(
    name="news_search",
    description='搜索新闻，返回最新新闻的标题、摘要和链接。用于：时效性事件、最新动态、当天新闻、近期热点。⚠️ 当用户问「最近/最新/今天/这周/本月」时必须用此工具。不要用于搜概念解释或历史知识，那些用 web_search。',
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

_BING_NEWS_URL = "https://www.bing.com/news/search"
_SO360_NEWS_URL = "https://news.so.com/ns"


# ---- Bing News parser ----

_BING_NEWS_TITLE_RE = re.compile(
    r'<a[^>]*class="[^"]*title[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL,
)

_BING_NEWS_SNIPPET_RE = re.compile(
    r'<div[^>]*class="[^"]*snippet[^"]*"[^>]*>(.*?)</div>',
    re.DOTALL,
)


def _parse_bing_news(html: str, limit: int) -> list[dict[str, str]]:
    """Parse Bing News search results page."""
    results: list[dict[str, str]] = []

    # Extract titles + URLs from news-card divs
    cards = re.findall(
        r'<div[^>]*class="[^"]*news-card[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>',
        html, re.DOTALL,
    )
    if not cards:
        cards = re.findall(
            r'<div[^>]*class="[^"]*card[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>',
            html, re.DOTALL,
        )

    for card in cards:
        if len(results) >= limit:
            break
        title_match = _BING_NEWS_TITLE_RE.search(card)
        if not title_match:
            title_match = re.search(
                r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                card, re.DOTALL,
            )
        if not title_match:
            continue
        url = title_match.group(1)
        title = clean_html(title_match.group(2))
        if not title:
            continue
        results.append({"title": title, "url": url, "snippet": ""})

    # Extract snippets from the full page (siblings of news-card, not nested)
    snippets = _BING_NEWS_SNIPPET_RE.findall(html)
    for i, snippet_html in enumerate(snippets):
        if i < len(results):
            results[i]["snippet"] = clean_html(snippet_html)

    return results


# ---- 360 News parser (fallback) ----

def _parse_so360_news(html: str, limit: int) -> list[dict[str, str]]:
    """Parse 360 News search results page."""
    results: list[dict[str, str]] = []
    all_h3 = re.findall(r"<h3[^>]*>(.*?)</h3>", html, re.DOTALL)

    for h3_content in all_h3:
        if len(results) >= limit:
            break
        title = clean_html(h3_content)
        if not title or title in SKIP_TITLES:
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
        all_snippets = [s for s in all_snippets if 30 < len(clean_html(s)) < 500]

    for i, snippet_html in enumerate(all_snippets):
        if i < len(results):
            results[i]["snippet"] = clean_html(snippet_html)

    return results


# ---- public API ----

def news_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search news: Bing News + 360 News (parallel) → Financial sites (fallback)."""
    max_results = min(max(max_results, 1), 10)
    original_query = query
    query = inject_current_year(query)
    ts = is_time_sensitive(original_query)

    # Check cache first
    ckey = cache_key("news_search", query, max_results)
    cached = cache_get(ckey)
    if cached is not None:
        return cached

    # Define engine fetch functions for parallel racing
    def _try_bing_news() -> dict[str, Any] | None:
        is_zh = any("\u4e00" <= c <= "\u9fff" for c in query)
        bing_params = {"q": query}
        if is_zh:
            bing_params["setlang"] = "zh-cn"
            bing_params["cc"] = "cn"
            bing_params["setmkt"] = "zh-CN"
        else:
            bing_params["setlang"] = "en"
        bing_url = _BING_NEWS_URL + "?" + urllib.parse.urlencode(bing_params)
        html = fetch(bing_url, timeout_sec=10, label="bing-news")
        if html and not is_blocked_page(html):
            results = _parse_bing_news(html, max_results)
            if results:
                return {"results": filter_redirect_urls(results), "query": query, "engine": "bing-news"}
        return None

    def _try_so360_news() -> dict[str, Any] | None:
        news_url = _SO360_NEWS_URL + "?" + urllib.parse.urlencode({"q": query})
        html = fetch(news_url, timeout_sec=10, label="so360-news")
        if html and not is_blocked_page(html):
            results = _parse_so360_news(html, max_results)
            if results:
                return {
                    "results": boost_recent_results(filter_redirect_urls(results), ts),
                    "query": query,
                    "engine": "so360-news",
                }
        return None

    # Race Bing News + 360 News in parallel
    winner = race_engines([
        ("bing-news", _try_bing_news),
        ("so360-news", _try_so360_news),
    ], timeout_sec=12)

    if winner:
        _, result = winner
        cache_set(ckey, result)
        return result

    result = {"results": [], "message": "未找到相关新闻，请尝试其他关键词"}
    cache_set(ckey, result)
    return result
