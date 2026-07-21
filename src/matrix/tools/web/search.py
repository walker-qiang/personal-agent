"""web_search — general web search with multi-engine fallback.

Tries 360 first, then Bing, then DuckDuckGo.
For news / time-sensitive queries, use the news_search tool instead.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from typing import Any

from ..base import ToolDefinition
from ._common import (
    UA,
    SKIP_TITLES,
    boost_recent_results,
    cache_get,
    cache_key,
    cache_set,
    clean_html,
    fetch,
    filter_redirect_urls,
    inject_current_year,
    is_time_sensitive,
    race_engines,
)

tool_definition = ToolDefinition(
    name="web_search",
    description="搜索互联网，返回网页标题、摘要和链接。用于：事实核查、概念解释、历史事件、知识查询等非时效性搜索。⚠️ 不要用于搜最新新闻，搜新闻必须用 news_search。",
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
    handler=None,  # replaced at registration time
)


# ---- 360 web search ----

_SO360_URL = "https://www.so.com/s"


def _parse_so360(html: str, limit: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    all_h3 = re.findall(r"<h3[^>]*>(.*?)</h3>", html, re.DOTALL)
    all_snippets = re.findall(
        r'<p[^>]*class="[^"]*res-desc[^"]*"[^>]*>(.*?)</p>',
        html, re.DOTALL,
    )

    for h3_content in all_h3:
        if len(results) >= limit:
            break
        title = clean_html(h3_content)
        if not title or title in SKIP_TITLES:
            continue
        link_match = re.search(r'href="([^"]+)"', h3_content)
        url = link_match.group(1) if link_match else ""
        results.append({"title": title, "url": url, "snippet": ""})

    for i, snippet_html in enumerate(all_snippets):
        if i < len(results):
            results[i]["snippet"] = clean_html(snippet_html)

    return results


# ---- Bing search ----

_BING_URL = "https://www.bing.com/search"


def _parse_bing(html: str, limit: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    blocks = re.split(r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>', html)

    for block in blocks[1:]:
        if len(results) >= limit:
            break
        link_match = re.search(
            r'<h2[^>]*>.*?<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
            block, re.DOTALL,
        )
        if not link_match:
            continue
        url = link_match.group(1)
        title = clean_html(link_match.group(2))
        if not title.strip():
            continue
        snippet = ""
        cap_match = re.search(
            r'<div[^>]*class="[^"]*b_caption[^"]*"[^>]*>(.*?)(?:</li>|$)',
            block, re.DOTALL,
        )
        if cap_match:
            snippet_match = re.search(r"<p[^>]*>(.*?)</p>", cap_match.group(1), re.DOTALL)
            if snippet_match:
                snippet = clean_html(snippet_match.group(1))
        results.append({"title": title, "url": url, "snippet": snippet})

    return results


# ---- DuckDuckGo search ----

_DDG_URL = "https://html.duckduckgo.com/html/"


def _parse_ddg(html: str, limit: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    blocks = re.split(r'<div[^>]*class="[^"]*result[^"]*"[^>]*>', html)

    for block in blocks[1:]:
        if len(results) >= limit:
            break
        link_match = re.search(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            block, re.DOTALL,
        )
        if not link_match:
            link_match = re.search(
                r'<a[^>]*href="(https?://[^"]+)"[^>]*class="[^"]*result[^"]*"[^>]*>(.*?)</a>',
                block, re.DOTALL,
            )
        if not link_match:
            continue
        url = link_match.group(1)
        title = clean_html(link_match.group(2))
        if not title.strip():
            continue
        snippet_match = re.search(
            r'<[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|td|div|span)>',
            block, re.DOTALL,
        )
        snippet = clean_html(snippet_match.group(1)) if snippet_match else ""
        results.append({"title": title, "url": url, "snippet": snippet})

    return results


# ---- public API ----

def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """General web search. For news, use news_search."""
    max_results = min(max(max_results, 1), 10)
    original_query = query
    query = inject_current_year(query)
    ts = is_time_sensitive(original_query)

    # Check cache first
    ckey = cache_key("web_search", query, max_results)
    cached = cache_get(ckey)
    if cached is not None:
        return cached

    # Define engine fetch functions for parallel racing
    def _try_so360() -> dict[str, Any] | None:
        so360_url = _SO360_URL + "?" + urllib.parse.urlencode({"q": query})
        html = fetch(so360_url, timeout_sec=10, label="so360")
        if html:
            results = _parse_so360(html, max_results)
            if results:
                return {"results": boost_recent_results(filter_redirect_urls(results), ts), "query": query, "engine": "so360"}
        return None

    def _try_bing() -> dict[str, Any] | None:
        bing_url = _BING_URL + "?" + urllib.parse.urlencode({
            "q": query,
            "setlang": "zh-cn" if any("\u4e00" <= c <= "\u9fff" for c in query) else "en",
        })
        html = fetch(bing_url, timeout_sec=10, label="bing")
        if html:
            results = _parse_bing(html, max_results)
            if results:
                return {"results": boost_recent_results(filter_redirect_urls(results), ts), "query": query, "engine": "bing"}
        return None

    def _try_ddg() -> dict[str, Any] | None:
        ddg_data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        ddg_req = urllib.request.Request(
            _DDG_URL, data=ddg_data,
            headers={"User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(ddg_req, timeout=8) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception:
            return None
        if html:
            results = _parse_ddg(html, max_results)
            if results:
                return {"results": boost_recent_results(results, ts), "query": query, "engine": "duckduckgo"}
        return None

    # Race all engines in parallel, take the first non-empty result
    winner = race_engines([
        ("so360", _try_so360),
        ("bing", _try_bing),
        ("ddg", _try_ddg),
    ], timeout_sec=12)

    if winner:
        _, result = winner
        cache_set(ckey, result)
        return result

    result = {"results": [], "message": "未找到相关结果，请尝试其他关键词"}
    cache_set(ckey, result)
    return result
