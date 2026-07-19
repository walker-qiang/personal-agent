"""web_search — general web search with multi-engine fallback.

Tries 360 first, then Bing, then DuckDuckGo.
For news / time-sensitive queries, use the news_search tool instead.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from ..base import ToolDefinition

tool_definition = ToolDefinition(
    name="web_search",
    description="搜索互联网，返回网页标题、摘要和链接。用于事实核查、知识查询。如需最新新闻请用 news_search。",
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

# ---- HTTP helpers ----

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

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


# ---- 360 web search ----

_SO360_URL = "https://www.so.com/s"

_SKIP_TITLES = {"其他人还搜了", "最新相关消息", "相关搜索", "最新相关信息"}


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
        title = _clean_html(h3_content)
        if not title or title in _SKIP_TITLES:
            continue
        link_match = re.search(r'href="([^"]+)"', h3_content)
        url = link_match.group(1) if link_match else ""
        results.append({"title": title, "url": url, "snippet": ""})

    for i, snippet_html in enumerate(all_snippets):
        if i < len(results):
            results[i]["snippet"] = _clean_html(snippet_html)

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
        title = _clean_html(link_match.group(2))
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
                snippet = _clean_html(snippet_match.group(1))
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
        title = _clean_html(link_match.group(2))
        if not title.strip():
            continue
        snippet_match = re.search(
            r'<[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|td|div|span)>',
            block, re.DOTALL,
        )
        snippet = _clean_html(snippet_match.group(1)) if snippet_match else ""
        results.append({"title": title, "url": url, "snippet": snippet})

    return results


# ---- URL filter: strip search-engine redirect links ----

# 360 redirect URLs (so.com/link?m=...) and AI search result pages (ai.so.com/search/...)
# These are not real web pages — LLM should not attempt web_fetch on them
_SEARCH_ENGINE_REDIRECT_RE = re.compile(
    r"^https?://(?:www\.)?so\.com/link\?|"
    r"^https?://ai\.so\.com/search/",
    re.IGNORECASE,
)


def _filter_redirect_urls(results: list[dict[str, str]]) -> list[dict[str, str]]:
    """Replace redirect/synthetic URLs with empty string so LLM won't try to web_fetch them."""
    for r in results:
        url = r.get("url", "")
        if url and _SEARCH_ENGINE_REDIRECT_RE.search(url):
            r["url"] = ""
    return results


# ---- public API ----

# Time-sensitive keywords that indicate the query needs the current year
_TIME_SENSITIVE_RE = re.compile(
    r"最近|最新|今天|今日|近期|当前|现在|近日|今年|本(?:周|月|年)",
)

# Multi-year patterns that dilute search results (e.g. "2025 2026" → "2026")
_MULTI_YEAR_RE = re.compile(r"\b(?:20\d{2}[-\s]*)+20\d{2}\b|\b20\d{2}\s+20\d{2}\b")


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


# Year extraction from snippet text
_YEAR_IN_SNIPPET_RE = re.compile(r"\b(20\d{2})\b")


def _result_year_score(snippet: str, current_year: int) -> int:
    """Score a result by recency: current year = 100, each year older = -100.
    No year found = 0 (neutral)."""
    years = _YEAR_IN_SNIPPET_RE.findall(snippet)
    if not years:
        return 0
    best = max(int(y) for y in years)
    return 100 - (current_year - best) * 100


def _boost_recent_results(results: list[dict[str, str]], is_time_sensitive: bool) -> list[dict[str, str]]:
    """If time-sensitive, reorder results so recent-year results appear first.
    Old-year results are demoted to the bottom."""
    if not is_time_sensitive or len(results) <= 1:
        return results
    current_year = datetime.now(timezone.utc).year
    scored = [(r, _result_year_score(r.get("snippet", ""), current_year)) for r in results]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored]


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """General web search. For news, use news_search."""
    max_results = min(max(max_results, 1), 10)
    original_query = query
    query = _inject_current_year(query)
    is_time_sensitive = _TIME_SENSITIVE_RE.search(original_query) is not None

    # Engine 1: 360
    so360_url = _SO360_URL + "?" + urllib.parse.urlencode({"q": query})
    html = _fetch(so360_url, timeout_sec=10)
    if html:
        results = _parse_so360(html, max_results)
        if results:
            return {"results": _boost_recent_results(_filter_redirect_urls(results), is_time_sensitive), "query": query, "engine": "so360"}

    # Engine 2: Bing
    bing_url = _BING_URL + "?" + urllib.parse.urlencode({
        "q": query,
        "setlang": "zh-cn" if any("\u4e00" <= c <= "\u9fff" for c in query) else "en",
    })
    html = _fetch(bing_url, timeout_sec=10)
    if html:
        results = _parse_bing(html, max_results)
        if results:
            return {"results": _boost_recent_results(_filter_redirect_urls(results), is_time_sensitive), "query": query, "engine": "bing"}

    # Engine 3: DuckDuckGo
    ddg_data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    ddg_req = urllib.request.Request(
        _DDG_URL, data=ddg_data,
        headers={"User-Agent": _UA, "Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(ddg_req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        html = None

    if html:
        results = _parse_ddg(html, max_results)
        if results:
            return {"results": _boost_recent_results(results, is_time_sensitive), "query": query, "engine": "duckduckgo"}

    return {"results": [], "message": "未找到相关结果，请尝试其他关键词"}