"""web_search tool — multi-engine web search with fallback.

Tries 360 (news for time-sensitive queries, web otherwise), then Bing, then DuckDuckGo.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from ..base import ToolDefinition

tool_definition = ToolDefinition(
    name="web_search",
    description="搜索互联网，返回相关网页的标题、摘要和链接。用于查询最新信息、事实核查、或任何需要外部知识的问题。",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，英文或中文均可。查询最近/最新消息时请使用具体日期限定（如2026年7月）",
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

# Time-sensitive keywords — if any appear in query, prefer news search
_TIME_SENSITIVE_PAT = re.compile(
    r"最近|最新|刚刚|今天|今日|本周|本月|今年|这次|本次|当前|近期|近日|"
    r"latest|recent|today|this week|this month|just now|breaking|"
    r"202[5-9]|2030"
)


def _clean_html(text: str) -> str:
    """Strip HTML tags and entities, collapse whitespace."""
    text = _STRIP_HTML.sub(" ", text)
    text = _STRIP_ENTITY.sub(" ", text)
    text = _COLLAPSE_SPACE.sub(" ", text)
    return text.strip()


def _fetch(url: str, timeout_sec: float) -> str | None:
    """Fetch a URL and return decoded text, or None on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _is_time_sensitive(query: str) -> bool:
    """Check if the query is asking for recent/latest information."""
    return bool(_TIME_SENSITIVE_PAT.search(query))


# ---- 360 web search ----

_SO360_URL = "https://www.so.com/s"


def _parse_so360(html: str, limit: int) -> list[dict[str, str]]:
    """Parse 360 search (so.com) HTML results page."""
    results: list[dict[str, str]] = []

    all_h3 = re.findall(r"<h3[^>]*>(.*?)</h3>", html, re.DOTALL)
    all_snippets = re.findall(
        r'<p[^>]*class="[^"]*res-desc[^"]*"[^>]*>(.*?)</p>',
        html, re.DOTALL,
    )

    skip_titles = {"其他人还搜了", "最新相关消息", "相关搜索", "最新相关信息"}

    for h3_content in all_h3:
        if len(results) >= limit:
            break

        title = _clean_html(h3_content)
        if not title or title in skip_titles:
            continue

        link_match = re.search(r'href="([^"]+)"', h3_content)
        url = link_match.group(1) if link_match else ""

        results.append({"title": title, "url": url, "snippet": ""})

    for i, snippet_html in enumerate(all_snippets):
        if i < len(results):
            results[i]["snippet"] = _clean_html(snippet_html)

    return results


# ---- 360 news search (for time-sensitive queries) ----

_SO360_NEWS_URL = "https://news.so.com/ns"


def _parse_so360_news(html: str, limit: int) -> list[dict[str, str]]:
    """Parse 360 news search (news.so.com) HTML results page."""
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

    # Try to extract snippets from the news page
    all_snippets = re.findall(
        r'<p[^>]*class="[^"]*(?:res-desc|news-desc|summary|abstract)[^"]*"[^>]*>(.*?)</p>',
        html, re.DOTALL,
    )
    if not all_snippets:
        # Fallback: any <p> after each h3
        all_snippets = re.findall(
            r'<p[^>]*>(.*?)</p>',
            html, re.DOTALL,
        )
        # Filter to only reasonable-length snippets
        all_snippets = [s for s in all_snippets if 30 < len(_clean_html(s)) < 500]

    for i, snippet_html in enumerate(all_snippets):
        if i < len(results):
            results[i]["snippet"] = _clean_html(snippet_html)

    return results


# ---- Bing search ----

_BING_URL = "https://www.bing.com/search"


def _parse_bing(html: str, limit: int) -> list[dict[str, str]]:
    """Parse Bing HTML search results page."""
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
            r'<div[^>]*class="[^"]*b_caption[^"]*"[^>]*>(.*?)(?:</li>|$)', block, re.DOTALL,
        )
        if cap_match:
            snippet_match = re.search(r'<p[^>]*>(.*?)</p>', cap_match.group(1), re.DOTALL)
            if snippet_match:
                snippet = _clean_html(snippet_match.group(1))

        results.append({"title": title, "url": url, "snippet": snippet})

    return results


# ---- DuckDuckGo search ----

_DDG_URL = "https://html.duckduckgo.com/html/"


def _parse_ddg(html: str, limit: int) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML search results page."""
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


# ---- public API ----


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web using multiple engines with fallback.

    For time-sensitive queries (最近/最新/今天/2026 etc.), prefers 360 News.
    Otherwise uses 360 Web, then Bing, then DuckDuckGo.
    """
    max_results = min(max(max_results, 1), 10)
    time_sensitive = _is_time_sensitive(query)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ---- Engine 1a: 360 News (for time-sensitive queries) ----
    if time_sensitive:
        news_url = _SO360_NEWS_URL + "?" + urllib.parse.urlencode({"q": query})
        html = _fetch(news_url, timeout_sec=10)
        if html:
            results = _parse_so360_news(html, max_results)
            if results:
                return {
                    "results": results,
                    "query": query,
                    "engine": "so360-news",
                    "_today": today,
                    "_hint": f"以上是新闻搜索结果。今天是 {today}，请优先参考最新日期的新闻，忽略过时信息。",
                }

    # ---- Engine 1b: 360 Web (default) ----
    so360_url = _SO360_URL + "?" + urllib.parse.urlencode({"q": query})
    html = _fetch(so360_url, timeout_sec=10)
    if html:
        results = _parse_so360(html, max_results)
        if results:
            return {
                "results": results,
                "query": query,
                "engine": "so360",
                "_today": today,
                "_hint": f"今天是 {today}。请基于搜索结果中的日期判断信息时效性，不要编造日期。",
            }

    # ---- Engine 2: Bing (fallback) ----
    bing_url = _BING_URL + "?" + urllib.parse.urlencode({
        "q": query,
        "setlang": "zh-cn" if any("\u4e00" <= c <= "\u9fff" for c in query) else "en",
    })
    html = _fetch(bing_url, timeout_sec=10)
    if html:
        results = _parse_bing(html, max_results)
        if results:
            return {
                "results": results,
                "query": query,
                "engine": "bing",
                "_today": today,
                "_hint": f"今天是 {today}。请基于搜索结果中的日期判断信息时效性，不要编造日期。",
            }

    # ---- Engine 3: DuckDuckGo (last resort) ----
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
            return {
                "results": results,
                "query": query,
                "engine": "duckduckgo",
                "_today": today,
                "_hint": f"今天是 {today}。请基于搜索结果中的日期判断信息时效性，不要编造日期。",
            }

    return {"results": [], "message": "未找到相关结果，请尝试其他关键词"}