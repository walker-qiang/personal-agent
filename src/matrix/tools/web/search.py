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


# ---- public API ----


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """General web search. For news, use news_search."""
    max_results = min(max(max_results, 1), 10)

    # Engine 1: 360
    so360_url = _SO360_URL + "?" + urllib.parse.urlencode({"q": query})
    html = _fetch(so360_url, timeout_sec=10)
    if html:
        results = _parse_so360(html, max_results)
        if results:
            return {"results": results, "query": query, "engine": "so360"}

    # Engine 2: Bing
    bing_url = _BING_URL + "?" + urllib.parse.urlencode({
        "q": query,
        "setlang": "zh-cn" if any("\u4e00" <= c <= "\u9fff" for c in query) else "en",
    })
    html = _fetch(bing_url, timeout_sec=10)
    if html:
        results = _parse_bing(html, max_results)
        if results:
            return {"results": results, "query": query, "engine": "bing"}

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
            return {"results": results, "query": query, "engine": "duckduckgo"}

    return {"results": [], "message": "未找到相关结果，请尝试其他关键词"}