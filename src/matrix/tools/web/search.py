"""web_search tool — multi-engine web search with fallback.

Tries Bing first (works in China), falls back to DuckDuckGo.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
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


# ---- Bing search ----

_BING_URL = "https://www.bing.com/search"


def _parse_bing(html: str, limit: int) -> list[dict[str, str]]:
    """Parse Bing HTML search results page."""
    results: list[dict[str, str]] = []

    # Bing results are in <li class="b_algo"> blocks
    # Title: <h2><a href="...">title</a></h2>
    # Snippet: <p class="b_lineclamp2"> or <p> in <div class="b_caption">

    # Find b_algo blocks
    blocks = re.split(r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>', html)

    for block in blocks[1:]:
        if len(results) >= limit:
            break

        # Extract link and title from <h2><a href="...">
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

        # Extract snippet from <p> in <div class="b_caption">
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

    Tries Bing first (works in China), falls back to DuckDuckGo.
    """
    max_results = min(max(max_results, 1), 10)

    # ---- Engine 1: Bing (works in China) ----
    bing_url = _BING_URL + "?" + urllib.parse.urlencode({
        "q": query,
        "setlang": "zh-cn" if any("\u4e00" <= c <= "\u9fff" for c in query) else "en",
    })
    html = _fetch(bing_url, timeout_sec=10)
    if html:
        results = _parse_bing(html, max_results)
        if results:
            return {"results": results, "query": query, "engine": "bing"}

    # ---- Engine 2: DuckDuckGo (fallback) ----
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