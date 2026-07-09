"""web_search tool — search the web using DuckDuckGo HTML."""

from __future__ import annotations

import json
import re
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

# DuckDuckGo HTML search URL (no API key needed)
_SEARCH_URL = "https://html.duckduckgo.com/html/"

# Regex to extract vqd token from DuckDuckGo JS (used for API path)
_STRIP_HTML = re.compile(r"<[^>]+>")
_STRIP_ENTITY = re.compile(r"&[a-z]+;")
_COLLAPSE_SPACE = re.compile(r"\s+")


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search DuckDuckGo HTML and return structured results."""
    max_results = min(max(max_results, 1), 10)

    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    req = urllib.request.Request(
        _SEARCH_URL,
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as err:
        return {"error": f"搜索请求失败: {err}", "results": []}

    results = _parse_ddg_html(html, max_results)
    if not results:
        return {"results": [], "message": "未找到相关结果，请尝试其他关键词"}

    return {"results": results, "query": query}


def _parse_ddg_html(html: str, limit: int) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML search results page."""
    results: list[dict[str, str]] = []

    # Each result is in a div with class "result"
    # Looking for: <a class="result__a" href="...">title</a>
    # and <a class="result__snippet">snippet</a>

    # Split by result blocks
    blocks = re.split(r'<div[^>]*class="[^"]*result[^"]*"[^>]*>', html)

    for block in blocks[1:]:  # skip first (before first result)
        if len(results) >= limit:
            break

        # Extract link
        link_match = re.search(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )
        if not link_match:
            # Try alternative: rel="nofollow" links
            link_match = re.search(
                r'<a[^>]*href="(https?://[^"]+)"[^>]*class="[^"]*result[^"]*"[^>]*>(.*?)</a>',
                block,
                re.DOTALL,
            )

        if not link_match:
            continue

        url = link_match.group(1)
        title = _clean_html(link_match.group(2))

        if not title.strip():
            continue

        # Extract snippet
        snippet_match = re.search(
            r'<[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|td|div|span)>',
            block,
            re.DOTALL,
        )
        snippet = ""
        if snippet_match:
            snippet = _clean_html(snippet_match.group(1))

        results.append({
            "title": title,
            "url": url,
            "snippet": snippet,
        })

    return results


def _clean_html(text: str) -> str:
    """Strip HTML tags and entities, collapse whitespace."""
    text = _STRIP_HTML.sub(" ", text)
    text = _STRIP_ENTITY.sub(" ", text)
    text = _COLLAPSE_SPACE.sub(" ", text)
    return text.strip()