"""news_search — news-only search using 360 News.

For general web search, use web_search instead.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
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


def news_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search news using 360 News."""
    max_results = min(max(max_results, 1), 10)

    news_url = _SO360_NEWS_URL + "?" + urllib.parse.urlencode({"q": query})
    html = _fetch(news_url, timeout_sec=10)
    if html:
        results = _parse_so360_news(html, max_results)
        if results:
            return {"results": results, "query": query, "engine": "so360-news"}

    return {"results": [], "message": "未找到相关新闻，请尝试其他关键词"}