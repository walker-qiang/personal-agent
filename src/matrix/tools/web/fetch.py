"""web_fetch tool — fetch and extract text from a web page."""

from __future__ import annotations

import gzip
import io
import re
import urllib.request
import zlib
from typing import Any

from ..base import ToolDefinition

tool_definition = ToolDefinition(
    name="web_fetch",
    description="获取指定网页的完整文本内容。用于：阅读搜索结果中的文章全文、获取详细信息、验证引用来源。⚠️ 需先通过 web_search 或 news_search 获取 URL，不要直接猜测 URL。",
    input_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要获取的网页 URL",
            },
            "max_chars": {
                "type": "integer",
                "description": "最大返回字符数，默认 5000，最大 20000",
                "default": 5000,
            },
        },
        "required": ["url"],
    },
    handler=None,  # replaced at registration time
)

_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_STYLE_RE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_RE = re.compile(r"&[a-z]+;")
_SPACE_RE = re.compile(r"\s+")
_META_RE = re.compile(r'<meta[^>]*name="description"[^>]*content="([^"]*)"', re.IGNORECASE)


# Search engine redirect URLs that should not be fetched
_REDIRECT_URL_RE = re.compile(
    r"^https?://(?:www\.)?so\.com/link\?|"
    r"^https?://ai\.so\.com/search/",
    re.IGNORECASE,
)


def web_fetch(url: str, max_chars: int = 5000) -> dict[str, Any]:
    """Fetch a web page and extract its text content."""
    max_chars = min(max(max_chars, 500), 20000)

    # Reject empty or search-engine redirect URLs early
    if not url or not url.strip():
        return {"error": "URL 为空，无法获取。请使用搜索结果的摘要信息直接回答，或搜索其他关键词。", "text": ""}
    if _REDIRECT_URL_RE.search(url):
        return {"error": "该 URL 是搜索引擎跳转链接，无法直接获取。请使用搜索结果的摘要信息，或搜索其他来源。", "text": ""}

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return {"error": f"不支持的内容类型: {content_type}", "text": ""}
            raw = resp.read()

            # Handle gzip/deflate compression
            content_encoding = resp.headers.get("Content-Encoding", "").lower()
            if content_encoding == "gzip":
                raw = gzip.decompress(raw)
            elif content_encoding == "deflate":
                try:
                    raw = zlib.decompress(raw)
                except zlib.error:
                    raw = zlib.decompress(raw, -zlib.MAX_WBITS)

            # Try UTF-8 first, fallback to other encodings
            html = ""
            for enc in ["utf-8", "gbk", "gb2312", "latin-1"]:
                try:
                    html = raw.decode(enc)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            if not html:
                html = raw.decode("utf-8", errors="replace")
    except Exception as err:
        return {"error": f"获取网页失败: {err}", "text": ""}

    text = _extract_text(html)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... (内容已截断)"

    return {
        "url": url,
        "text": text,
        "length": len(text),
    }


def _extract_text(html: str) -> str:
    """Extract readable text from HTML, prioritizing content areas."""
    # Try to get meta description
    meta_match = _META_RE.search(html)
    meta = f"[页面描述] {meta_match.group(1)}\n\n" if meta_match else ""

    # Try to extract <main> or <article> content first
    body = html
    for tag in ("main", "article"):
        m = re.search(
            rf"<{tag}[^>]*>(.*?)</{tag}>", html, re.DOTALL | re.IGNORECASE,
        )
        if m:
            body = m.group(1)
            break

    if body == html:
        # No main/article — remove nav, header, footer, aside
        for tag in ("nav", "header", "footer", "aside", "script", "style", "head"):
            body = re.sub(
                rf"<{tag}[^>]*>.*?</{tag}>", "", body,
                flags=re.DOTALL | re.IGNORECASE,
            )

    # Remove remaining scripts and styles
    body = _SCRIPT_RE.sub("", body)
    body = _STYLE_RE.sub("", body)

    # Remove remaining tags
    text = _TAG_RE.sub(" ", body)
    text = _ENTITY_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)

    # Clean up: remove very short lines (nav items) and blank lines
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line and len(line) > 1]
    text = "\n".join(lines)

    return meta + text