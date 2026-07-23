"""Shared helpers for web search tools.

Eliminates duplication between search.py and news_search.py:
- HTTP fetch with logging
- HTML cleaning
- Query preprocessing (time-sensitive, year injection)
- Result ranking (recency boost)
- URL filtering (search-engine redirects)
"""

from __future__ import annotations

import logging
import re
import urllib.request
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("matrix.tools.web")

# ---- HTTP helpers ----

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_STRIP_HTML = re.compile(r"<[^>]+>")
_STRIP_ENTITY = re.compile(r"&[a-z]+;")
_COLLAPSE_SPACE = re.compile(r"\s+")


def clean_html(text: str) -> str:
    """Strip HTML tags, entities, and collapse whitespace."""
    text = _STRIP_HTML.sub(" ", text)
    text = _STRIP_ENTITY.sub(" ", text)
    text = _COLLAPSE_SPACE.sub(" ", text)
    return text.strip()


def fetch(url: str, timeout_sec: float = 10, label: str = "") -> str | None:
    """Fetch a URL and return HTML text, or None on failure.

    Unlike the old _fetch, this logs warnings on failure so we can
    distinguish 'empty results' from 'request failed'.
    """
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        tag = f"[{label}] " if label else ""
        logger.warning("%sfetch failed: %s — %s", tag, url[:80], type(e).__name__)
        return None


def is_blocked_page(html: str | None) -> bool:
    """Detect anti-bot block pages that return 200 OK but contain no results.

    360 Search returns a small (~6KB) '访问异常页面' for Chinese queries.
    DuckDuckGo sometimes returns a challenge page when rate-limited.
    """
    if not html or len(html) < 8000:
        return True
    if "访问异常" in html[:2000]:
        return True
    return False


# ---- Skip titles (common noise from search engines) ----

SKIP_TITLES = {"其他人还搜了", "最新相关消息", "相关搜索", "最新相关信息"}


# ---- Query preprocessing ----

TIME_SENSITIVE_RE = re.compile(
    r"最近|最新|今天|今日|近期|当前|现在|近日|今年|本(?:周|月|年)",
)

MULTI_YEAR_RE = re.compile(r"\b(?:20\d{2}[-\s]*)+20\d{2}\b|\b20\d{2}\s+20\d{2}\b")

YEAR_IN_SNIPPET_RE = re.compile(r"\b(20\d{2})\b")


def inject_current_year(query: str) -> str:
    """Strip time-sensitive keywords and append current year if missing.

    Time-sensitive words like '今天/最新' bias search engines toward old
    articles that contain those exact words. Stripping them and appending
    the current year produces fresher results.
    """
    if not TIME_SENSITIVE_RE.search(query) and not MULTI_YEAR_RE.search(query):
        return query
    current_year = str(datetime.now(timezone.utc).year)
    cleaned = TIME_SENSITIVE_RE.sub("", query)
    cleaned = MULTI_YEAR_RE.sub(current_year, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if current_year not in cleaned:
        cleaned = f"{cleaned} {current_year}"
    return cleaned


def is_time_sensitive(query: str) -> bool:
    """Check if a query contains time-sensitive keywords."""
    return TIME_SENSITIVE_RE.search(query) is not None


# ---- Result ranking ----

def result_year_score(snippet: str, current_year: int) -> int:
    """Score a result by recency: current year = 100, each year older = -100."""
    years = YEAR_IN_SNIPPET_RE.findall(snippet)
    if not years:
        return 0
    best = max(int(y) for y in years)
    return 100 - (current_year - best) * 100


def boost_recent_results(
    results: list[dict[str, str]],
    time_sensitive: bool = True,
) -> list[dict[str, str]]:
    """Reorder results so recent-year results appear first.

    Only reorders when time_sensitive=True (default). Pass False to skip.
    """
    if not time_sensitive or len(results) <= 1:
        return results
    current_year = datetime.now(timezone.utc).year
    scored = [(r, result_year_score(r.get("snippet", ""), current_year)) for r in results]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored]


# ---- URL filtering ----

# 360 redirect URLs and AI search result pages — not real web pages
SEARCH_ENGINE_REDIRECT_RE = re.compile(
    r"^https?://(?:www\.)?so\.com/link\?|"
    r"^https?://ai\.so\.com/search/",
    re.IGNORECASE,
)


def filter_redirect_urls(results: list[dict[str, str]]) -> list[dict[str, str]]:
    """Replace search-engine redirect/synthetic URLs with empty string."""
    for r in results:
        url = r.get("url", "")
        if url and SEARCH_ENGINE_REDIRECT_RE.search(url):
            r["url"] = ""
    return results


# ---- Search result cache (TTL 5 minutes) ----

import time
import threading

_CACHE_TTL_SEC = 300  # 5 minutes
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, Any]] = {}


def cache_get(key: str) -> Any | None:
    """Get a cached search result if still valid, else None."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > _CACHE_TTL_SEC:
            del _cache[key]
            return None
        return value


def cache_set(key: str, value: Any) -> None:
    """Cache a search result with current timestamp."""
    with _cache_lock:
        # Evict expired entries when cache grows beyond 100 entries
        if len(_cache) > 100:
            now = time.time()
            expired = [k for k, (ts, _) in _cache.items() if now - ts > _CACHE_TTL_SEC]
            for k in expired:
                del _cache[k]
        _cache[key] = (time.time(), value)


def cache_key(tool_name: str, query: str, max_results: int) -> str:
    """Build a cache key from tool name and parameters."""
    return f"{tool_name}:{query}:{max_results}"


# ---- Parallel engine racing ----

from concurrent.futures import ThreadPoolExecutor, as_completed


def race_engines(
    engines: list[tuple[str, callable]],
    timeout_sec: float = 12,
) -> tuple[str, Any] | None:
    """Run multiple search engines in parallel, return the first non-empty result.

    Each engine is a (label, fetch_fn) tuple where fetch_fn() returns a result
    dict or None. The first engine to return a non-None, non-empty result wins.
    Engines that fail or return empty are silently ignored.

    Returns (winning_label, result) or None if all engines fail.
    """
    with ThreadPoolExecutor(max_workers=len(engines)) as executor:
        futures = {
            executor.submit(fn): label
            for label, fn in engines
        }
        for future in as_completed(futures, timeout=timeout_sec):
            label = futures[future]
            try:
                result = future.result()
                if result:
                    return (label, result)
            except Exception as e:
                logger.warning("[%s] engine race failed: %s", label, type(e).__name__)
                continue
    return None
