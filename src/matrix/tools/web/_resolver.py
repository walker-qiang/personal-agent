"""Dynamic stock code resolver — replaces hardcoded mapping tables.

Architecture:
    1. Fast-path: hardcoded common indices/keywords (A股, 全球股市, etc.)
    2. SQLite cache: name→code lookups from prior API searches
    3. Sina suggest API: live search for uncached names
    4. Code conversion: normalize suggest results to Sina HQ API codes

This replaces the old _codes.py approach where every stock/index had to be
manually added to a mapping table. Now any stock/index searchable on Sina
Finance can be resolved dynamically.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("matrix.tools.web.finance")

# ---- Sina suggest API ----

_SUGGEST_URL = "https://suggest3.sinajs.cn/suggest/type=&key={keyword}"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Cache TTL: 7 days (stock codes rarely change)
_CACHE_TTL_SEC = 7 * 24 * 3600


@dataclass(frozen=True)
class ResolvedCode:
    """A resolved stock/index code ready for the Sina HQ API."""
    sina_code: str        # e.g., "sh600585", "hkHSTECH", "int_dji"
    display_name: str     # e.g., "海螺水泥", "恒生科技指数"
    market: str           # "a_share" | "hk" | "us" | "global_index" | "a_share_index"
    source: str           # "fast_path" | "cache" | "api"


# ---- Code conversion ----

# Sina suggest types → market
_TYPE_MARKET = {
    "11": "a_share",       # A-share stock
    "12": "hk",             # HK stock
    "13": "us",             # US stock
    "14": "a_share_index",  # A-share index (rare in suggest)
    "22": "a_share",        # A-share ETF
    "25": "fund",           # QDII/fund
    "31": "hk",             # HK ETF
    "33": "hk",             # HK index
    "41": "us",             # ADR
}

# 6-digit A-share code → sh/sz prefix
_DIGIT_CODE_RE = re.compile(r"^\d{6}$")


def _a_share_prefix(code: str) -> str:
    """Map a 6-digit code to sh/sz prefix."""
    if code.startswith(("60", "68", "90")):
        return f"sh{code}"
    return f"sz{code}"


def _convert_suggest_entry(fields: list[str]) -> ResolvedCode | None:
    """Convert a Sina suggest API entry to a ResolvedCode.

    Sina suggest format (comma-separated):
        name, type, code, sina_code, name_cn, _, name_en, market, _, _, tags

    The `sina_code` (4th field) is usually the correct HQ API code, but some
    types need conversion (e.g., HK indices need `hk` prefix + uppercase).
    """
    if len(fields) < 4:
        return None

    name = fields[0]
    stype = fields[1]
    code = fields[2]
    sina_code = fields[3]
    market = _TYPE_MARKET.get(stype, "")

    # Skip funds and other non-quote types
    if market == "fund" or not market:
        return None

    # Convert to Sina HQ API code based on type
    if stype == "11" or stype == "22":
        # A-share stock/ETF: sina_code is already "sh600585" format
        hq_code = sina_code
        if not hq_code.startswith(("sh", "sz")):
            # Fallback: use 6-digit code with prefix
            if _DIGIT_CODE_RE.match(code):
                hq_code = _a_share_prefix(code)
            else:
                hq_code = sina_code
    elif stype == "12" or stype == "31":
        # HK stock/ETF: code is like "00700", pad to 5 digits, prefix "hk"
        hq_code = f"hk{code.zfill(5)}"
    elif stype == "13":
        # US stock: prefix "gb_" + lowercase
        hq_code = f"gb_{code.lower()}"
    elif stype == "33":
        # HK index: prefix "hk" + uppercase
        hq_code = f"hk{code.upper()}"
    elif stype == "41":
        # ADR: use sina_code as-is (usually like "gb_aapl" already)
        hq_code = sina_code if sina_code.startswith("gb_") else f"gb_{code.lower()}"
    elif stype == "14":
        # A-share index: need s_ prefix
        if code.startswith(("sh", "sz")):
            hq_code = f"s_{code}"
        else:
            hq_code = sina_code
    else:
        hq_code = sina_code

    return ResolvedCode(
        sina_code=hq_code,
        display_name=name,
        market=market,
        source="api",
    )


def _parse_suggest_response(text: str, limit: int = 5) -> list[ResolvedCode]:
    """Parse Sina suggest API response.

    Response format: `var suggestvalue="entry1;entry2;...";`
    Each entry: `name,type,code,sina_code,name_cn,...`
    """
    match = re.search(r'var suggestvalue="(.*?)"', text, re.DOTALL)
    if not match:
        return []

    body = match.group(1)
    if not body:
        return []

    results: list[ResolvedCode] = []
    seen_codes: set[str] = set()
    for entry in body.split(";"):
        if not entry:
            continue
        fields = entry.split(",")
        resolved = _convert_suggest_entry(fields)
        if resolved and resolved.sina_code not in seen_codes:
            seen_codes.add(resolved.sina_code)
            results.append(resolved)
            if len(results) >= limit:
                break

    return results


# ---- SQLite cache ----

class _CodeCache:
    """SQLite-backed cache for name→code resolutions."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS code_cache (
                    query_key TEXT NOT NULL,
                    sina_code TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    market TEXT NOT NULL,
                    cached_at REAL NOT NULL,
                    PRIMARY KEY (query_key, sina_code)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_key ON code_cache(query_key)")
            conn.commit()
            conn.close()

    def get(self, query: str) -> list[ResolvedCode]:
        """Return cached results if fresh, else empty list."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            cutoff = time.time() - _CACHE_TTL_SEC
            rows = conn.execute(
                "SELECT * FROM code_cache WHERE query_key = ? AND cached_at > ?",
                (query.lower(), cutoff),
            ).fetchall()
            conn.close()

        if not rows:
            return []
        return [
            ResolvedCode(
                sina_code=r["sina_code"],
                display_name=r["display_name"],
                market=r["market"],
                source="cache",
            )
            for r in rows
        ]

    def put(self, query: str, codes: list[ResolvedCode]) -> None:
        if not codes:
            return
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            now = time.time()
            # Delete old entries for this key
            conn.execute("DELETE FROM code_cache WHERE query_key = ?", (query.lower(),))
            conn.executemany(
                "INSERT INTO code_cache (query_key, sina_code, display_name, market, cached_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [(query.lower(), c.sina_code, c.display_name, c.market, now) for c in codes],
            )
            conn.commit()
            conn.close()


# ---- Fast-path: hardcoded common indices/keywords ----

# Market overview keyword → codes (for batch queries)
_MARKET_OVERVIEW: dict[str, list[str]] = {
    # A-share market
    "a股": ["s_sh000001", "s_sz399001", "s_sz399006", "s_sh000300"],
    "a 股": ["s_sh000001", "s_sz399001", "s_sz399006", "s_sh000300"],
    "大盘": ["s_sh000001", "s_sz399001", "s_sz399006", "s_sh000300"],
    "沪深": ["s_sh000001", "s_sz399001", "s_sz399006", "s_sh000300"],
    # US market
    "美股": ["int_dji", "int_nasdaq", "int_sp500"],
    # HK market
    "港股": ["int_hangseng", "hkHSTECH"],
    # Global overview
    "全球股市": ["int_dji", "int_nasdaq", "int_sp500", "int_hangseng", "int_nikkei",
                "s_sh000001", "s_sz399001"],
    "全球指数": ["int_dji", "int_nasdaq", "int_sp500", "int_hangseng", "int_nikkei",
                 "s_sh000001", "s_sz399001"],
    "全球市场": ["int_dji", "int_nasdaq", "int_sp500", "int_hangseng", "int_nikkei",
                 "s_sh000001", "s_sz399001"],
    "亚太股市": ["int_hangseng", "int_nikkei", "int_kospi", "s_sh000001"],
    "欧洲股市": ["int_ftse", "int_dax", "int_cac"],
}

# Common indices — fast-path exact match (no API call needed)
_COMMON_INDICES: dict[str, str] = {
    # A-share indices
    "上证指数": "s_sh000001", "上证综指": "s_sh000001", "沪指": "s_sh000001",
    "深证成指": "s_sz399001", "深成指": "s_sz399001",
    "创业板指": "s_sz399006", "创业板": "s_sz399006",
    "沪深300": "s_sh000300", "科创50": "s_sh000688",
    "上证50": "s_sh000016", "中证500": "s_sh000905", "中证1000": "s_sh000852",
    # Global indices
    "道琼斯": "int_dji", "道指": "int_dji",
    "纳斯达克": "int_nasdaq", "纳指": "int_nasdaq",
    "标普500": "int_sp500", "标普": "int_sp500",
    "日经225": "int_nikkei", "日经": "int_nikkei",
    "恒生指数": "int_hangseng", "恒指": "int_hangseng",
    "恒生科技": "hkHSTECH", "恒生科技指数": "hkHSTECH",
    "韩国综合": "int_kospi",
    "富时100": "int_ftse",
    "德国dax": "int_dax",
    "法国cac": "int_cac",
}


def _check_fast_path(query: str) -> list[ResolvedCode] | None:
    """Check hardcoded fast-path for common indices and market keywords.

    Returns None if not found (should try cache/API).
    """
    q = query.strip().lower()

    # Market overview keywords
    for kw, codes in _MARKET_OVERVIEW.items():
        if kw in q:
            return [
                ResolvedCode(sina_code=c, display_name="", market="overview", source="fast_path")
                for c in codes
            ]

    # Exact index match
    for name, code in _COMMON_INDICES.items():
        if q == name.lower():
            return [ResolvedCode(
                sina_code=code, display_name=name, market="index", source="fast_path",
            )]

    # 6-digit A-share code
    if _DIGIT_CODE_RE.match(query.strip()):
        return [ResolvedCode(
            sina_code=_a_share_prefix(query.strip()),
            display_name=query.strip(),
            market="a_share",
            source="fast_path",
        )]

    return None


# ---- Main resolver ----

class CodeResolver:
    """Dynamic stock code resolver with cache and API fallback.

    Resolution order:
    1. Fast-path: common indices/keywords (zero latency)
    2. SQLite cache: previously resolved queries
    3. Sina suggest API: live search (cached after first lookup)
    """

    def __init__(self, cache_path: Path | None = None) -> None:
        if cache_path is None:
            cache_path = Path.home() / ".matrix" / "cache" / "code_resolver.sqlite"
        self._cache = _CodeCache(cache_path)
        self._api_timeout = 5.0

    def resolve(self, query: str) -> list[ResolvedCode]:
        """Resolve a natural language query to Sina HQ codes.

        Args:
            query: User's query, e.g., "海螺水泥", "恒生科技", "苹果"

        Returns:
            List of ResolvedCode objects. Empty if nothing found.
        """
        # 1. Fast-path
        fast = _check_fast_path(query)
        if fast:
            return fast

        # 2. Cache lookup
        cached = self._cache.get(query)
        if cached:
            return cached

        # 3. API search
        api_results = self._search_api(query)
        if api_results:
            self._cache.put(query, api_results)
            return api_results

        # 4. Fallback: try substring matching against common indices
        return self._substring_fallback(query)

    def _search_api(self, keyword: str) -> list[ResolvedCode]:
        """Call Sina suggest API to resolve keyword to codes."""
        encoded = urllib.parse.quote(keyword)
        url = _SUGGEST_URL.format(keyword=encoded)
        req = urllib.request.Request(url, headers={
            "User-Agent": _UA,
            "Referer": "https://finance.sina.com.cn",
        })
        try:
            with urllib.request.urlopen(req, timeout=self._api_timeout) as resp:
                raw = resp.read().decode("gbk", errors="replace")
        except Exception as e:
            logger.warning("[resolver] suggest API failed for '%s': %s", keyword, type(e).__name__)
            return []

        results = _parse_suggest_response(raw, limit=5)
        if not results:
            logger.debug("[resolver] no results for '%s'", keyword)
        return results

    def _substring_fallback(self, query: str) -> list[ResolvedCode]:
        """Last resort: bidirectional substring match against common indices.

        Matches when:
        1. Index name is substring of query (e.g., "上证指数" in "上证指数走势")
        2. Query is a prefix of index name, ≥2 chars (e.g., "上证" → "上证指数")
        """
        results: list[ResolvedCode] = []
        seen: set[str] = set()
        q = query.strip()
        for name, code in _COMMON_INDICES.items():
            matched = (name in q) or (len(q) >= 2 and name.startswith(q))
            if matched and code not in seen:
                seen.add(code)
                results.append(ResolvedCode(
                    sina_code=code, display_name=name, market="index", source="fast_path",
                ))
        return results
