"""Legacy code mapping tables — kept for backward compatibility.

The actual resolution logic is now in _resolver.py, which uses the Sina suggest
API for dynamic code resolution with SQLite caching. These tables are used as a
fast-path fallback for the most common indices and market keywords.

Do not add new stocks/indices here — the resolver handles them automatically.
"""

from __future__ import annotations

import re

# ---- A-share indices (simplified format via s_ prefix) ----

A_SHARE_INDICES: dict[str, str] = {
    "上证指数": "s_sh000001",
    "上证综指": "s_sh000001",
    "沪指": "s_sh000001",
    "深证成指": "s_sz399001",
    "深成指": "s_sz399001",
    "创业板指": "s_sz399006",
    "创业板": "s_sz399006",
    "沪深300": "s_sh000300",
    "科创50": "s_sh000688",
    "上证50": "s_sh000016",
    "中证500": "s_sh000905",
    "中证1000": "s_sh000852",
}

# ---- Global indices (int_ prefix, 4 fields) ----

GLOBAL_INDICES: dict[str, str] = {
    "道琼斯": "int_dji",
    "道指": "int_dji",
    "纳斯达克": "int_nasdaq",
    "纳指": "int_nasdaq",
    "标普500": "int_sp500",
    "标普": "int_sp500",
    "日经225": "int_nikkei",
    "日经": "int_nikkei",
    "恒生指数": "int_hangseng",
    "恒指": "int_hangseng",
    "恒生科技": "hkHSTECH",
    "恒生科技指数": "hkHSTECH",
    "恒科": "hkHSTECH",
    "恒科指": "hkHSTECH",
    "韩国综合": "int_kospi",
    "富时100": "int_ftse",
    "德国dax": "int_dax",
    "法国cac": "int_cac",
}

# Regex for 6-digit A-share stock codes
_STOCK_CODE_RE = re.compile(r'^(\d{6})$')

# Merged index table for convenience
ALL_INDICES: dict[str, str] = {**A_SHARE_INDICES, **GLOBAL_INDICES}


def _a_share_code_from_digits(code: str) -> str:
    """Map a 6-digit stock code to Sina sh/sz prefix."""
    if code.startswith(("60", "68", "90")):
        return f"sh{code}"
    return f"sz{code}"


# ---- Backward-compatible interface ----
# These functions delegate to the new resolver at call time.

def resolve_code(query: str) -> str | None:
    """Try to resolve a query to a single Sina code (legacy interface).

    Delegates to _resolver.CodeResolver for dynamic resolution.
    """
    from ._resolver import CodeResolver, _check_fast_path

    # Fast-path: exact index match or 6-digit code
    fast = _check_fast_path(query)
    if fast and len(fast) == 1:
        return fast[0].sina_code
    if fast and len(fast) > 1:
        return fast[0].sina_code  # Return first match

    # Use resolver singleton
    resolver = _get_resolver()
    results = resolver.resolve(query)
    if results:
        return results[0].sina_code
    return None


def resolve_codes(query: str) -> list[str]:
    """Resolve a query to multiple Sina codes (legacy interface).

    Delegates to _resolver.CodeResolver for dynamic resolution.
    """
    from ._resolver import CodeResolver, _check_fast_path

    # Fast-path: market keywords return multiple codes
    fast = _check_fast_path(query)
    if fast:
        return [r.sina_code for r in fast]

    # Use resolver singleton
    resolver = _get_resolver()
    results = resolver.resolve(query)
    return [r.sina_code for r in results]


# ---- Module-level resolver singleton (lazy init) ----

_resolver_instance: CodeResolver | None = None


def _get_resolver() -> CodeResolver:
    global _resolver_instance
    if _resolver_instance is None:
        from ._resolver import CodeResolver
        from pathlib import Path
        _resolver_instance = CodeResolver(
            cache_path=Path.home() / ".matrix" / "cache" / "code_resolver.sqlite",
        )
    return _resolver_instance


# ---- Legacy keyword groups (kept for compatibility) ----

A_SHARE_KEYWORDS = {"a股", "a 股", "沪深", "上证", "深证", "深成", "创业板", "科创板",
                    "沪指", "深指", "大盘", "沪深300", "上证50", "科创50"}
GLOBAL_KEYWORDS = {"全球股市", "全球市场", "全球指数", "国际市场", "全球行情",
                   "海外市场", "全球主要", "全球大盘", "欧美股市", "亚太股市"}
US_KEYWORDS = {"美股", "纳斯达克", "纳指", "道琼斯", "道指", "标普"}
HK_KEYWORDS = {"港股", "恒生", "恒指", "恒生科技", "恒科"}
