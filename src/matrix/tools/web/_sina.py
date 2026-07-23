"""Sina hq API wrapper — fetch real-time quotes via hq.sinajs.cn.

Handles:
- GBK decoding
- Referer header requirement
- Multi-market field parsing (global index / A-share index / US stock / HK stock)
- Batch queries (comma-separated codes)
"""

from __future__ import annotations

import logging
import re
import urllib.request
from typing import Any

logger = logging.getLogger("matrix.tools.web.finance")

_SINA_URL = "https://hq.sinajs.cn/list="
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Pattern: var hq_str_xxx="field1,field2,...";
_VAR_RE = re.compile(r'var hq_str_(\S+?)="(.*?)"', re.DOTALL)


def _fetch_raw(codes: list[str]) -> dict[str, list[str]]:
    """Fetch quotes from Sina API. Returns {code: [field1, field2, ...]}.

    Returns empty dict on failure. Empty string fields are preserved.
    """
    url = _SINA_URL + ",".join(codes)
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA,
        "Referer": "https://finance.sina.com.cn",
    })
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read()
    except Exception as e:
        logger.warning("[sina] fetch failed: %s — %s", url[:60], type(e).__name__)
        return {}

    text = raw.decode("gbk", errors="replace")
    result: dict[str, list[str]] = {}
    for match in _VAR_RE.finditer(text):
        code = match.group(1)
        content = match.group(2)
        if not content:
            continue
        result[code] = content.split(",")
    return result


# ---- Per-market parsers ----

def _parse_global_index(code: str, fields: list[str]) -> dict[str, Any]:
    """Parse int_ format: name, price, change, change_pct (4 fields)."""
    if len(fields) < 4:
        return {"code": code, "name": fields[0] if fields else "", "error": "insufficient fields"}
    return {
        "code": code,
        "name": fields[0],
        "price": _to_float(fields[1]),
        "change": _to_float(fields[2]),
        "change_pct": _to_float(fields[3]),
    }


def _parse_a_share_index(code: str, fields: list[str]) -> dict[str, Any]:
    """Parse s_ format: name, price, change, change_pct, volume, amount (6 fields)."""
    if len(fields) < 4:
        return {"code": code, "name": fields[0] if fields else "", "error": "insufficient fields"}
    return {
        "code": code,
        "name": fields[0],
        "price": _to_float(fields[1]),
        "change": _to_float(fields[2]),
        "change_pct": _to_float(fields[3]),
        "volume": _to_int(fields[4]) if len(fields) > 4 else None,
        "amount": _to_int(fields[5]) if len(fields) > 5 else None,
    }


def _parse_us_stock(code: str, fields: list[str]) -> dict[str, Any]:
    """Parse gb_ format: 33+ fields."""
    if len(fields) < 8:
        return {"code": code, "name": fields[0] if fields else "", "error": "insufficient fields"}
    return {
        "code": code,
        "name": fields[0],
        "price": _to_float(fields[1]),
        "change_pct": _to_float(fields[2]),
        "datetime": fields[3] if len(fields) > 3 else "",
        "change": _to_float(fields[4]) if len(fields) > 4 else None,
        "open": _to_float(fields[5]) if len(fields) > 5 else None,
        "high": _to_float(fields[6]) if len(fields) > 6 else None,
        "low": _to_float(fields[7]) if len(fields) > 7 else None,
        "prev_close": _to_float(fields[26]) if len(fields) > 26 else None,
        "volume": _to_int(fields[10]) if len(fields) > 10 else None,
        "market_cap": _to_int(fields[12]) if len(fields) > 12 else None,
        "pe_ratio": _to_float(fields[14]) if len(fields) > 14 else None,
        "pre_market_price": _to_float(fields[21]) if len(fields) > 21 else None,
        "pre_market_pct": _to_float(fields[23]) if len(fields) > 23 else None,
    }


def _parse_hk_stock(code: str, fields: list[str]) -> dict[str, Any]:
    """Parse hk format: 19 fields."""
    if len(fields) < 7:
        return {"code": code, "name": fields[1] if len(fields) > 1 else "", "error": "insufficient fields"}
    return {
        "code": code,
        "name": fields[1],  # Chinese name is field[1], English is field[0]
        "price": _to_float(fields[6]),
        "open": _to_float(fields[2]) if len(fields) > 2 else None,
        "prev_close": _to_float(fields[3]) if len(fields) > 3 else None,
        "high": _to_float(fields[4]) if len(fields) > 4 else None,
        "low": _to_float(fields[5]) if len(fields) > 5 else None,
        "change": _to_float(fields[7]) if len(fields) > 7 else None,
        "change_pct": _to_float(fields[8]) if len(fields) > 8 else None,
        "volume": _to_int(fields[12]) if len(fields) > 12 else None,
        "amount": _to_int(fields[11]) if len(fields) > 11 else None,
        "pe_ratio": _to_float(fields[13]) if len(fields) > 13 else None,
        "high_52w": _to_float(fields[15]) if len(fields) > 15 else None,
        "low_52w": _to_float(fields[16]) if len(fields) > 16 else None,
        "datetime": f"{fields[17]} {fields[18]}" if len(fields) > 18 else "",
    }


def _parse_a_share_stock(code: str, fields: list[str]) -> dict[str, Any]:
    """Parse sh/sz format: 33+ fields (full A-share stock)."""
    if len(fields) < 6:
        return {"code": code, "name": fields[0] if fields else "", "error": "insufficient fields"}
    return {
        "code": code,
        "name": fields[0],
        "open": _to_float(fields[1]),
        "prev_close": _to_float(fields[2]),
        "price": _to_float(fields[3]),
        "high": _to_float(fields[4]),
        "low": _to_float(fields[5]),
        "volume": _to_int(fields[8]) if len(fields) > 8 else None,
        "amount": _to_float(fields[9]) if len(fields) > 9 else None,
        "datetime": f"{fields[30]} {fields[31]}" if len(fields) > 31 else "",
    }


# ---- Dispatch ----

def _parse_one(code: str, fields: list[str]) -> dict[str, Any]:
    """Dispatch to the correct parser based on code prefix."""
    if code.startswith("int_"):
        return _parse_global_index(code, fields)
    elif code.startswith("s_sh") or code.startswith("s_sz"):
        return _parse_a_share_index(code, fields)
    elif code.startswith("gb_"):
        return _parse_us_stock(code, fields)
    elif code.startswith("hk"):
        return _parse_hk_stock(code, fields)
    elif code.startswith("sh") or code.startswith("sz"):
        return _parse_a_share_stock(code, fields)
    else:
        return {"code": code, "name": "", "error": f"unknown code prefix: {code}"}


# ---- Public API ----

def fetch_quotes(codes: list[str]) -> list[dict[str, Any]]:
    """Fetch and parse quotes for a list of Sina codes.

    Returns a list of parsed quote dicts. Failed codes are omitted.
    """
    if not codes:
        return []

    raw = _fetch_raw(codes)
    if not raw:
        return []

    results: list[dict[str, Any]] = []
    for code in codes:
        fields = raw.get(code)
        if not fields:
            continue
        parsed = _parse_one(code, fields)
        if "error" not in parsed:
            results.append(parsed)
        else:
            logger.debug("[sina] parse error for %s: %s", code, parsed.get("error"))
    return results


# ---- Helpers ----

def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _to_int(s: str) -> int | None:
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None
