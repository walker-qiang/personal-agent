"""Fast-path market code helpers.

Dynamic security resolution and quote retrieval belong to personal-os. This
module only keeps deterministic aliases used by the legacy finance tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedCode:
    provider_code: str
    display_name: str
    market: str
    source: str = "fast_path"


def _a_share_prefix(code: str) -> str:
    if code.startswith(("60", "68", "90")):
        return f"sh{code}"
    return f"sz{code}"


_DIGIT_CODE_RE = re.compile(r"^\d{6}$")

_MARKET_OVERVIEW: dict[str, list[ResolvedCode]] = {
    "a股": [ResolvedCode("sh000001", "上证指数", "a_share"), ResolvedCode("sz399001", "深证成指", "a_share"), ResolvedCode("sz399006", "创业板指", "a_share"), ResolvedCode("sh000300", "沪深300", "a_share")],
    "a 股": [ResolvedCode("sh000001", "上证指数", "a_share"), ResolvedCode("sz399001", "深证成指", "a_share"), ResolvedCode("sz399006", "创业板指", "a_share"), ResolvedCode("sh000300", "沪深300", "a_share")],
    "大盘": [ResolvedCode("sh000001", "上证指数", "a_share"), ResolvedCode("sz399001", "深证成指", "a_share"), ResolvedCode("sz399006", "创业板指", "a_share"), ResolvedCode("sh000300", "沪深300", "a_share")],
    "美股": [ResolvedCode("us.DJI", "道琼斯", "us"), ResolvedCode("us.IXIC", "纳斯达克", "us"), ResolvedCode("us.INX", "标普500", "us")],
    "港股": [ResolvedCode("hkHSI", "恒生指数", "hk"), ResolvedCode("hkHSTECH", "恒生科技", "hk")],
    "全球股市": [ResolvedCode("us.DJI", "道琼斯", "us"), ResolvedCode("us.IXIC", "纳斯达克", "us"), ResolvedCode("us.INX", "标普500", "us"), ResolvedCode("hkHSI", "恒生指数", "hk"), ResolvedCode("sh000001", "上证指数", "a_share"), ResolvedCode("sz399001", "深证成指", "a_share")],
}

_COMMON_INDICES: dict[str, ResolvedCode] = {
    "上证指数": ResolvedCode("sh000001", "上证指数", "a_share"), "上证综指": ResolvedCode("sh000001", "上证综指", "a_share"), "沪指": ResolvedCode("sh000001", "沪指", "a_share"),
    "深证成指": ResolvedCode("sz399001", "深证成指", "a_share"), "深成指": ResolvedCode("sz399001", "深成指", "a_share"), "创业板指": ResolvedCode("sz399006", "创业板指", "a_share"), "创业板": ResolvedCode("sz399006", "创业板", "a_share"), "沪深300": ResolvedCode("sh000300", "沪深300", "a_share"),
    "恒生指数": ResolvedCode("hkHSI", "恒生指数", "hk"), "恒指": ResolvedCode("hkHSI", "恒指", "hk"), "恒生科技": ResolvedCode("hkHSTECH", "恒生科技", "hk"), "恒生科技指数": ResolvedCode("hkHSTECH", "恒生科技指数", "hk"),
    "道琼斯": ResolvedCode("us.DJI", "道琼斯", "us"), "道指": ResolvedCode("us.DJI", "道指", "us"), "纳斯达克": ResolvedCode("us.IXIC", "纳斯达克", "us"), "纳指": ResolvedCode("us.IXIC", "纳指", "us"), "标普500": ResolvedCode("us.INX", "标普500", "us"), "标普": ResolvedCode("us.INX", "标普", "us"),
}


def _check_fast_path(query: str) -> list[ResolvedCode] | None:
    value = query.strip().lower()
    for keyword, codes in _MARKET_OVERVIEW.items():
        if keyword in value:
            return list(codes)
    if value in _COMMON_INDICES:
        return [_COMMON_INDICES[value]]
    if _DIGIT_CODE_RE.match(query.strip()):
        return [ResolvedCode(_a_share_prefix(query.strip()), query.strip(), "a_share")]
    results: list[ResolvedCode] = []
    for name, code in _COMMON_INDICES.items():
        if (name in query or (len(value) >= 2 and name.lower().startswith(value))) and code not in results:
            results.append(code)
    return results or None


def resolve_code(query: str) -> str | None:
    results = _check_fast_path(query)
    return results[0].provider_code if results else None


def resolve_codes(query: str) -> list[str]:
    results = _check_fast_path(query)
    return [item.provider_code for item in results] if results else []
