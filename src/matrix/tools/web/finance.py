"""finance_query — real-time market data via structured APIs.

Queries A-share indices, global indices, US stocks, and HK stocks
using the Sina hq API. Returns structured JSON with precise numbers,
not news articles.

For news about finance, use news_search instead.
"""

from __future__ import annotations

import logging
from typing import Any

from ..base import ToolDefinition
from ._codes import resolve_code, resolve_codes
from ._sina import fetch_quotes

logger = logging.getLogger("matrix.tools.web.finance")

tool_definition = ToolDefinition(
    name="finance_query",
    description=(
        "查询实时行情数据（A股指数/全球指数/美股/港股）。用于：股价、大盘指数、涨跌幅、行情走势。"
        "返回精确数值（价格、涨跌额、涨跌幅），不是新闻。"
        "⚠️ 用户问「今天股市」「大盘多少点」「苹果股价」「全球股市表现」时用此工具，不要用 news_search。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "查询内容，支持自然语言。例如："
                    "「上证指数」「创业板」「苹果股价」「特斯拉」「腾讯」"
                    "「全球股市」「美股」「港股」「A股」「亚太股市」「欧洲股市」"
                ),
            },
            "market": {
                "type": "string",
                "description": "指定市场，可选。默认 auto 自动识别。可选值：a_share / us / hk / global",
                "default": "auto",
            },
        },
        "required": ["query"],
    },
    handler=None,  # replaced at registration time
)


def finance_query(query: str, market: str = "auto") -> dict[str, Any]:
    """Query real-time market data.

    Args:
        query: Natural language query.
        market: Optional market hint (auto/a_share/us/hk/global).

    Returns:
        Dict with 'results' list and 'query' string.
    """
    # Try to resolve to specific codes
    codes: list[str] = []

    # First try single-code resolution (exact match)
    single = resolve_code(query)
    if single:
        codes = [single]
    else:
        # Try multi-code resolution (keyword matching)
        codes = resolve_codes(query)

    # If market is specified, it may override/augment
    if not codes and market != "auto":
        codes = _market_default_codes(market)

    if not codes:
        return {
            "results": [],
            "query": query,
            "message": "未识别到有效的查询目标。请尝试更具体的关键词，如「上证指数」「苹果股价」「全球股市」。",
        }

    # Fetch quotes
    quotes = fetch_quotes(codes)

    if not quotes:
        return {
            "results": [],
            "query": query,
            "message": "行情数据获取失败，请稍后重试。",
        }

    # Format output
    return {
        "results": quotes,
        "query": query,
        "count": len(quotes),
    }


def _market_default_codes(market: str) -> list[str]:
    """Return default codes for a given market."""
    if market == "a_share":
        return ["s_sh000001", "s_sz399001", "s_sz399006", "s_sh000300"]
    elif market == "us":
        return ["int_dji", "int_nasdaq", "int_sp500"]
    elif market == "hk":
        return ["int_hangseng"]
    elif market == "global":
        return [
            "int_dji", "int_nasdaq", "int_sp500",
            "int_hangseng", "int_nikkei",
            "s_sh000001", "s_sz399001",
        ]
    return []
