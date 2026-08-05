"""finance_query — real-time market data via structured APIs.

Queries A-share indices, global indices, US stocks, and HK stocks
using the Sina hq API. Returns structured JSON with precise numbers,
not news articles.

Code resolution is handled by _resolver.py (dynamic Sina suggest API + SQLite cache).
For news about finance, use news_search instead.
"""

from __future__ import annotations

import logging
from typing import Any

from ..base import ToolDefinition, tool_error
from ._codes import resolve_codes  # backward-compatible interface
from ._sina import fetch_quotes

logger = logging.getLogger("matrix.tools.web.finance")

tool_definition = ToolDefinition(
    name="finance_query",
    description=(
        "查询实时行情数据（A股指数/全球指数/美股/港股/ETF）。用于：股价、大盘指数、涨跌幅、行情走势。"
        "返回精确数值（价格、涨跌额、涨跌幅），不是新闻。"
        "⚠️ 用户问「今天股市」「大盘多少点」「苹果股价」「全球股市表现」时用此工具，不要用 news_search。"
        "💡 效率提示：用宽泛关键词一次查全部。例如查A股直接传 query='A股'（返回上证+深证+创业板+沪深300），"
        "不要分多次查「上证指数」「深证成指」「创业板指」。查全球股市传 query='全球股市' 即可。"
    ),
    capabilities=["market_data"],
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "查询内容，支持自然语言。例如："
                    "「上证指数」「创业板」「苹果股价」「特斯拉」「腾讯」"
                    "「全球股市」「美股」「港股」「A股」「亚太股市」「欧洲股市」"
                    "「恒生科技」「恒生指数」「海螺水泥」"
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

    Uses dynamic code resolution (Sina suggest API + SQLite cache) to resolve
    any stock/index name to its Sina HQ API code. No hardcoded mapping needed.

    Args:
        query: Natural language query, e.g., "海螺水泥", "恒生科技", "苹果".
        market: Optional market hint (auto/a_share/us/hk/global).

    Returns:
        Dict with 'results' list and 'query' string.
    """
    # Resolve via dynamic resolver (returns all matching codes)
    codes = resolve_codes(query)

    # If market is specified, it may override/augment
    if not codes and market != "auto":
        codes = _market_default_codes(market)

    if not codes:
        return tool_error(
            "finance_query", "行情查询",
            f"未识别到有效的查询目标: {query}",
            "请尝试更具体的关键词，如「上证指数」「苹果股价」「全球股市」「A股」「美股」。",
            {"query": query},
        )

    # Fetch quotes
    quotes = fetch_quotes(codes)

    if not quotes:
        return tool_error(
            "finance_query", "行情查询",
            f"行情数据获取失败，可能是网络问题或 API 限流: {query}",
            "请稍后重试，或尝试使用 news_search 获取相关财经新闻。",
            {"query": query, "codes": codes},
        )

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
        return ["int_hangseng", "hkHSTECH"]
    elif market == "global":
        return [
            "int_dji", "int_nasdaq", "int_sp500",
            "int_hangseng", "int_nikkei",
            "s_sh000001", "s_sz399001",
        ]
    return []
