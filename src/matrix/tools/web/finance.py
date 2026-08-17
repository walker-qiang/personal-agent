"""finance_query — market data through the personal-os WeStock adapter."""

from __future__ import annotations

import logging
from typing import Any

from ..base import ToolDefinition, tool_error
from ..personal_os import market_quote, resolve_security
from ._codes import _check_fast_path

logger = logging.getLogger("matrix.tools.web.finance")

tool_definition = ToolDefinition(
    name="finance_query",
    description=(
        "查询行情数据（A股指数/全球指数/美股/港股/ETF）。用于：股价、大盘指数、涨跌幅、行情走势。"
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
    """Query market data through personal-os.

    Args:
        query: Natural language query, e.g., "海螺水泥", "恒生科技", "苹果".
        market: Optional market hint (auto/a_share/us/hk/global).

    Returns:
        Dict with 'results' list and 'query' string.
    """
    fast = _check_fast_path(query) if query.strip() else None
    targets: list[tuple[str, str]] = []
    if fast:
        targets = [(item.provider_code, item.display_name) for item in fast]
    elif not query.strip() and market != "auto":
        targets = [(code, "") for code in _market_default_codes(market)]
    elif query.strip():
        resolved = resolve_security(query)
        for match in resolved.get("matches", []) if isinstance(resolved, dict) else []:
            code = str(match.get("symbol") or match.get("canonical_symbol") or "").strip()
            if code:
                targets.append((code, str(match.get("name") or "")))

    if not targets:
        return tool_error(
            "finance_query", "行情查询",
            f"未识别到有效的查询目标: {query}",
            "请尝试更具体的关键词，如「上证指数」「苹果股价」「全球股市」「A股」「美股」。",
            {"query": query},
        )

    quotes: list[dict[str, Any]] = []
    failures: list[str] = []
    for code, display_name in targets:
        quote = market_quote(code)
        if not isinstance(quote, dict) or quote.get("error"):
            failures.append(code)
            continue
        result = dict(quote)
        result["code"] = result.get("code") or code
        if display_name:
            result["name"] = display_name
        quotes.append(result)

    if not quotes:
        return tool_error(
            "finance_query", "行情查询",
            f"行情数据获取失败，可能是网络问题或 API 限流: {query}",
            "请稍后重试，或尝试使用 news_search 获取相关财经新闻。",
            {"query": query, "codes": [code for code, _ in targets], "failures": failures},
        )

    # Format output
    return {
        "results": quotes,
        "query": query,
        "count": len(quotes),
        "warnings": [f"部分标的行情不可用: {', '.join(failures)}"] if failures else [],
    }


def _market_default_codes(market: str) -> list[str]:
    """Return default codes for a given market."""
    if market == "a_share":
        return ["sh000001", "sz399001", "sz399006", "sh000300"]
    elif market == "us":
        return ["us.DJI", "us.IXIC", "us.INX"]
    elif market == "hk":
        return ["hkHSI", "hkHSTECH"]
    elif market == "global":
        return [
            "us.DJI", "us.IXIC", "us.INX",
            "hkHSI", "sh000001", "sz399001",
        ]
    return []
