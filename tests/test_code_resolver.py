"""Deterministic tests for market code aliases and finance tool routing."""

from __future__ import annotations

from unittest.mock import patch


def test_fast_path_aliases_use_westock_codes():
    from matrix.tools.web._codes import _check_fast_path, _a_share_prefix

    assert _a_share_prefix("600519") == "sh600519"
    assert _a_share_prefix("000001") == "sz000001"
    assert _check_fast_path("上证指数")[0].provider_code == "sh000001"
    assert _check_fast_path("恒生科技")[0].provider_code == "hkHSTECH"
    assert _check_fast_path("道琼斯")[0].provider_code == "us.DJI"
    assert len(_check_fast_path("A股")) == 4


def test_unknown_name_is_resolved_by_personal_os():
    from matrix.tools.web.finance import finance_query

    with patch(
        "matrix.tools.web.finance.resolve_security",
        return_value={"matches": [{"symbol": "sh600585", "name": "海螺水泥"}]},
    ), patch(
        "matrix.tools.web.finance.market_quote",
        return_value={
            "code": "sh600585",
            "price": 20.5,
            "change": 0.5,
            "change_pct": 2.5,
            "metadata": {"provider": "westock-data"},
        },
    ):
        result = finance_query("海螺水泥")

    assert result["count"] == 1
    assert result["results"][0]["code"] == "sh600585"
    assert result["results"][0]["name"] == "海螺水泥"


def test_market_overview_uses_personal_os_quotes_without_network_in_agent():
    from matrix.tools.web.finance import finance_query

    def fake_quote(code: str):
        return {"code": code, "price": 10.0, "metadata": {"provider": "westock-data"}}

    with patch("matrix.tools.web.finance.market_quote", side_effect=fake_quote):
        result = finance_query("A股")

    assert result["count"] == 4
    assert {item["code"] for item in result["results"]} == {
        "sh000001", "sz399001", "sz399006", "sh000300",
    }


def test_finance_query_returns_structured_error_when_personal_os_unavailable():
    from matrix.tools.web.finance import finance_query

    with patch("matrix.tools.web.finance.resolve_security", return_value={"error": "unavailable"}):
        result = finance_query("完全不存在的股票xyz")

    assert "error" in result
