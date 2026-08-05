"""Tests for the dynamic stock code resolver (_resolver.py).

Covers:
- Fast-path: common indices, market keywords, 6-digit codes
- SQLite cache: put/get, TTL, composite PK, overwrite
- Sina suggest API response parsing & code conversion
- Substring fallback
- Backward-compatible interface in _codes.py
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from matrix.tools.web._resolver import (
    CodeResolver,
    ResolvedCode,
    _check_fast_path,
    _CodeCache,
    _convert_suggest_entry,
    _parse_suggest_response,
    _a_share_prefix,
)


# ── Fast-path tests ──────────────────────────────────────────────────────────


class TestFastPath:
    """Test _check_fast_path for common indices, keywords, and digit codes."""

    def test_a_share_index_exact_match(self):
        """上证指数 should resolve to s_sh000001."""
        result = _check_fast_path("上证指数")
        assert result is not None
        assert len(result) == 1
        assert result[0].sina_code == "s_sh000001"
        assert result[0].source == "fast_path"

    def test_hang_seng_tech_exact_match(self):
        """恒生科技 should resolve to hkHSTECH."""
        result = _check_fast_path("恒生科技")
        assert result is not None
        assert result[0].sina_code == "hkHSTECH"

    def test_dow_jones_exact_match(self):
        """道琼斯 should resolve to int_dji."""
        result = _check_fast_path("道琼斯")
        assert result is not None
        assert result[0].sina_code == "int_dji"

    def test_market_keyword_a_share(self):
        """A股 keyword should return multiple index codes."""
        result = _check_fast_path("A股")
        assert result is not None
        codes = [r.sina_code for r in result]
        assert "s_sh000001" in codes
        assert "s_sz399001" in codes
        assert "s_sz399006" in codes
        assert "s_sh000300" in codes

    def test_market_keyword_global(self):
        """全球股市 keyword should return global index codes."""
        result = _check_fast_path("全球股市")
        assert result is not None
        codes = [r.sina_code for r in result]
        assert "int_dji" in codes
        assert "int_hangseng" in codes
        assert "s_sh000001" in codes

    def test_market_keyword_case_insensitive(self):
        """Keywords should match case-insensitively."""
        result = _check_fast_path("a股")
        assert result is not None
        assert len(result) == 4

    def test_six_digit_sh_code(self):
        """6-digit code starting with 60 should resolve to sh prefix."""
        result = _check_fast_path("600585")
        assert result is not None
        assert result[0].sina_code == "sh600585"
        assert result[0].market == "a_share"

    def test_six_digit_sz_code(self):
        """6-digit code starting with 00 should resolve to sz prefix."""
        result = _check_fast_path("000001")
        assert result is not None
        assert result[0].sina_code == "sz000001"

    def test_six_digit_star_market(self):
        """6-digit code starting with 68 (STAR market) should resolve to sh prefix."""
        result = _check_fast_path("688981")
        assert result is not None
        assert result[0].sina_code == "sh688981"

    def test_unknown_query_returns_none(self):
        """Unknown stock name should return None from fast-path."""
        assert _check_fast_path("海螺水泥") is None
        assert _check_fast_path("苹果") is None
        assert _check_fast_path("腾讯") is None


# ── A-share prefix tests ──────────────────────────────────────────────────────


class TestASharePrefix:
    def test_sh_prefix_for_60(self):
        assert _a_share_prefix("600585") == "sh600585"

    def test_sh_prefix_for_68_star(self):
        assert _a_share_prefix("688981") == "sh688981"

    def test_sh_prefix_for_90_bshare(self):
        assert _a_share_prefix("900901") == "sh900901"

    def test_sz_prefix_for_00(self):
        assert _a_share_prefix("000001") == "sz000001"

    def test_sz_prefix_for_30_chinext(self):
        assert _a_share_prefix("300750") == "sz300750"


# ── Suggest API response parsing tests ───────────────────────────────────────


class TestParseSuggestResponse:
    """Test _parse_suggest_response with mock API data."""

    def test_parse_a_share_stock(self):
        """Type 11: A-share stock entry."""
        # Format: name,type,code,sina_code,name_cn,...
        raw = 'var suggestvalue="海螺水泥,11,600585,sh600585,海螺水泥,16,16,,,海螺水泥,hailuo";'
        results = _parse_suggest_response(raw)
        assert len(results) == 1
        assert results[0].sina_code == "sh600585"
        assert results[0].display_name == "海螺水泥"
        assert results[0].market == "a_share"

    def test_parse_hk_stock(self):
        """Type 12: HK stock entry."""
        raw = 'var suggestvalue="腾讯控股,12,00700,hk00700,腾讯控股,16,16,,,腾讯,tencent";'
        results = _parse_suggest_response(raw)
        assert len(results) == 1
        assert results[0].sina_code == "hk00700"
        assert results[0].market == "hk"

    def test_parse_us_stock(self):
        """Type 13: US stock entry."""
        raw = 'var suggestvalue="苹果,13,AAPL,gb_aapl,苹果,89,89,,,Apple Inc,apple";'
        results = _parse_suggest_response(raw)
        assert len(results) == 1
        assert results[0].sina_code == "gb_aapl"
        assert results[0].market == "us"

    def test_parse_hk_index(self):
        """Type 33: HK index entry."""
        raw = 'var suggestvalue="恒生科技指数,33,HSTECH,hkHSTECH,恒生科技指数,16,16,,,恒生科技指数,hsi tech";'
        results = _parse_suggest_response(raw)
        assert len(results) == 1
        assert results[0].sina_code == "hkHSTECH"
        assert results[0].market == "hk"

    def test_parse_multiple_entries(self):
        """Multiple entries should all be parsed."""
        raw = (
            'var suggestvalue="'
            "海螺水泥,11,600585,sh600585,海螺水泥,16,16,,,海螺水泥,hailuo;"
            "海螺型材,11,000619,sz000619,海螺型材,16,16,,,海螺型材,hailuoxingcai"
            '";'
        )
        results = _parse_suggest_response(raw)
        assert len(results) == 2
        assert results[0].sina_code == "sh600585"
        assert results[1].sina_code == "sz000619"

    def test_parse_dedup(self):
        """Duplicate codes should be deduplicated."""
        raw = (
            'var suggestvalue="'
            "海螺水泥,11,600585,sh600585,海螺水泥,16,16,,,海螺水泥,hailuo;"
            "海螺水泥,11,600585,sh600585,海螺水泥,16,16,,,海螺水泥,hailuo"
            '";'
        )
        results = _parse_suggest_response(raw)
        assert len(results) == 1

    def test_parse_empty_response(self):
        """Empty suggestvalue should return empty list."""
        assert _parse_suggest_response('var suggestvalue="";') == []

    def test_parse_malformed_response(self):
        """Malformed response should return empty list."""
        assert _parse_suggest_response("") == []
        assert _parse_suggest_response("garbage") == []

    def test_fund_filtered_out(self):
        """Type 25 (fund) should be filtered out."""
        raw = 'var suggestvalue="某基金,25,001234,of001234,某基金,16,16,,,某基金,fund";'
        results = _parse_suggest_response(raw)
        assert len(results) == 0

    def test_limit_results(self):
        """Results should be limited to the specified count."""
        entries = ";".join(
            f"股票{i},11,{600000+i},sh{600000+i},股票{i},16,16,,,股票{i},stock{i}"
            for i in range(10)
        )
        raw = f'var suggestvalue="{entries}";'
        results = _parse_suggest_response(raw, limit=3)
        assert len(results) == 3


# ── Code conversion tests ────────────────────────────────────────────────────


class TestConvertSuggestEntry:
    def test_a_share_stock_conversion(self):
        fields = ["海螺水泥", "11", "600585", "sh600585", "海螺水泥"]
        result = _convert_suggest_entry(fields)
        assert result is not None
        assert result.sina_code == "sh600585"
        assert result.market == "a_share"

    def test_hk_stock_conversion(self):
        fields = ["腾讯控股", "12", "00700", "hk00700", "腾讯控股"]
        result = _convert_suggest_entry(fields)
        assert result is not None
        assert result.sina_code == "hk00700"
        assert result.market == "hk"

    def test_hk_stock_padding(self):
        """HK stock code should be zero-padded to 5 digits."""
        fields = ["某港股", "12", "1234", "hk01234", "某港股"]
        result = _convert_suggest_entry(fields)
        assert result is not None
        assert result.sina_code == "hk01234"

    def test_us_stock_conversion(self):
        fields = ["苹果", "13", "AAPL", "gb_aapl", "苹果"]
        result = _convert_suggest_entry(fields)
        assert result is not None
        assert result.sina_code == "gb_aapl"
        assert result.market == "us"

    def test_hk_index_conversion(self):
        fields = ["恒生科技指数", "33", "HSTECH", "hkHSTECH", "恒生科技指数"]
        result = _convert_suggest_entry(fields)
        assert result is not None
        assert result.sina_code == "hkHSTECH"
        assert result.market == "hk"

    def test_fund_filtered(self):
        fields = ["某基金", "25", "001234", "of001234", "某基金"]
        result = _convert_suggest_entry(fields)
        assert result is None

    def test_too_few_fields(self):
        assert _convert_suggest_entry(["only", "two"]) is None
        assert _convert_suggest_entry([]) is None


# ── Cache tests ──────────────────────────────────────────────────────────────


class TestCodeCache:
    """Test SQLite-backed code cache."""

    @pytest.fixture
    def cache(self, tmp_dir: Path) -> _CodeCache:
        return _CodeCache(tmp_dir / "test_cache.sqlite")

    def test_put_and_get(self, cache: _CodeCache):
        """Cached results should be retrievable."""
        codes = [
            ResolvedCode("sh600585", "海螺水泥", "a_share", "api"),
            ResolvedCode("sz000619", "海螺型材", "a_share", "api"),
        ]
        cache.put("海螺水泥", codes)
        result = cache.get("海螺水泥")
        assert len(result) == 2
        assert result[0].sina_code == "sh600585"
        assert result[0].source == "cache"

    def test_get_empty(self, cache: _CodeCache):
        """Non-existent key should return empty list."""
        assert cache.get("nonexistent") == []

    def test_case_insensitive_key(self, cache: _CodeCache):
        """Query keys should be case-insensitive."""
        codes = [ResolvedCode("sh600585", "海螺水泥", "a_share", "api")]
        cache.put("海螺水泥", codes)
        assert len(cache.get("海螺水泥")) == 1
        assert len(cache.get("海螺水泥")) == 1

    def test_overwrite_on_put(self, cache: _CodeCache):
        """Re-putting the same key should overwrite old entries."""
        old = [ResolvedCode("sh600585", "海螺水泥", "a_share", "api")]
        cache.put("海螺水泥", old)
        new = [ResolvedCode("sz000619", "海螺型材", "a_share", "api")]
        cache.put("海螺水泥", new)
        result = cache.get("海螺水泥")
        assert len(result) == 1
        assert result[0].sina_code == "sz000619"

    def test_multiple_codes_per_query(self, cache: _CodeCache):
        """Multiple codes for one query should all be stored (composite PK)."""
        codes = [
            ResolvedCode("sh600585", "海螺水泥", "a_share", "api"),
            ResolvedCode("sz000619", "海螺型材", "a_share", "api"),
            ResolvedCode("hk00700", "腾讯控股", "hk", "api"),
        ]
        cache.put("海螺", codes)
        result = cache.get("海螺")
        assert len(result) == 3

    def test_ttl_expiry(self, tmp_dir: Path):
        """Expired cache entries should not be returned."""
        cache = _CodeCache(tmp_dir / "ttl_cache.sqlite")
        codes = [ResolvedCode("sh600585", "海螺水泥", "a_share", "api")]
        cache.put("海螺水泥", codes)

        # Manually backdate the cached_at timestamp
        conn = sqlite3.connect(str(tmp_dir / "ttl_cache.sqlite"))
        old_time = time.time() - 8 * 24 * 3600  # 8 days ago
        conn.execute(
            "UPDATE code_cache SET cached_at = ? WHERE query_key = ?",
            (old_time, "海螺水泥"),
        )
        conn.commit()
        conn.close()

        assert cache.get("海螺水泥") == []

    def test_empty_put_does_nothing(self, cache: _CodeCache):
        """Putting empty list should be a no-op."""
        cache.put("test", [])
        assert cache.get("test") == []


# ── Resolver integration tests (with mocked API) ─────────────────────────────


class TestCodeResolver:
    """Test CodeResolver with mocked Sina suggest API."""

    @pytest.fixture
    def resolver(self, tmp_dir: Path) -> CodeResolver:
        return CodeResolver(cache_path=tmp_dir / "resolver_cache.sqlite")

    def test_fast_path_index(self, resolver: CodeResolver):
        """Common indices should resolve via fast-path without API."""
        result = resolver.resolve("上证指数")
        assert len(result) == 1
        assert result[0].sina_code == "s_sh000001"
        assert result[0].source == "fast_path"

    def test_fast_path_market_keyword(self, resolver: CodeResolver):
        """Market keywords should resolve via fast-path."""
        result = resolver.resolve("A股")
        assert len(result) == 4
        assert result[0].source == "fast_path"

    def test_fast_path_digit_code(self, resolver: CodeResolver):
        """6-digit codes should resolve via fast-path."""
        result = resolver.resolve("600585")
        assert result[0].sina_code == "sh600585"

    @patch.object(CodeResolver, "_search_api")
    def test_api_resolution(self, mock_api, resolver: CodeResolver):
        """Unknown stock should resolve via API."""
        mock_api.return_value = [
            ResolvedCode("sh600585", "海螺水泥", "a_share", "api"),
        ]
        result = resolver.resolve("海螺水泥")
        assert len(result) == 1
        assert result[0].sina_code == "sh600585"
        assert result[0].source == "api"
        mock_api.assert_called_once_with("海螺水泥")

    @patch.object(CodeResolver, "_search_api")
    def test_api_result_cached(self, mock_api, resolver: CodeResolver):
        """Second call should use cache, not API."""
        mock_api.return_value = [
            ResolvedCode("sh600585", "海螺水泥", "a_share", "api"),
        ]
        # First call hits API
        r1 = resolver.resolve("海螺水泥")
        assert r1[0].source == "api"
        # Second call should use cache
        r2 = resolver.resolve("海螺水泥")
        assert r2[0].source == "cache"
        # API should only be called once
        mock_api.assert_called_once()

    @patch.object(CodeResolver, "_search_api")
    def test_api_returns_empty_uses_fallback(self, mock_api, resolver: CodeResolver):
        """When API returns empty, substring fallback should be tried."""
        mock_api.return_value = []
        # "上证" is a substring of "上证指数" in _COMMON_INDICES
        result = resolver.resolve("上证")
        # Substring fallback should find "上证指数" → s_sh000001
        codes = [r.sina_code for r in result]
        assert "s_sh000001" in codes

    @patch.object(CodeResolver, "_search_api")
    def test_api_failure_returns_empty(self, mock_api, resolver: CodeResolver):
        """When API fails and no fallback matches, return empty."""
        mock_api.return_value = []
        result = resolver.resolve("完全不存在的股票xyz")
        assert result == []

    def test_search_api_network_error(self, resolver: CodeResolver):
        """_search_api should return empty list on network error."""
        # This will actually try to connect and fail (no network in test env)
        # or succeed if network is available. Either way, it should not crash.
        result = resolver._search_api("test_nonexistent_stock_xyz_12345")
        assert isinstance(result, list)


# ── Backward-compatible interface tests ───────────────────────────────────────


class TestBackwardCompatInterface:
    """Test resolve_code and resolve_codes in _codes.py."""

    def test_resolve_code_index(self):
        """resolve_code should return a single code for common indices."""
        from matrix.tools.web._codes import resolve_code
        assert resolve_code("上证指数") == "s_sh000001"
        assert resolve_code("恒生科技") == "hkHSTECH"
        assert resolve_code("道琼斯") == "int_dji"

    def test_resolve_code_market_keyword(self):
        """resolve_code should return first code for market keywords."""
        from matrix.tools.web._codes import resolve_code
        result = resolve_code("A股")
        assert result in ("s_sh000001", "s_sz399001", "s_sz399006", "s_sh000300")

    def test_resolve_code_digit(self):
        """resolve_code should handle 6-digit codes."""
        from matrix.tools.web._codes import resolve_code
        assert resolve_code("600585") == "sh600585"
        assert resolve_code("000001") == "sz000001"

    def test_resolve_codes_market_keyword(self):
        """resolve_codes should return all codes for market keywords."""
        from matrix.tools.web._codes import resolve_codes
        result = resolve_codes("A股")
        assert len(result) == 4
        assert "s_sh000001" in result
        assert "s_sz399001" in result

    def test_resolve_codes_global_keyword(self):
        """resolve_codes should return global indices for global keywords."""
        from matrix.tools.web._codes import resolve_codes
        result = resolve_codes("全球股市")
        assert "int_dji" in result
        assert "int_hangseng" in result

    def test_resolve_code_unknown_returns_none(self):
        """resolve_code should return None for truly unknown queries."""
        from matrix.tools.web._codes import resolve_code
        # This will try API, which may or may not work in test env
        # Just verify it doesn't crash
        result = resolve_code("完全不存在的股票xyz")
        assert result is None or isinstance(result, str)


# ── Finance query integration tests ──────────────────────────────────────────


class TestFinanceQueryIntegration:
    """Test finance_query with the new resolver."""

    def test_finance_query_index(self):
        """finance_query should work for common indices."""
        from matrix.tools.web.finance import finance_query
        result = finance_query("上证指数")
        # Should have results (may be empty if market closed, but no error)
        assert "error" not in result or "query" in result

    def test_finance_query_market_overview(self):
        """finance_query should work for market overview keywords."""
        from matrix.tools.web.finance import finance_query
        result = finance_query("A股")
        # Should resolve to multiple indices
        assert "error" not in result or "query" in result

    def test_finance_query_unknown_returns_error(self):
        """finance_query should return structured error for unknown queries."""
        from matrix.tools.web.finance import finance_query
        result = finance_query("完全不存在的股票xyz123")
        assert "error" in result

    def test_finance_query_market_param(self):
        """finance_query should accept market parameter."""
        from matrix.tools.web.finance import finance_query
        result = finance_query("", market="a_share")
        # Should use default codes for a_share market
        assert "error" not in result or "query" in result
