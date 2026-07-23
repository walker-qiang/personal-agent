"""Code mapping table for finance_query.

Maps Chinese/English names to Sina hq API codes with market prefixes:
- s_sh / s_sz  → A-share index (simplified, 6 fields)
- int_         → global index (4 fields)
- gb_          → US stock (lowercase ticker)
- hk           → HK stock (5-digit zero-padded)
- sh / sz      → A-share stock (full, 33 fields)
"""

from __future__ import annotations

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
    "dow": "int_dji",
    "djia": "int_dji",
    "纳斯达克": "int_nasdaq",
    "纳指": "int_nasdaq",
    "nasdaq": "int_nasdaq",
    "ndq": "int_nasdaq",
    "标普500": "int_sp500",
    "标普": "int_sp500",
    "s&p500": "int_sp500",
    "sp500": "int_sp500",
    "spx": "int_sp500",
    "日经225": "int_nikkei",
    "日经": "int_nikkei",
    "nikkei": "int_nikkei",
    "恒生指数": "int_hangseng",
    "恒指": "int_hangseng",
    "hangseng": "int_hangseng",
    "hsi": "int_hangseng",
    "韩国综合": "int_kospi",
    "kospi": "int_kospi",
    "澳洲标普": "int_as51",
    "富时100": "int_ftse",
    "ftse": "int_ftse",
    "德国dax": "int_dax",
    "dax": "int_dax",
    "法国cac": "int_cac",
    "cac": "int_cac",
}

# ---- All indices combined ----

ALL_INDICES: dict[str, str] = {**A_SHARE_INDICES, **GLOBAL_INDICES}

# ---- Market keyword groups ----

# Keywords that map to A-share market
A_SHARE_KEYWORDS = {"a股", "a 股", "沪深", "上证", "深证", "深成", "创业板", "科创板",
                    "沪指", "深指", "大盘", "沪深300", "上证50", "科创50"}

# Keywords that map to global/multi-market overview
GLOBAL_KEYWORDS = {"全球股市", "全球市场", "全球指数", "国际市场", "全球行情",
                   "海外市场", "全球主要", "全球大盘", "欧美股市", "亚太股市",
                   "global", "world", "international"}

# Keywords that map to US market
US_KEYWORDS = {"美股", "纳斯达克", "纳指", "道琼斯", "道指", "标普",
               "us stock", "us market", "wall street"}

# Keywords that map to HK market
HK_KEYWORDS = {"港股", "恒生", "恒指", "hk stock", "hk market", "hong kong"}

# ---- Well-known US stocks (common tickers) ----

US_TICKERS: dict[str, str] = {
    "苹果": "gb_aapl", "aapl": "gb_aapl", "apple": "gb_aapl",
    "特斯拉": "gb_tsla", "tsla": "gb_tsla", "tesla": "gb_tsla",
    "微软": "gb_msft", "msft": "gb_msft", "microsoft": "gb_msft",
    "谷歌": "gb_googl", "googl": "gb_googl", "google": "gb_googl",
    "亚马逊": "gb_amzn", "amzn": "gb_amzn", "amazon": "gb_amzn",
    "英伟达": "gb_nvda", "nvda": "gb_nvda", "nvidia": "gb_nvda",
    "meta": "gb_meta", "facebook": "gb_meta", "脸书": "gb_meta",
    "奈飞": "gb_nflx", "nflx": "gb_nflx", "netflix": "gb_nflx",
    "阿里巴巴": "gb_baba", "baba": "gb_baba", "alibaba": "gb_baba",
    "京东": "gb_jd", "jd": "gb_jd",
    "拼多多": "gb_pdd", "pdd": "gb_pdd",
    "百度": "gb_bidu", "bidu": "gb_bidu",
    "网易": "gb_ntes", "ntes": "gb_ntes",
    "美团": "gb_3690hk",  # HK listing
    "比亚迪": "gb_1211hk",  # HK listing
}

# ---- Well-known HK stocks ----

HK_TICKERS: dict[str, str] = {
    "腾讯": "hk00700", "腾讯控股": "hk00700",
    "阿里": "hk09988", "阿里巴巴-w": "hk09988",
    "美团": "hk03690", "美团-w": "hk03690",
    "小米": "hk01810", "小米集团": "hk01810",
    "比亚迪": "hk01211", "比亚迪股份": "hk01211",
    "京东": "hk09618", "京东集团": "hk09618",
    "快手": "hk01024", "快手-w": "hk01024",
    "百度": "hk09888", "百度集团": "hk09888",
    "网易": "hk09999", "网易-s": "hk09999",
    "友邦保险": "hk01299",
    "汇丰控股": "hk00005", "汇丰": "hk00005",
    "中国平安": "hk02318",
    "建设银行": "hk00939",
    "工商银行": "hk01398",
    "中国移动": "hk00941",
}


def resolve_code(query: str) -> str | None:
    """Try to resolve a query string to a single Sina code.

    Returns the code if exactly one match is found, else None.
    """
    q = query.strip().lower()

    # Direct index name match
    for name, code in ALL_INDICES.items():
        if q == name.lower():
            return code

    # Direct US ticker match
    for name, code in US_TICKERS.items():
        if q == name.lower():
            return code

    # Direct HK ticker match
    for name, code in HK_TICKERS.items():
        if q == name.lower():
            return code

    return None


def resolve_codes(query: str) -> list[str]:
    """Resolve a query to multiple Sina codes (for batch queries).

    Handles natural language like "全球股市" → all major indices.
    """
    q = query.strip().lower()
    codes: list[str] = []

    # Check for global market overview
    if any(kw in q for kw in GLOBAL_KEYWORDS):
        return [
            "int_dji", "int_nasdaq", "int_sp500",
            "int_hangseng", "int_nikkei",
            "s_sh000001", "s_sz399001",
        ]

    # Check for A-share market overview
    if any(kw in q for kw in ("a股", "a 股", "大盘", "沪深")):
        return ["s_sh000001", "s_sz399001", "s_sz399006", "s_sh000300"]

    # Check for US market overview
    if any(kw in q for kw in ("美股", "us stock", "us market")):
        return ["int_dji", "int_nasdaq", "int_sp500"]

    # Check for HK market overview
    if any(kw in q for kw in ("港股", "hk stock", "hk market")):
        return ["int_hangseng"]

    # Check for Asia-Pacific
    if "亚太" in q or "asia" in q:
        return ["int_hangseng", "int_nikkei", "int_kospi", "s_sh000001"]

    # Check for Europe
    if "欧洲" in q or "europe" in q or "欧股" in q:
        return ["int_ftse", "int_dax", "int_cac"]

    # Try individual name matching
    for name, code in ALL_INDICES.items():
        if name in query:
            if code not in codes:
                codes.append(code)

    for name, code in US_TICKERS.items():
        if name in query and not name.startswith("gb_"):
            if code not in codes:
                codes.append(code)

    for name, code in HK_TICKERS.items():
        if name in query:
            if code not in codes:
                codes.append(code)

    return codes
