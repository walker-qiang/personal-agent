"""Temporal helpers: extracting fact dates and resolving query time ranges.

A memory store that only knows ``updated_at`` cannot answer "what was I
researching three months ago". Two capabilities are needed:

- ``parse_fact_time`` — pull a concrete date out of a fact's text and store it
  in ``fact_time``, so the memory is anchored to when it was *true* rather
  than when it was written. Dated facts are exempt from decay (see
  ``SessionStore._get_decayed_memories``).
- ``resolve_time_range`` — turn "上周" / "三个月前" / "2026年8月" into a
  ``(start, end)`` window for filtering retrieval.
"""

from __future__ import annotations

import calendar
import logging
import re
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# 2026-08-16 / 2026/8/16 / 2026.08.16
_ISO_DATE = re.compile(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})")
# 2026年8月16日 / 2026年8月
_CN_DATE = re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日)?")

# Relative expressions -> (start, end) deltas in days
_RELATIVE_PATTERNS: tuple[tuple[re.Pattern[str], int, int], ...] = (
    (re.compile(r"今天|今日|当天"), 0, 0),
    (re.compile(r"昨天|昨日"), 1, 1),
    (re.compile(r"前天"), 2, 2),
    (re.compile(r"本周|这周"), 0, 7),
    (re.compile(r"上周|上个?星期"), 7, 14),
    (re.compile(r"最近\s*(\d+)\s*天"), 0, 0),  # handled specially
    (re.compile(r"(\d+)\s*天前"), 0, 0),       # handled specially
    (re.compile(r"最近|近期"), 0, 30),
    (re.compile(r"这个?月|本月"), 0, 31),
    (re.compile(r"上个?月"), 31, 62),
    (re.compile(r"(\d+)\s*个?月前"), 0, 0),    # handled specially
    (re.compile(r"今年"), 0, 365),
    (re.compile(r"去年|去年一年"), 365, 730),
)


def parse_fact_time(text: str) -> float:
    """Return the timestamp of the first concrete date found in ``text``.

    Returns 0.0 when the text carries no date, which means "not a dated fact"
    and leaves the memory subject to normal decay.
    """
    if not text:
        return 0.0
    match = _ISO_DATE.search(text) or _CN_DATE.search(text)
    if not match:
        return 0.0
    try:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3)) if match.lastindex and match.lastindex >= 3 and match.group(3) else 1
        month = min(max(month, 1), 12)
        day = min(max(day, 1), calendar.monthrange(year, month)[1])
        return datetime(year, month, day).timestamp()
    except (ValueError, TypeError, OverflowError) as exc:
        logger.debug("parse_fact_time: unparsable %r: %s", text[:40], exc)
        return 0.0


_CN_DIGITS = {
    "零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _normalize_query(query: str) -> str:
    """Turn Chinese numerals into ASCII digits so \\d+ patterns apply."""
    if not re.search(r"[一二三四五六七八九十两]", query):
        return query
    out = list(query)
    for i, char in enumerate(out):
        digit = _CN_DIGITS.get(char)
        if digit is not None:
            out[i] = str(digit)
    result = "".join(out)
    # "10天" from "十天" is already handled; "1十" would not appear.
    return result


def resolve_time_range(query: str) -> tuple[float, float]:
    """Map a query's temporal expression onto ``(start_ts, end_ts)``.

    Returns ``(0.0, 0.0)`` when the query is not time-scoped, which callers
    treat as "no filter".
    """
    if not query:
        return 0.0, 0.0
    normalized = _normalize_query(query)
    now = time.time()
    day = 86400.0

    special = re.search(r"最近\s*(\d+)\s*天", normalized)
    if special:
        days = int(special.group(1))
        return now - days * day, now

    special = re.search(r"(\d+)\s*天前", normalized)
    if special:
        days = int(special.group(1))
        return now - (days + 1) * day, now - (days - 1) * day

    special = re.search(r"(\d+)\s*个?月前", normalized)
    if special:
        months = int(special.group(1))
        end = now - months * 30 * day
        return end - 30 * day, end + 15 * day

    # A concrete date in the query scopes around that date.
    fact_ts = parse_fact_time(query)
    if fact_ts > 0:
        return fact_ts - 15 * day, fact_ts + 15 * day

    for pattern, start_days, end_days in _RELATIVE_PATTERNS:
        if start_days == 0 and end_days == 0:
            continue
        if pattern.search(normalized):
            if end_days == 0:
                end_days = start_days
            return now - end_days * day, now - start_days * day

    return 0.0, 0.0


def humanize_span(start: float, end: float) -> str:
    """Render a window for logs and UI."""
    if start <= 0 and end <= 0:
        return "all-time"
    fmt = "%Y-%m-%d"
    start_text = datetime.fromtimestamp(start).strftime(fmt) if start > 0 else "..."
    end_text = datetime.fromtimestamp(end).strftime(fmt) if end > 0 else "..."
    return f"{start_text}..{end_text}"


def age_days(updated_at: float, now: float | None = None) -> float:
    if updated_at <= 0:
        return 0.0
    return ((now or time.time()) - updated_at) / 86400.0


def within_span(ts: float, start: float, end: float) -> bool:
    if start > 0 and ts < start:
        return False
    if end > 0 and ts > end:
        return False
    return True
