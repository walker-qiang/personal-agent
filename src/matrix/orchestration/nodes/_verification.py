"""Anti-hallucination verification: heuristic number checks and multi-sampling.

Contains:
- _is_empty_tool_result: guard against empty tool results
- _heuristic_number_check: extract numbers from LLM answers and verify
  against tool results
- Multi-sampling verification (2+1 escalation with LRU cache), inspired
  by SelfCheckGPT (arXiv:2303.08896)

Split from _helpers.py.
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from typing import Any

logger = logging.getLogger("matrix.orchestration")


# ── Empty result guard ────────────────────────────────────────────────────────


def _is_empty_tool_result(result: Any) -> bool:
    """Check if a tool result is effectively empty (no real data).

    Accepts either a raw result value or a full tool_result entry dict
    (with "name", "arguments", "result"/"error" keys).

    Handles all common result shapes:
    - {"name": "x", "result": {"error": "..."}}  (tool_error return → always empty)
    - {"name": "x", "error": "timeout"}  (tool error → always empty)
    - {"error": "..."}  (tool_error return → always empty)
    - {"results": [], "query": "..."}  (search empty, no results found)
    - {"holdings": [], "total_balance_cents": 0}  (holdings_summary empty)
    - {"data": []}  (generic empty)
    - {}  (empty dict)
    - None / ""  (falsy)
    """
    if not result:
        return True
    if not isinstance(result, dict):
        return False  # non-dict is likely actual data (string, list of items)

    # Tool result entry: {"name": ..., "arguments": ..., "result": ...} or {"name": ..., "error": ...}
    # If this looks like a tool_result entry (has "name" and "result"/"error"), unwrap it.
    if "name" in result and ("result" in result or "error" in result):
        if result.get("error"):
            return True  # tool error → always empty
        return _is_empty_tool_result(result.get("result"))

    # Plain result dict
    if result.get("error"):
        return True
    # Check if ALL list fields are empty and all non-list fields are non-data
    has_data = False
    for key, value in result.items():
        if key == "error":
            continue
        if isinstance(value, list):
            if len(value) > 0:
                has_data = True
        elif isinstance(value, dict):
            if value:  # non-empty dict
                has_data = True
        elif isinstance(value, (int, float)):
            if value != 0:  # non-zero number is data
                has_data = True
        elif value:  # truthy string, etc.
            has_data = True
    return not has_data


# ── Heuristic number check ───────────────────────────────────────────────────


def _heuristic_number_check(
    answer: str,
    tool_results: list[dict[str, Any]],
    llm,
    question: str,
) -> str:
    """Heuristic check: verify that key numbers in the answer appear in tool results.

    Extracts significant numbers (prices, amounts, percentages with context) from
    the LLM's answer and checks if they appear in the flattened tool results.
    If a substantial portion of numbers cannot be found in the tool results,
    the answer is likely fabricated and we add a warning or trigger verification.

    Returns the (possibly modified) answer string.
    """
    # Flatten tool results into a single searchable text
    result_texts: list[str] = []
    for tr in tool_results:
        name = tr.get("name", "")
        data = tr.get("result", tr.get("error", ""))
        if isinstance(data, (dict, list)):
            data = json.dumps(data, ensure_ascii=False)
        result_texts.append(str(data))
    flat_results = " ".join(result_texts)

    # Extract "significant" numbers from the answer — numbers that look like
    # data points rather than formatting artifacts:
    # - Numbers with Chinese units: 亿, 万, 千, 百, 元, 点, %, 倍
    # - Numbers with currency symbols: ¥, $, HK$, US$
    # - Decimal numbers that look like prices/rates (e.g., 3.14, 0.05)
    # - Percentages: 1.5%, -2.3%
    patterns = [
        # Chinese unit patterns
        re.compile(r"(\d+(?:\.\d+)?)\s*(亿|万|千|百|元|港元|美元|点|％|%|倍)"),
        # Currency patterns
        re.compile(r"[¥$￥]\s*(\d+(?:\.\d+)?)"),
        re.compile(r"HK\$\s*(\d+(?:\.\d+)?)"),
        re.compile(r"US\$\s*(\d+(?:\.\d+)?)"),
        # Percentage patterns (standalone)
        re.compile(r"(\d+(?:\.\d+)?)\s*[%％]"),
        # Price-like decimals (1-5 digit integer part with 1-2 decimals, e.g., 17.74, 380.5, 3400.12)
        re.compile(r"(?<!\d)(\d{1,5}\.\d{1,2})(?!\d)"),
        # Large integers (>= 1000, likely data points)
        re.compile(r"(?<!\d)(\d{4,})(?!\d)"),
    ]

    extracted_numbers: set[str] = set()
    for pat in patterns:
        for m in pat.finditer(answer):
            num_text = m.group(0).strip()
            # Skip numbers that look like dates (e.g., 2024, 2025, 2026)
            if re.match(r"^\d{4}$", num_text) and 2000 <= int(num_text) <= 2100:
                continue
            extracted_numbers.add(num_text)

    if not extracted_numbers:
        return answer  # No numbers to check, answer is likely qualitative

    # Check each number against tool results using multi-pass matching:
    # Pass 1: Exact string comparison (with units/symbols intact).
    # Pass 2: Float comparison — strip units from answer numbers and compare
    #         numeric values as floats. This handles:
    #         - Trailing zeros: 17.80 (answer) == 17.8 (tool JSON)
    #         - Sign differences: 0.34 (answer "跌幅 0.34%") == -0.34 (tool)
    # Pass 3: Chinese unit conversion — if answer says "5.33亿", convert to
    #         533000000.0 and check against tool floats.
    _tool_nums: set[str] = set()
    _tool_floats: set[float] = set()
    _tool_abs_floats: set[float] = set()
    _bare_num_re = re.compile(r"(\d+(?:\.\d+)?)")
    for pat in patterns:
        for m in pat.finditer(flat_results):
            raw = m.group(0).strip()
            if re.match(r"^\d{4}$", raw) and 2000 <= int(raw) <= 2100:
                continue
            _tool_nums.add(raw)
            for bn in _bare_num_re.finditer(raw):
                try:
                    f = float(bn.group(1))
                    _tool_floats.add(f)
                    _tool_abs_floats.add(abs(f))
                except ValueError:
                    pass

    _UNIT_MULTIPLIERS = {"亿": 1e8, "万": 1e4, "千": 1e3, "百": 1e2}

    missing: list[str] = []
    found: list[str] = []
    for num in extracted_numbers:
        # Pass 1: Exact string match against tool result numbers (including units)
        if num in _tool_nums:
            found.append(num)
            continue
        # Pass 2: Float comparison (handles trailing zeros and sign differences)
        bare_match = _bare_num_re.search(num)
        if bare_match:
            try:
                bare_float = float(bare_match.group(1))
                if bare_float in _tool_floats or abs(bare_float) in _tool_abs_floats:
                    found.append(num)
                    continue
                # Pass 3: Chinese unit conversion (e.g., "5.33亿" → 533000000.0)
                for unit, mult in _UNIT_MULTIPLIERS.items():
                    if unit in num:
                        converted = bare_float * mult
                        if converted in _tool_floats or abs(converted) in _tool_abs_floats:
                            found.append(num)
                            break
                else:
                    missing.append(num)
                    continue
                found.append(num)
                continue
            except ValueError:
                pass
        missing.append(num)

    total = len(extracted_numbers)
    missing_ratio = len(missing) / total if total > 0 else 0

    logger.info(
        "heuristic_number_check: total=%d found=%d missing=%d ratio=%.2f missing=%s",
        total, len(found), len(missing), missing_ratio,
        json.dumps(missing[:5], ensure_ascii=False),
    )

    # ── Tightened thresholds: 25% for warning, 50% for strong ──
    # If >= 50% of numbers are missing → high fabrication risk
    if missing_ratio >= 0.5 and total >= 2:
        logger.warning(
            "heuristic_number_check: high fabrication risk — %d/%d numbers not in tool results",
            len(missing), total,
        )
        # Try multi-sampling LLM verification as a second pass
        if llm is not None:
            try:
                verified = _multi_sample_verify(answer, missing, flat_results, llm, question)
                if verified:
                    return verified
            except Exception:
                logger.exception("heuristic_number_check: multi-sample verification failed")

        # Fallback: add a strong warning
        missing_preview = "、".join(missing[:5])
        warning = (
            f"\n\n> ⚠️ **数据一致性警告**：以上回答中的关键数据（{missing_preview}等）"
            f"未在工具搜索结果中找到对应来源，可能不准确。建议核实后参考。"
        )
        return answer + warning

    # If >= 25% of numbers are missing, add a lighter warning
    if missing_ratio >= 0.25 and total >= 3:
        missing_preview = "、".join(missing[:3])
        warning = (
            f"\n\n> ⚠️ 部分数据（{missing_preview}）未在搜索结果中确认，请谨慎参考。"
        )
        return answer + warning

    return answer


# ── Multi-sampling verification cache ──────────────────────────────────────
# Caches "verdict" for (number, source_hash) pairs to avoid re-verifying
# the same disputed numbers across invocations.  An LRU of 128 entries
# keeps memory bounded while hitting the common case (same stock queried
# multiple times in a session).

_VERIFY_CACHE: OrderedDict[str, str] = OrderedDict()
_VERIFY_CACHE_MAX = 128


def _cache_key(missing_numbers: list[str], tool_results_text: str) -> str:
    """Build a stable cache key from the disputed numbers + tool result hash."""
    import hashlib as _hashlib
    nums_str = ",".join(sorted(missing_numbers[:5]))
    source_hash = _hashlib.md5(
        tool_results_text[:2000].encode("utf-8")
    ).hexdigest()[:12]
    return f"{nums_str}|{source_hash}"


def _cached_verdict(key: str) -> str | None:
    """Get a cached verdict, or None if not cached."""
    if key in _VERIFY_CACHE:
        _VERIFY_CACHE.move_to_end(key)
        return _VERIFY_CACHE[key]
    return None


def _cache_verdict(key: str, verdict: str) -> None:
    """Store a verdict in the cache, evicting the oldest if at capacity."""
    _VERIFY_CACHE[key] = verdict
    _VERIFY_CACHE.move_to_end(key)
    if len(_VERIFY_CACHE) > _VERIFY_CACHE_MAX:
        _VERIFY_CACHE.popitem(last=False)


def _single_verify(
    answer: str,
    missing_numbers: list[str],
    tool_results_text: str,
    llm,
    question: str,
    temperature: float,
) -> str:
    """Single short-prompt verification call.

    Returns one of: "SUPPORTED", "PARTIAL", "FABRICATED", "INCONCLUSIVE".
    Uses a *short* prompt that only sends disputed numbers + source snippet,
    rather than the full answer, to minimize token consumption.
    """
    # Short prompt: only disputed numbers + source context (not the full answer)
    # This reduces token usage by ~10x compared to sending the full answer.
    nums_list = ", ".join(missing_numbers[:5])
    source_snippet = tool_results_text[:1500]

    verify_prompt = (
        "你是事实核查员。判断以下数字是否出现在给定的工具结果中。\n\n"
        f"待核查数字：{nums_list}\n\n"
        f"工具结果（截取）：\n{source_snippet}\n\n"
        "返回 JSON：\n"
        '{"verdict": "SUPPORTED|PARTIAL|FABRICATED", "reason": "简短原因"}\n\n'
        "- SUPPORTED: 所有数字都能在工具结果中找到对应值\n"
        "- PARTIAL: 部分数字能找到，部分找不到（考虑单位转换、小数精度差异）\n"
        "- FABRICATED: 数字完全不在工具结果中"
    )

    try:
        data = llm.complete_json(
            verify_prompt,
            [{"role": "user", "content": "请核查以上数字的事实准确性。"}],
            temperature=temperature,
        )
        if not isinstance(data, dict):
            return "INCONCLUSIVE"
        verdict = str(data.get("verdict", "")).upper().strip()
        if verdict not in ("SUPPORTED", "PARTIAL", "FABRICATED"):
            return "INCONCLUSIVE"
        return verdict
    except Exception:
        return "INCONCLUSIVE"


def _multi_sample_verify(
    answer: str,
    missing_numbers: list[str],
    tool_results_text: str,
    llm,
    question: str,
) -> str | None:
    """Multi-sampling verification with 2+1 escalation and caching.

    Strategy (inspired by SelfCheckGPT, arXiv:2303.08896):
    1. Check cache — if this number set was verified before, reuse the verdict.
    2. Sample 2 times at temperature=0.3. If both agree, accept the verdict.
    3. If 2 samples disagree, sample a 3rd time (temperature=0.0) as tiebreaker.
    4. If FABRICATED, fall back to a full-answer verification to get
       a corrected answer (only when fabrication is confirmed by majority).

    Returns:
        - The original `answer` if numbers are SUPPORTED or PARTIAL.
        - A corrected answer string if FABRICATED.
        - None if verification is inconclusive.
    """
    # 1. Cache lookup
    key = _cache_key(missing_numbers, tool_results_text)
    cached = _cached_verdict(key)
    if cached:
        logger.info("heuristic_number_check: cache hit, verdict=%s", cached)
        if cached == "FABRICATED":
            # Even with cached FABRICATED, we need to generate a correction.
            # Fall through to full-answer verification below.
            pass
        else:
            return answer  # SUPPORTED or PARTIAL — return original

    # 2. First two samples (temperature > 0 for diversity)
    v1 = _single_verify(answer, missing_numbers, tool_results_text, llm, question, 0.3)
    v2 = _single_verify(answer, missing_numbers, tool_results_text, llm, question, 0.3)
    logger.info(
        "heuristic_number_check: multi-sample v1=%s v2=%s (missing=%s)",
        v1, v2, missing_numbers[:3],
    )

    # 3. Escalation: if first two agree, use that verdict
    if v1 == v2 and v1 != "INCONCLUSIVE":
        verdict = v1
    else:
        # 4. Tiebreaker: 3rd sample at temperature=0.0
        v3 = _single_verify(answer, missing_numbers, tool_results_text, llm, question, 0.0)
        logger.info(
            "heuristic_number_check: tiebreaker v3=%s (v1=%s v2=%s)",
            v3, v1, v2,
        )
        # Majority vote among the 3 samples
        votes = [v for v in (v1, v2, v3) if v != "INCONCLUSIVE"]
        if not votes:
            return None  # All 3 inconclusive — can't verify
        # Pick the most common verdict
        from collections import Counter as _Counter
        verdict = _Counter(votes).most_common(1)[0][0]

    # Cache the verdict
    _cache_verdict(key, verdict)

    # 5. Handle verdict
    if verdict in ("SUPPORTED", "PARTIAL"):
        logger.info(
            "heuristic_number_check: multi-sample verdict=%s, keeping original answer",
            verdict,
        )
        return answer

    if verdict == "FABRICATED":
        # Full-answer verification to get a corrected answer
        # (only when fabrication is confirmed by majority vote)
        logger.warning(
            "heuristic_number_check: multi-sample confirmed fabrication, "
            "generating corrected answer",
        )
        return _full_verify_and_correct(
            answer, missing_numbers, tool_results_text, llm, question,
        )

    return None


def _full_verify_and_correct(
    answer: str,
    missing_numbers: list[str],
    tool_results_text: str,
    llm,
    question: str,
) -> str | None:
    """Full-answer verification: send the complete answer + tool results to get
    a corrected answer when multi-sampling confirms fabrication.

    This is the expensive path — only called when 2+ samples agree on FABRICATED.
    """
    full_prompt = (
        f"你是事实核查员。以下 AI 回答中的部分数字可能不存在于工具搜索结果中。\n\n"
        f"用户问题：{question}\n\n"
        f"AI 回答：\n{answer[:1500]}\n\n"
        f"工具搜索结果（截取）：\n{tool_results_text[:3000]}\n\n"
        f"可疑数字：{', '.join(missing_numbers[:5])}\n\n"
        "请基于工具搜索结果生成修正后的回答。"
        "只包含工具结果中确实存在的信息，对于无法获取的数据请诚实说明。\n\n"
        "返回 JSON：\n"
        '{"verdict": "FABRICATED", "reason": "简短原因", "corrected_answer": "修正后的回答"}'
    )

    try:
        data = llm.complete_json(
            full_prompt,
            [{"role": "user", "content": "请生成修正后的回答。"}],
            temperature=0.0,
        )
        if not isinstance(data, dict):
            return None
        if data.get("corrected_answer"):
            logger.info("heuristic_number_check: corrected answer generated")
            return str(data["corrected_answer"])
        return None
    except Exception:
        return None
