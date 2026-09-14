"""Versioned, read-only evidence planning for an investment review.

The API owns identity, snapshot version and writes. This module never grants a
verification status based on model prose or merely on a successful HTTP call.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any


def snapshot_from_question(question: str) -> dict[str, Any] | None:
    marker = "<review_context>"
    if "复查协议版本：1" not in question or marker not in question:
        return None
    value, _ = json.JSONDecoder().raw_decode(question.split(marker, 1)[1])
    if not isinstance(value, dict) or value.get("version") != 1 or not value.get("revision"):
        raise ValueError("复查上下文缺少有效版本，请重新创建复查任务")
    return value


def is_exchange_fund(code: str) -> bool:
    # An unqualified six-digit product identifier is not a stock symbol.
    return bool(re.fullmatch(r"(?:sh|sz)\d{6}|hk\d{5}|[A-Za-z0-9]+\.(?:SH|SZ|HK)", code, re.I))


def review_instructions(snapshot: dict[str, Any]) -> str:
    return """
本次使用复查协议。逐项处理输入 gaps，并识别新的重大证据缺口。不要用历史结论代替当前核验。
附加输出 review_result={"context_revision":"输入 revision","checks":[
{"id":"原 gap.id，新问题使用 new- 前缀稳定编号","title":"具体事项","url":"原事项链接",
"kind":"evidence 或 special","status":"resolved/pending/not_applicable",
"finding":"核验事实及其影响；部分成立时明确剩余内容",
"citations":[{"url":"实际抓取的来源","content_hash":"工具返回的内容哈希","quote":"正文逐字摘录"}]}]}。
每个已有事项必须有结果；不能因为成功下载正文就标为 resolved。结论必须回答原问题，不能用标题代替事实。
独立证据事项不能用公司自己的主张替代；来源独立性不明时保留 pending。
无法核验、正文403、无法覆盖或需要进一步分析的事项仍为 pending。not_applicable 也需要正文证据和解释。
专项分析至少使用两个相关来源，说明交易/经营变化、资金投入、回报与风险，未完成则保留 pending。
抓取日期不是公告日期；报告期间不是披露日期。事实、公司主张与研究解释要分开。
不得恢复历史价格阈值、用未核验数据得出确定结论或把短期利润机械年化为买入依据。
即使报告仍信息不足，也保留本次有证据的发现、逐项结果和具体缺口；不得为通过校验编造内容。
"""


def document_candidates(
    snapshot: dict[str, Any] | None, sources: list[dict[str, Any]],
    select_latest: Any, year: str,
) -> list[str]:
    urls: list[str] = []
    # Existing gaps come first, including original notices outside the latest-page window.
    for gap in (snapshot or {}).get("gaps", []):
        url = str(gap.get("url") or "")
        if url.startswith(("https://", "http://")) and url not in urls:
            urls.append(url)
    remaining = list(sources)
    while remaining:
        url = select_latest(remaining, year)
        if not url:
            break
        if url not in urls:
            urls.append(url)
        remaining = [x for x in remaining if (x.get("url") or x.get("source_url")) != url]
    # Bounded calls; unattempted gaps remain pending, never silently marked covered.
    return urls[:12] if snapshot else urls[:3]


def evidence_text(evidence: list[dict[str, Any]]) -> str:
    """Trim each payload explicitly; never cut serialized JSON or drop the tail."""
    visible = copy.deepcopy(evidence)
    for entry in visible:
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        if entry.get("tool") == "personal_os.research_context":
            review = result.get("review_context")
            if isinstance(review, dict):
                body = str(review.get("latest_body") or "")
                if len(body) > 18000:
                    # Preserve the named risk/source/trigger sections near the end.
                    review["latest_body"] = body[:9000] + "\n[历史正文中段省略]\n" + body[-9000:]
                review["historical_body_partial"] = len(body) > 18000
            records = result.get("records")
            if isinstance(records, list):
                result["records"] = records[:3]
                result["history_omitted_count"] = max(0, len(records) - 3)
        elif entry.get("tool") == "personal_os.web_fetch":
            content = str(result.get("content") or "")
            if len(content) > 18000:
                result["content"] = content[:10000] + "\n[正文中段省略]\n" + content[-8000:]
                result["model_content_partial"] = True
        elif entry.get("tool") == "personal_os.financials":
            data = result.get("data")
            if isinstance(data, dict) and isinstance(data.get("reports"), list):
                reports = data["reports"]
                data["reports"] = reports[:3]
                data["history_omitted_count"] = max(0, len(reports) - 3)
    return json.dumps(visible, ensure_ascii=False, default=str)


def finalize_result(result: dict[str, Any], snapshot: dict[str, Any],
                    evidence: list[dict[str, Any]], issues: list[str]) -> dict[str, Any]:
    raw = result.get("review_result")
    raw = raw if isinstance(raw, dict) else {}
    checks = [dict(x) for x in raw.get("checks", []) if isinstance(x, dict)]
    by_id = {str(x.get("id")): x for x in checks}
    fetched = {
        str(x["result"].get("url")): x["result"] for x in evidence
        if x.get("tool") == "personal_os.web_fetch" and isinstance(x.get("result"), dict)
    }
    for gap in snapshot.get("gaps", []):
        if str(gap["id"]) not in by_id:
            check = {**gap, "status": "pending", "finding": "本次未完成核验，继续保留。", "citations": []}
            checks.append(check)
    for check in checks:
        check.setdefault("id", "new-" + hashlib.sha256(str(check.get("title", "")).encode()).hexdigest()[:16])
        citations = check.get("citations") or []
        valid = bool(citations) and len(str(check.get("finding") or "")) >= 12
        for citation in citations:
            if not isinstance(citation, dict):
                valid = False
                continue
            doc = fetched.get(str(citation.get("url")), {})
            quote = re.sub(r"\s+", "", str(citation.get("quote") or ""))
            valid = valid and (
                doc.get("source_tier") in {"official", "independent", "industry"}
                and bool(doc.get("content_hash"))
                and doc.get("content_hash") == citation.get("content_hash")
                and len(quote) >= 12
                and quote in re.sub(r"\s+", "", str(doc.get("content") or ""))
            )
        if check.get("status") in {"resolved", "not_applicable"} and not valid:
            check["status"] = "pending"
            check["finding"] = "证据引用未通过核验；" + str(check.get("finding") or "继续保留缺口。")
        if check.get("status") not in {"resolved", "not_applicable"}:
            check["status"] = "pending"
    # Quality failures are themselves durable work items, recoverable on the next run.
    if issues:
        checks = [x for x in checks if x.get("id") != "research-quality"]
        checks.append({"id": "research-quality", "title": "本次研究基础证据完整性",
                       "url": "", "kind": "quality", "status": "pending",
                       "finding": "；".join(issues), "citations": []})
    elif "research-quality" in by_id or any(x.get("id") == "research-quality" for x in checks):
        # The API independently validates the new card before clearing this item.
        checks = [x for x in checks if x.get("id") != "research-quality"]
        checks.append({"id": "research-quality", "title": "本次研究基础证据完整性",
                       "url": "", "kind": "quality", "status": "resolved",
                       "finding": "本次研究通过确定性内容与证据校验。", "citations": []})
    result["review_result"] = {"context_revision": snapshot["revision"], "checks": checks, "issues": list(issues)}
    if issues:
        result["information_completeness"] = "low"
        result["status"] = "incomplete"
    return result
