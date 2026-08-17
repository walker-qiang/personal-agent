"""Chat orchestration: Commander + Domain Agents multi-agent flow."""

from __future__ import annotations

import json
from dataclasses import replace
import logging
import queue
import sqlite3
import threading
import time
import traceback
from urllib.parse import unquote
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from ..agent import AgentRegistry, resolve_agent_policy
from ..agent.commander import COMMANDER
from ..agent.domain_agents import CODING_ASSISTANT, INVESTMENT_ANALYST, KNOWLEDGE_MANAGER, MEDIA_GENERATOR
from ..config import AgentConfig, IMAGE_MODELS, KNOWN_MODELS, VIDEO_MODELS, default_model
from ..llm import LLMClient, LLMError, build_llm_client
from ..llm.http import set_rate_limiter
from ..orchestration.anti_hallucination import _strip_all_verification_tags
from ..orchestration import build_graph
from ..orchestration.runtime_adapter import build_multimodal_content
from ..orchestration.state import AgentState
from ..orchestration.nodes._helpers import CircuitBreaker, _today_cn
from ..rate_limiter import TokenBucketRateLimiter
from ..store import SessionStore
from ..tools import FinanceToolError, ToolRegistry, ToolDefinition
from ..runtime.adapters.sqlite_store import SQLiteRuntimeStore
from ..runtime.adapters.external_agent import ExternalAgentAdapter
from ..runtime.adapters.deep_research import DeepResearchWorkflow
from ..runtime import AgentRuntime, ExecutionPolicy, ResumeInput
from ..runtime.adapters.model import MatrixModelAdapter
from ..runtime.adapters.tools import MatrixToolAdapter
from ..context import ToolResultRefStore, make_get_stored_data_tool
from ..memory import EvolutionConfig, MemoryEvolution
from ..memory.lesson_store import LessonStore
from ._utils import(
    MEMORY_EXTRACTION_PROMPT,
    _drain_queue,
    preview_json,
    result_count,
    timestamp,
)


class TraceSink(Protocol):
    def record(self, event: dict[str, Any]) -> None:
        ...


logger = logging.getLogger("matrix.chat")


def _normalize_research_result(
    result: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    code: str,
    name: str,
    object_type: str,
    research_date: str,
) -> dict[str, Any]:
    """Make the model response persistence-ready without inventing facts."""
    normalized = dict(result)
    normalized.setdefault("schema_version", 2)
    normalized.setdefault("type", "investment-research")
    normalized.setdefault("object_type", object_type)
    normalized.setdefault("research_date", research_date)
    normalized.setdefault("data_date", research_date)
    normalized.setdefault("subject", {"code": code, "name": name})
    normalized.setdefault("tags", ["investment-research", object_type])

    subject = normalized.get("subject")
    if not isinstance(subject, dict):
        normalized["subject"] = {"code": code, "name": name}
    else:
        subject.setdefault("code", code)
        subject.setdefault("name", name)

    metrics = normalized.get("metrics")
    if isinstance(metrics, dict):
        normalized["metrics"] = [
            {"name": key, "value": value}
            for key, value in metrics.items()
        ]
    for field in ("highlights", "thesis", "antithesis", "risks", "triggers", "tags"):
        value = normalized.get(field)
        if isinstance(value, str) and value.strip():
            normalized[field] = [value.strip()]
        elif isinstance(value, dict):
            normalized[field] = [
                item for item in value.values()
                if isinstance(item, (str, dict)) and str(item).strip()
            ]

    existing_sources = normalized.get("sources")
    sources = []
    if isinstance(existing_sources, list):
        for item in existing_sources:
            if not isinstance(item, dict):
                continue
            source = dict(item)
            source["date"] = _clean_source_date(source.get("date"))
            sources.append(source)
    seen = {
        str(item.get("url") or item.get("title") or "").strip().lower()
        for item in sources
        if isinstance(item, dict)
    }
    announcement_items: list[dict[str, Any]] = []
    info_items: list[dict[str, Any]] = []
    for entry in evidence:
        if entry.get("tool") not in {
            "personal_os.announcements", "personal_os.information_search",
        }:
            continue
        payload = entry.get("result")
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            target = announcement_items if entry.get("tool") == "personal_os.announcements" else info_items
            target.extend(item for item in payload["items"] if isinstance(item, dict))

    for item in announcement_items + info_items:
        title = str(item.get("title") or item.get("source_name") or "").strip()
        url = str(item.get("url") or "").strip()
        key = (url or title).lower()
        if not key or key in seen:
            continue
        source_type = str(
            item.get("source_type")
            or ("official" if item.get("tier") == "official" else "supplementary")
        )
        date = _clean_source_date(
            item.get("date")
            or item.get("published_at")
            or item.get("publishedAt")
        )
        sources.append({
            "title": title or url,
            "url": url,
            "date": date,
            "source_type": source_type,
        })
        seen.add(key)

    for entry in evidence:
        if entry.get("tool") != "personal_os.web_fetch":
            continue
        payload = entry.get("result")
        if not isinstance(payload, dict) or not payload.get("url"):
            continue
        url = str(payload.get("url") or "").strip()
        if not url or url.lower() in seen:
            continue
        title = str(payload.get("source_name") or url).strip()
        source = {
            "title": title,
            "url": url,
            "date": _clean_source_date(payload.get("published_at")),
            "source_type": "web_fetch",
            "verification_status": payload.get("verification_status") or "fetched",
        }
        report_period = _clean_source_date(payload.get("report_period"))
        if report_period:
            source["report_period"] = report_period
        sources.append(source)
        seen.add(url.lower())

    if not sources:
        sources = [
            {
                "title": entry["tool"],
                "url": "",
                "date": "",
                "source_type": "tool",
            }
            for entry in evidence
            if entry.get("tool", "").startswith("personal_os.")
        ]
    normalized["sources"] = sources
    normalized["tool_availability"] = _tool_availability(evidence)
    authoritative_period = _latest_report_period_from_evidence(evidence, research_date)
    if authoritative_period:
        normalized["latest_report_period"] = authoritative_period
    _enforce_research_quality(normalized, evidence)
    return normalized


def _clean_source_date(value: Any) -> str:
    """Keep real source dates only; never persist Go's zero-time placeholder."""
    text = str(value or "").strip()
    if not text or text.startswith("0001-01-01"):
        return ""
    return text


def _tool_availability(evidence: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Expose whether a tool returned data so synthesis cannot call it missing."""
    availability: dict[str, dict[str, Any]] = {}
    for entry in evidence:
        tool = str(entry.get("tool") or "").strip()
        if not tool:
            continue
        result = entry.get("result")
        error = entry.get("error")
        available = not error and not (
            isinstance(result, dict) and result.get("error")
        )
        item: dict[str, Any] = {"status": "available" if available else "unavailable"}
        if not available:
            item["error"] = str(error or (result.get("error") if isinstance(result, dict) else "tool unavailable"))
        elif isinstance(result, dict):
            item["has_data"] = bool(
                result.get("data")
                or result.get("items")
                or result.get("companies")
                or result.get("content")
                or result.get("text")
                or result.get("price")
                or result.get("name")
            )
        availability[tool] = item
    return availability


def _safe_evidence_report_period(value: Any, research_date: str) -> str:
    text = str(value or "").strip()[:10]
    try:
        import datetime as _datetime

        period = _datetime.date.fromisoformat(text)
        upper = _datetime.date.fromisoformat(str(research_date)[:10])
    except (TypeError, ValueError):
        return ""
    if period.year < 2000 or period > upper:
        return ""
    return period.isoformat()


def _latest_report_period_from_evidence(
    evidence: list[dict[str, Any]], research_date: str,
) -> str:
    periods: list[str] = []
    for entry in evidence:
        tool = entry.get("tool")
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        if tool == "personal_os.financials":
            data = result.get("data")
            if isinstance(data, dict):
                metadata = data.get("metadata")
                if isinstance(metadata, dict):
                    period = _safe_evidence_report_period(
                        metadata.get("latest_report_period"), research_date,
                    )
                    if period:
                        periods.append(period)
                reports = data.get("reports")
                if isinstance(reports, list):
                    for report in reports:
                        if isinstance(report, dict):
                            period = _safe_evidence_report_period(
                                report.get("period_end"), research_date,
                            )
                            if period:
                                periods.append(period)
        elif tool == "personal_os.web_fetch":
            if result.get("verification_status") == "verified":
                period = _safe_evidence_report_period(result.get("report_period"), research_date)
                if period:
                    periods.append(period)
        elif tool == "personal_os.research_context":
            observed = result.get("observed_facts")
            records = observed.get("reports", []) if isinstance(observed, dict) else []
            if isinstance(records, list):
                for report in records:
                    if isinstance(report, dict):
                        period = _safe_evidence_report_period(
                            report.get("period_end"), research_date,
                        )
                        if period:
                            periods.append(period)
    return max(periods) if periods else ""


def _enforce_research_quality(
    normalized: dict[str, Any], evidence: list[dict[str, Any]]
) -> None:
    """Apply deterministic safety gates after model synthesis."""
    results = {
        str(entry.get("tool")): entry.get("result")
        for entry in evidence
        if entry.get("tool")
    }
    blockers: list[str] = []
    quote = results.get("personal_os.market_quote")
    if not isinstance(quote, dict) or quote.get("price") in (None, ""):
        blockers.append("行情缺少可靠价格")
    elif not quote.get("datetime"):
        blockers.append("行情缺少数据时间")

    financials = results.get("personal_os.financials")
    financial_data = financials.get("data") if isinstance(financials, dict) else None
    financial_data = financial_data if isinstance(financial_data, dict) else {}
    reports = financial_data.get("reports")
    metadata = financial_data.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    structured_report_period = str(
        metadata.get("latest_report_period")
        or (
            reports[0].get("period_end")
            if isinstance(reports, list)
            and reports
            and isinstance(reports[0], dict)
            else ""
        )
        or ""
    )
    if not isinstance(reports, list) or not reports:
        blockers.append("没有可用的多期财务报告")
    if not metadata.get("latest_report_period") and not reports:
        blockers.append("无法确认最新财务报告期")
    if isinstance(reports, list) and reports:
        latest_report = reports[0] if isinstance(reports[0], dict) else {}
        if not latest_report.get("currency"):
            blockers.append("财务报告缺少币种")
        if not latest_report.get("unit"):
            blockers.append("财务报告缺少金额单位")

    verified_official = any(
        isinstance(entry.get("result"), dict)
        and entry["result"].get("verification_status") == "verified"
        and entry["result"].get("source_tier") == "official"
        for entry in evidence
        if entry.get("tool") == "personal_os.web_fetch"
    )
    reconciled_financials = (
        metadata.get("source_verification") == "reconciled"
        and metadata.get("provenance_ready") is True
    )
    official_evidence_ready = verified_official or reconciled_financials
    verified_official_periods = [
        str(entry["result"].get("report_period") or "")
        for entry in evidence
        if entry.get("tool") == "personal_os.web_fetch"
        and isinstance(entry.get("result"), dict)
        and entry["result"].get("verification_status") == "verified"
        and entry["result"].get("source_tier") == "official"
        and entry["result"].get("report_period")
    ]
    latest_official_period = max(verified_official_periods, default="")
    stale_structured_data = metadata.get("stale") is True
    stale_covered_by_official = bool(
        stale_structured_data
        and official_evidence_ready
        and latest_official_period
        and (
            not structured_report_period
            or latest_official_period > structured_report_period
        )
    )
    if stale_structured_data and not stale_covered_by_official:
        blockers.append("最新财务报告已过期")
    if normalized.get("object_type", "stock") == "stock" and not official_evidence_ready:
        blockers.append("没有通过正文核验的官方公告或财报")

    quality = normalized.get("data_quality")
    if not isinstance(quality, dict):
        quality = {}
    quality["status"] = "blocked" if blockers else "pass"
    quality["blockers"] = blockers
    warnings = quality.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    if not official_evidence_ready:
        warning = "尚无通过正文报告期核验的官方文档"
        if warning not in warnings:
            warnings.append(warning)
    if stale_covered_by_official:
        warning = (
            f"结构化财报缓存截至 {structured_report_period or '未知'}；"
            f"已用官方正文核验 {latest_official_period} 补充"
        )
        if warning not in warnings:
            warnings.append(warning)
    elif reconciled_financials and not verified_official:
        warning = "财务事实已完成官方财报对账；本次证据未单独展开 web_fetch 正文"
        if warning not in warnings:
            warnings.append(warning)
    quality["warnings"] = warnings
    normalized["data_quality"] = quality

    if not blockers:
        return
    normalized["status"] = "incomplete"
    normalized["information_completeness"] = "low"
    decision = normalized.get("decision")
    if not isinstance(decision, dict):
        decision = {}
    decision["quality"] = "weak"
    decision["action"] = "research before action"
    decision["confidence"] = "low"
    normalized["decision"] = decision
    risks = normalized.get("risks")
    if not isinstance(risks, list):
        risks = []
    for blocker in blockers:
        message = f"数据质量闸门：{blocker}。"
        if message not in risks:
            risks.append(message)
    normalized["risks"] = risks


def _research_result_issues(
    result: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    research_date: str,
) -> list[str]:
    """Return deterministic persistence blockers for a synthesized card.

    The model may produce valid JSON while still returning an unusable research
    card.  Keep this gate independent from the prompt so callers cannot bypass
    date, evidence, or structural checks by changing wording.
    """
    issues: list[str] = []
    required = (
        "schema_version", "type", "subject", "research_date", "data_date",
        "latest_report_period", "summary", "decision",
    )
    for field in required:
        if result.get(field) in (None, "", []):
            issues.append(f"缺少 {field}")
    if result.get("schema_version") != 2:
        issues.append("schema_version 必须为 2")
    if result.get("type") != "investment-research":
        issues.append("type 必须为 investment-research")
    if not isinstance(result.get("subject"), dict):
        issues.append("subject 必须是对象")
    if not isinstance(result.get("decision"), dict):
        issues.append("decision 必须是对象")

    def parse_date(value: Any) -> Any:
        import datetime as _datetime

        try:
            return _datetime.date.fromisoformat(str(value or "")[:10])
        except (TypeError, ValueError):
            return None

    upper = parse_date(research_date)
    if upper is None:
        issues.append("研究日期无效")
    else:
        for field in ("research_date", "data_date", "latest_report_period"):
            parsed = parse_date(result.get(field))
            if parsed is None:
                issues.append(f"{field} 不是有效日期")
            elif parsed > upper:
                issues.append(f"{field} 不能晚于研究日期 {research_date}")

    authoritative = _latest_report_period_from_evidence(evidence, research_date)
    if authoritative:
        current = parse_date(result.get("latest_report_period"))
        if current is None or current > parse_date(authoritative):
            # Evidence is authoritative; repair the model's future/stale value
            # before the validation result is sent to the client or persisted.
            result["latest_report_period"] = authoritative
            for metric in result.get("metrics", []) if isinstance(result.get("metrics"), list) else []:
                if not isinstance(metric, dict):
                    continue
                name = str(metric.get("name") or "").lower()
                if "报告期" in name or "report period" in name:
                    metric["value"] = authoritative
            issues = [item for item in issues if "latest_report_period" not in item]

    for field in ("highlights", "thesis", "antithesis", "risks", "metrics", "triggers"):
        value = result.get(field)
        if not isinstance(value, list) or not value:
            issues.append(f"{field} 不能为空且必须是数组")

    metrics = result.get("metrics")
    if isinstance(metrics, list):
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                issues.append(f"metrics[{index}] 必须是对象")
                continue
            if not str(metric.get("name") or "").strip():
                issues.append(f"metrics[{index}] 缺少 name")
            if metric.get("value") in (None, ""):
                issues.append(f"metrics[{index}] 缺少 value")

    sources = result.get("sources")
    if not isinstance(sources, list) or not sources:
        issues.append("sources 不能为空且必须是数组")
    else:
        usable_sources = [
            item for item in sources
            if isinstance(item, dict) and str(item.get("url") or "").startswith(("http://", "https://"))
        ]
        if not usable_sources:
            issues.append("sources 至少需要一个真实 URL")
        official_sources = [
            item for item in usable_sources
            if str(item.get("source_type") or "").lower() in {"official", "web_fetch", "cninfo", "hkexnews"}
        ]
        if not official_sources and not authoritative:
            issues.append("缺少官方来源或正文核验")

    quality = result.get("data_quality")
    if isinstance(quality, dict) and quality.get("status") == "blocked":
        blockers = quality.get("blockers") or []
        issues.append("数据质量闸门未通过：" + "、".join(str(item) for item in blockers))
    if str(result.get("information_completeness") or "").lower() == "low":
        issues.append("信息完整度为 low")
    return list(dict.fromkeys(issues))


def _select_latest_official_url(
    sources: list[dict[str, Any]],
    research_year: str,
) -> str | None:
    """Prefer the newest official filing, using its reporting period.

    Search result titles are often generic (for example ``PDF Tencent``), while
    the actual report period is present in an encoded PDF filename or in an
    explicit ``report_period`` field.  A keyword score can therefore select an
    older annual report over a newer interim result.  Period-first ordering is
    deterministic and keeps the source selector independent of title quality.
    """
    candidates = [
        item for item in sources
        if isinstance(item, dict)
        and item.get("url")
        and (item.get("tier") == "official" or item.get("source_level") == "official")
    ]
    if not candidates:
        return None

    def sort_key(item: dict[str, Any]) -> tuple[str, int, int]:
        explicit = str(item.get("report_period") or "").strip()
        period = explicit or _infer_report_period_from_text(
            " ".join(str(item.get(key) or "") for key in ("title", "url", "summary"))
        )
        text = unquote(
            " ".join(str(item.get(key) or "") for key in ("title", "url", "summary"))
        ).lower()
        is_document = int(
            ".pdf" in text
            or any(token in text for token in ("annual report", "interim", "业绩", "季报", "中期"))
        )
        verified = int(str(item.get("verification_status") or "") == "verified")
        # Keep the requested year as a tie-breaker only; it must not outrank a
        # newer report whose title happens to be generic.
        year_hint = int(bool(research_year and research_year in text))
        return (period or "0000-00-00", verified + year_hint, is_document)

    return max(candidates, key=sort_key).get("url")


def _extract_official_report_period(evidence: list[dict[str, Any]]) -> str:
    """Extract an explicit reporting date from an official fetched document."""
    for entry in evidence:
        if entry.get("tool") != "personal_os.web_fetch":
            continue
        result = entry.get("result")
        if isinstance(result, dict):
            content = str(result.get("content") or result.get("text") or "")
            period = _infer_report_period_from_text(content)
            if period:
                return period
            period = _infer_report_period_from_text(str(result.get("url") or ""))
            if period:
                return period
    return ""


def _infer_report_period_from_text(value: str) -> str:
    """Infer a report period only from explicit report/date language."""
    import re

    text = unquote(str(value or "")).replace("\\u0026", "&")
    if not text:
        return ""

    arabic = re.search(
        r"(?:截至|截至于|as of|as at|ended|ending|for the)\s*"
        r"(\d{4})\s*(?:年|[-/.])\s*(\d{1,2})\s*(?:月|[-/.])\s*(\d{1,2})\s*日?",
        text,
        re.I,
    )
    if arabic:
        return _valid_report_period(*arabic.groups())

    chinese = re.search(
        r"(?:截至|截至于|止|报告期(?:末|为)?|本报告期)?\s*"
        r"([零〇一二三四五六七八九]{4})年\s*"
        r"([零〇一二三四五六七八九十百千万两]+)月\s*"
        r"([零〇一二三四五六七八九十百千万两]+)日",
        text,
    )
    if chinese:
        return _valid_report_period(
            str(_parse_chinese_number(chinese.group(1))),
            str(_parse_chinese_number(chinese.group(2))),
            str(_parse_chinese_number(chinese.group(3))),
        )

    year_match = re.search(r"(20\d{2})\s*(?:年|[-/.])", text)
    if year_match:
        year = year_match.group(1)
    else:
        chinese_year = re.search(r"([零〇一二三四五六七八九]{4})年", text)
        if not chinese_year:
            return ""
        year = str(_parse_chinese_number(chinese_year.group(1)))

    lower = text.lower()
    if any(token in lower for token in ("annual report", "annual results", "年度报告", "年报")):
        return _valid_report_period(year, "12", "31")
    if any(token in lower for token in (
        "interim report", "interim results", "中期报告", "中期业绩", "半年报",
        "半年度报告", "上半年", "1h", "six months", "q2", "二季度", "第二季",
    )):
        return _valid_report_period(year, "06", "30")
    if any(token in lower for token in ("q1", "一季度", "第一季")):
        return _valid_report_period(year, "03", "31")
    if any(token in lower for token in ("q3", "三季度", "第三季")):
        return _valid_report_period(year, "09", "30")
    return ""


def _valid_report_period(year: str, month: str, day: str) -> str:
    try:
        year_int, month_int, day_int = int(year), int(month), int(day)
        if not 2000 <= year_int <= 2100:
            return ""
        import datetime as _datetime
        return _datetime.date(year_int, month_int, day_int).isoformat()
    except (TypeError, ValueError):
        return ""


def _parse_chinese_number(value: str) -> int:
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if all(char in digits for char in value):
        result = 0
        for char in value:
            result = result * 10 + digits[char]
        return result
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return digits.get(value, 0)


class ChatService:
    """LangGraph-based chat orchestration: classify → react/plan/skill → summarize → reflection."""

    _BRANCH_SUMMARY_MAX_ATTEMPTS = 2

    def __init__(
        self,
        config: AgentConfig,
        tools: ToolRegistry,
        trace: TraceSink | None = None,
        llm: LLMClient | None = None,
        agent_registry: AgentRegistry | None = None,
        output_guard: object | None = None,
    ):
        self.config = config
        self.tools = tools
        self.trace = trace
        self._output_guard = output_guard  # OutputGuard or None
        self._default_llm = llm or build_llm_client(
            provider=config.agent_provider,
            deepseek_api_key=config.deepseek_api_key,
            anthropic_api_key=config.anthropic_api_key,
            agnes_api_key=config.agnes_api_key,
            model=config.agent_model,
            deepseek_base_url=config.deepseek_base_url,
            codex_bin=config.codex_bin,
            codex_workdir=config.codex_workdir,
            codex_sandbox=config.codex_sandbox,
            codex_reasoning_effort=config.codex_reasoning_effort,
            agnes_base_url=config.agnes_base_url,
            max_tokens=config.agent_max_tokens,
            timeout_sec=config.agent_model_timeout_sec,
            max_message_chars=config.max_message_chars,
        )
        self._default_provider = config.agent_provider
        self._llm_cache: dict[str, LLMClient] = {}  # per-provider+model cache

        # Pipeline LLM: fixed model for internal tasks (classify, plan, reflection)
        # When an explicit LLM is injected (e.g. tests), reuse it as pipeline_llm
        if llm is not None:
            self._pipeline_llm = llm
        else:
            self._pipeline_llm = build_llm_client(
                provider=config.pipeline_provider,
                deepseek_api_key=config.deepseek_api_key,
                anthropic_api_key=config.anthropic_api_key,
                agnes_api_key=config.agnes_api_key,
                model=config.pipeline_model,
                deepseek_base_url=config.deepseek_base_url,
                codex_bin=config.codex_bin,
                codex_workdir=config.codex_workdir,
                codex_sandbox=config.codex_sandbox,
                codex_reasoning_effort=config.codex_reasoning_effort,
                agnes_base_url=config.agnes_base_url,
                max_tokens=config.agent_max_tokens,
                timeout_sec=config.agent_model_timeout_sec,
                max_message_chars=config.max_message_chars,
            )
        # Initialize AgentRegistry
        self.agent_registry = agent_registry or _build_default_registry(config)
        self.store = SessionStore(config.store_path)
        self.store.backfill_titles()

        # L1-L4: Context management tools
        self._ref_store = ToolResultRefStore(
            config.root_path / "var" / "agent" / "tool_results.db"
        )
        # Per-user working memory insights (user_id → list[str])
        self._wm_insights: dict[str, list[str]] = {}
        # Memory evolution pipeline (consolidation, conflict resolution, forgetting)
        self._evolution = MemoryEvolution(
            self.store,
            config=EvolutionConfig(
                enable_llm_consolidation=self.config.llm_available,
            ),
            llm=self._pipeline_llm if self.config.llm_available else None,
        )
        # P3: Cross-session lesson store (failure experience persistence)
        self._lesson_store = LessonStore(
            config.root_path / "var" / "agent" / "lessons.db"
        )
        self._register_internal_tools()

        # P3: Register agent-as-tool wrappers (hierarchical agent architecture)
        try:
            from ..tools.agent_tool import register_agent_tools
            cfg_factory = self._make_cfg_factory()
            registered = register_agent_tools(self.tools, self.agent_registry, cfg_factory)
            if registered:
                logger.info("agent_as_tool: registered %d agent tools", registered)
        except Exception as exc:
            logger.warning("agent_as_tool: registration failed: %s", exc)

        # Configure rate limiter for LLM API calls
        if config.rate_limit_per_sec > 0:
            set_rate_limiter(TokenBucketRateLimiter(config.rate_limit_per_sec))

        # Pre-build and compile the LangGraph graph once
        self._graph = build_graph()
        self._checkpoint_conn = sqlite3.connect(
            config.checkpoint_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._checkpointer = SqliteSaver(self._checkpoint_conn)
        self._compiled_graph = self._graph.compile(checkpointer=self._checkpointer)

        # Store pending confirmations for HITL resume
        self._pending_confirms: dict[str, dict[str, Any]] = {}
        # Runtime state is authoritative in the same SQLite file as sessions,
        # but uses dedicated runtime_* tables and an independent store API.
        self._runtime_store = SQLiteRuntimeStore(config.store_path)
        self._branch_summary_lock = threading.Lock()
        self._branch_summary_tasks: set[str] = set()
        self._deep_research_lock = threading.Lock()
        self._deep_research_active: set[tuple[str, str]] = set()
        self._recover_branch_summaries()

    def __enter__(self) -> "ChatService":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close resources: checkpoint database connection and session store."""
        if hasattr(self, "_checkpoint_conn") and self._checkpoint_conn:
            self._checkpoint_conn.close()
        if hasattr(self, "store") and self.store:
            self.store.close()
        if hasattr(self, "_ref_store") and self._ref_store:
            self._ref_store.close()
        if hasattr(self, "_runtime_store") and self._runtime_store:
            self._runtime_store.close()

    # ---- Public API ----

    @property
    def available_providers(self) -> list[dict[str, Any]]:
        """List available providers with their models."""
        providers = []
        if self.config.llm_available or self.config.agent_provider == "codex":
            from shutil import which
            if which(self.config.codex_bin):
                providers.append({"id": "codex", "name": "本地 Codex", "models": KNOWN_MODELS.get("codex", [])})
        if self.config.deepseek_api_key:
            providers.append({"id": "deepseek", "name": "DeepSeek", "models": KNOWN_MODELS.get("deepseek", [])})
        return providers

    @property
    def available_image_models(self) -> list[dict[str, Any]]:
        """List available image generation models."""
        models = []
        if self.config.agnes_api_key:
            models.append({"provider": "agnes", "name": "Agnes AI", "models": IMAGE_MODELS.get("agnes", [])})
        return models

    @property
    def available_video_models(self) -> list[dict[str, Any]]:
        """List available video generation models."""
        models = []
        if self.config.agnes_api_key:
            models.append({"provider": "agnes", "name": "Agnes AI", "models": VIDEO_MODELS.get("agnes", [])})
        return models

    def get_provider(self, session_id: str | None = None, user_id: str = "default") -> dict[str, str]:
        """Get the LLM provider and model for a session, falling back to default."""
        if session_id:
            provider = self.store.get_provider(session_id, user_id=user_id)
            model = self.store.get_model(session_id, user_id=user_id)
            if provider:
                return {"provider": provider, "model": model or default_model(provider)}
        return {"provider": self._default_provider, "model": default_model(self._default_provider)}

    def switch_provider(self, session_id: str, provider: str, model: str = "", user_id: str = "default") -> dict[str, Any]:
        """Set the LLM provider and model for a specific session.

        Args:
            session_id: Session to configure.
            provider: One of 'deepseek', 'agnes'.
            model: Specific model ID (optional, falls back to provider default).
            user_id: Authenticated user ID.

        Returns:
            dict with 'ok', 'provider', and 'model' fields.
        """
        if provider not in {"codex", "deepseek"}:
            return {"ok": False, "error": f"unsupported provider: {provider}"}
        if not self.store.set_provider(session_id, provider, model, user_id=user_id):
            return {"ok": False, "error": "session not found or belongs to another user"}
        return {"ok": True, "provider": provider, "model": model or default_model(provider)}

    def _build_llm(self, provider: str, model: str | None = None) -> LLMClient:
        """Build (or return cached) LLM client for a provider+model."""
        cache_key = f"{provider}:{model or ''}"
        if cache_key not in self._llm_cache:
            self._llm_cache[cache_key] = build_llm_client(
                provider=provider,
                deepseek_api_key=self.config.deepseek_api_key,
                anthropic_api_key=self.config.anthropic_api_key,
                agnes_api_key=self.config.agnes_api_key,
                model=model or default_model(provider),
                deepseek_base_url=self.config.deepseek_base_url,
                codex_bin=self.config.codex_bin,
                codex_workdir=self.config.codex_workdir,
                codex_sandbox=self.config.codex_sandbox,
                codex_reasoning_effort=self.config.codex_reasoning_effort,
                agnes_base_url=self.config.agnes_base_url,
                max_tokens=self.config.agent_max_tokens,
                timeout_sec=self.config.agent_model_timeout_sec,
                max_message_chars=self.config.max_message_chars,
            )
        return self._llm_cache[cache_key]

    def _get_llm(self, session_id: str | None, user_id: str = "default") -> LLMClient:
        """Get the LLM client for a session, using stored provider/model."""
        if session_id:
            provider = self.store.get_provider(session_id, user_id=user_id)
            if provider:
                model = self.store.get_model(session_id, user_id=user_id)
                return self._build_llm(provider, model or None)
        return self._default_llm

    def reset(self, session_id: str, user_id: str = "default") -> bool:
        if session_id:
            # Reset remains idempotent for a session that does not exist,
            # while refusing to mutate an existing session owned by another user.
            if self.store.get_session(session_id) is None:
                reset = True
            else:
                reset = self.store.reset(session_id, user_id=user_id)
            if reset:
                self._runtime_store.delete_session_entries(user_id, session_id)
            return reset
        return True

    def schedule_branch_summary(
        self,
        session_id: str,
        from_message_id: str,
        abandoned_leaf_id: str | None,
        user_id: str = "default",
    ) -> None:
        """Generate a best-effort summary after a branch switch.

        Branch switching remains synchronous and append-only. Summary
        generation is deliberately detached so an unavailable model cannot
        block the user's next turn.
        """
        messages = self.store.get_abandoned_branch(
            session_id,
            from_message_id,
            abandoned_leaf_id,
            user_id=user_id,
        )
        if not messages:
            return
        abandoned_leaf = abandoned_leaf_id or ""
        try:
            entry_id, payload = self._runtime_store.ensure_branch_summary_entry(
                user_id, session_id, from_message_id, abandoned_leaf, len(messages),
            )
            if payload.get("status") == "completed":
                return
            if payload.get("status") == "failed":
                # A new explicit branch request is the user-level retry
                # boundary; automatic restart recovery keeps the old count.
                payload["attempts"] = 0
            payload.update({
                "status": "scheduled",
                "error": "",
                "message_count": len(messages),
            })
            if not self._runtime_store.update_session_entry(user_id, entry_id, payload):
                logger.warning("branch_summary entry missing: %s", entry_id)
                return
        except Exception as exc:
            # The branch has already been committed.  Summary persistence is
            # best effort and must not turn a successful branch into a 500.
            logger.warning("branch_summary scheduling persistence failed: %s", exc)
            return
        self._enqueue_branch_summary(
            entry_id, session_id, from_message_id, abandoned_leaf, user_id, messages,
        )

    def _recover_branch_summaries(self) -> None:
        """Requeue branch summaries left scheduled/running by a restart."""
        for entry in self._runtime_store.list_pending_session_entries(
            "branch_summary", ("scheduled", "running"),
        ):
            payload = entry["payload"]
            from_message_id = str(payload.get("from_message_id", "")).strip()
            abandoned_leaf_id = str(payload.get("abandoned_leaf_id", "")).strip()
            if not from_message_id or not abandoned_leaf_id:
                continue
            messages = self.store.get_abandoned_branch(
                entry["session_id"], from_message_id, abandoned_leaf_id,
                user_id=entry["owner_id"],
            )
            if messages:
                self._enqueue_branch_summary(
                    entry["entry_id"], entry["session_id"], from_message_id,
                    abandoned_leaf_id, entry["owner_id"], messages,
                )

    def _enqueue_branch_summary(
        self,
        entry_id: str,
        session_id: str,
        from_message_id: str,
        abandoned_leaf_id: str,
        user_id: str,
        messages: list[dict[str, str]],
    ) -> None:
        with self._branch_summary_lock:
            if entry_id in self._branch_summary_tasks:
                return
            self._branch_summary_tasks.add(entry_id)
        threading.Thread(
            target=self._generate_branch_summary,
            args=(entry_id, session_id, from_message_id, abandoned_leaf_id, user_id, messages),
            daemon=True,
            name=f"branch-summary-{entry_id}",
        ).start()

    def _generate_branch_summary(
        self,
        entry_id: str,
        session_id: str,
        from_message_id: str,
        abandoned_leaf_id: str,
        user_id: str,
        messages: list[dict[str, str]],
    ) -> None:
        payload: dict[str, Any] = {}
        try:
            entry = self._runtime_store.get_session_entry(user_id, entry_id)
            payload = dict(entry.get("payload", {})) if entry else {}
            payload.update({
                "from_message_id": from_message_id,
                "abandoned_leaf_id": abandoned_leaf_id,
                "message_count": len(messages),
                "status": "running",
                "started_at": time.time(),
                "error": "",
            })
            if not self._runtime_store.update_session_entry(user_id, entry_id, payload):
                logger.warning("branch_summary entry missing: %s", entry_id)
                return
            transcript = "\n".join(
                f"{item['role']}: {item['content'][:3000]}" for item in messages
            )[:24000]
            summary_result: dict[str, Any] | None = None
            last_error: Exception | None = None
            for attempt in range(self._BRANCH_SUMMARY_MAX_ATTEMPTS):
                current_attempts = int(payload.get("attempts", 0))
                if current_attempts >= self._BRANCH_SUMMARY_MAX_ATTEMPTS:
                    break
                payload["attempts"] = current_attempts + 1
                payload["last_attempt_at"] = time.time()
                if not self._runtime_store.update_session_entry(user_id, entry_id, payload):
                    raise RuntimeError("branch summary entry disappeared during retry")
                try:
                    result = self._pipeline_llm.complete_json(
                        """
你负责生成会话分支摘要。请只输出 JSON，不要 Markdown：
{"summary":"不超过 500 字的中文摘要","key_points":["不超过 5 条"],"unresolved":"未解决问题，没有则为空"}
摘要只描述用户和助手已经讨论的事实，不要补充推测，不要输出隐式思维链。
""".strip(),
                        [{"role": "user", "content": transcript}],
                        temperature=0.2,
                    )
                    if not isinstance(result, dict):
                        raise ValueError("branch summary response is not an object")
                    summary = str(result.get("summary", "")).strip()
                    if not summary:
                        raise ValueError("branch summary is empty")
                    summary_result = {
                        "summary": summary[:2000],
                        "key_points": [str(item)[:300] for item in result.get("key_points", [])][:5]
                        if isinstance(result.get("key_points", []), list) else [],
                        "unresolved": str(result.get("unresolved", ""))[:500],
                    }
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < self._BRANCH_SUMMARY_MAX_ATTEMPTS:
                        logger.warning(
                            "branch_summary retry entry=%s attempt=%d error=%s",
                            entry_id, attempt + 1, exc,
                        )
                        time.sleep(0.5)
            if summary_result is None:
                raise last_error or ValueError("branch summary attempts exhausted")
            payload.update({
                "status": "completed",
                **summary_result,
                "completed_at": time.time(),
            })
        except Exception as exc:
            logger.warning("branch_summary failed: %s", exc)
            payload.update({
                "status": "failed",
                "error": str(exc)[:300],
                "failed_at": time.time(),
            })
        finally:
            try:
                if not self._runtime_store.update_session_entry(user_id, entry_id, payload):
                    logger.warning("branch_summary entry missing: %s", entry_id)
            except Exception as exc:
                logger.warning("branch_summary persistence failed: %s", exc)
            with self._branch_summary_lock:
                self._branch_summary_tasks.discard(entry_id)

    def _load_file_content(self, file_id: str) -> str | dict[str, Any]:
        """Load uploaded file content for injection into chat messages.

        Returns:
            - str: text content for text/PDF files
            - dict: {"type": "image", "mime_type": "...", "base64": "..."} for images
        """
        upload_dir = self.config.root_path.parent / "var" / "uploads"
        for ext in (".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf"):
            file_path = upload_dir / f"{file_id}{ext}"
            if file_path.exists():
                if ext in (".txt", ".md", ".csv", ".json", ".yaml", ".yml"):
                    return file_path.read_text(encoding="utf-8", errors="replace")
                elif ext == ".pdf":
                    try:
                        import PyPDF2
                        reader = PyPDF2.PdfReader(str(file_path))
                        pages = []
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                pages.append(text)
                        return "\n\n".join(pages)
                    except ImportError:
                        return f"[PDF: {file_path.name}]"
                else:
                    # Image: return base64 data for vision model
                    import base64
                    content = file_path.read_bytes()
                    mime_map = {
                        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".webp": "image/webp", ".gif": "image/gif",
                    }
                    return {
                        "type": "image",
                        "mime_type": mime_map.get(ext, "image/png"),
                        "base64": base64.b64encode(content).decode("utf-8"),
                    }
        return ""

    def stream_chat(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str = "default",
        file_id: str | None = None,
        mode: str = "",
        agent_mode: str = "",
        preset: str = "",
        debug_trace: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """LangGraph-based streaming chat with classify → react/plan/skill → summarize → reflection."""
        started = time.perf_counter()
        sid = session_id or uuid.uuid4().hex
        text = message.strip()
        if session_id and self.store.get_session(sid, user_id=user_id) is None:
            if self.store.get_session(sid) is not None:
                yield {"type": "error", "message": "session not found or belongs to another user"}
                yield {"type": "done", "session_id": sid, "duration_ms": 0}
                return
        if not text:
            yield {"type": "error", "message": "message is required"}
            yield {"type": "done", "session_id": sid, "duration_ms": 0}
            return

        # Inject uploaded file content into the message
        attachments: list[dict[str, Any]] = []
        if file_id:
            file_content = self._load_file_content(file_id)
            if file_content:
                if isinstance(file_content, dict) and file_content.get("type") == "image":
                    attachments.append(file_content)
                    # Add a hint in the text so classification still works
                    if not text:
                        text = "请描述这张图片的内容"
                else:
                    # Text/PDF: inject as before
                    content_str = file_content if isinstance(file_content, str) else ""
                    if content_str:
                        text = f"[文件内容]\n{content_str}\n\n[用户问题]\n{text}"

        if not self.config.llm_available:
            yield {
                "type": "error",
                "message": f"LLM unavailable: {self.config.llm_unavailable_reason}",
            }
            yield {"type": "done", "session_id": sid, "duration_ms": 0}
            return

        try:
            execution_policy = resolve_agent_policy(mode=agent_mode, preset=preset)
            if debug_trace:
                execution_policy = replace(execution_policy, debug_trace=True)
        except ValueError as err:
            yield {"type": "error", "message": str(err)}
            yield {"type": "done", "session_id": sid, "duration_ms": 0}
            return

        # Load conversation history for context injection into LLM calls
        history = self._get_history(sid, user_id)
        call_id = str(uuid.uuid4())

        # Clean up stale checkpoint from previous call (P0-4: prevents reducer merge)
        self._cleanup_stale_checkpoint(sid, call_id)

        session_llm = self._get_llm(sid, user_id=user_id)
        # Top-level execution is Runtime-managed.  The legacy ReAct loop is
        # retained only for nested Agent-as-Tool compatibility in commander.py.
        runtime_mode = "runtime"

        initial_state = AgentState(
            user_message=text, session_id=sid, call_id=call_id,
            reflexion_max=0 if mode == "deep_research" else self.config.reflexion_max_attempts,
            attachments=attachments,
            owner_id=user_id,
            runtime_mode=runtime_mode,
            orchestration_run_id=call_id,
        )

        self._backfill_runtime_history(user_id, sid, history)
        self._runtime_store.append_session_entry(
            user_id, sid, "user", {
                "content": build_multimodal_content(text, attachments),
            },
        )
        runtime_user_entry_written = True

        interrupted = False
        try:
            logger.debug(
                "llm_request: provider=%s model=%s message_len=%d",
                session_llm.provider if hasattr(session_llm, 'provider') else "?",
                session_llm.model if hasattr(session_llm, 'model') else "?",
                len(text),
            )

            # Deep research is a workflow contract, not a Codex-only path.
            # DeepSeek also implements ``complete_json`` and must use the
            # fixed evidence-first workflow; otherwise a deep-research
            # request falls through to the general Commander graph and can
            # exhaust its tool-call budget without producing schema v2.
            if mode == "deep_research":
                yield from self._stream_deep_research_runtime(
                    session_llm, sid, text, history, user_id, attachments,
                )
            elif getattr(session_llm, "provider", "") == "codex":
                yield from self._stream_codex_direct_runtime(
                    session_llm, sid, text, history, user_id, attachments,
                )
            else:
                graph_config = self._build_graph_config(
                    sid, session_llm, history, text, user_id, attachments,
                    agent_policy=execution_policy,
                )
                try:
                    final_state = yield from self._stream_graph_events(
                        initial_state, graph_config, emit_classify=True,
                    )
                    yield from self._finalize_stream(
                        final_state, sid, text, session_llm, history, user_id,
                        runtime_user_entry_written=runtime_user_entry_written,
                    )
                except GraphInterrupt as gi:
                    interrupted = True
                    yield from self._handle_hitl_interrupt(
                        gi, sid, graph_config, session_llm, user_id,
                    )
        except GeneratorExit:
            raise
        except Exception as err:
            logger.error("stream_chat error: %s\n%s", err, traceback.format_exc())
            yield {"type": "error", "message": "系统内部错误，请稍后重试"}
            # ── Graceful degradation: always provide a user-facing message ──
            yield {"type": "token", "content": (
                "抱歉，系统处理您的请求时遇到了问题。\n\n"
                "建议：稍等片刻后重新提问，或尝试换一种方式描述您的问题。"
            )}
        finally:
            if not interrupted:
                # Keep latest checkpoint for recovery (P0-4: 断点恢复)
                self._prune_checkpoints(sid, keep_latest=True)
        yield {
            "type": "done",
            "session_id": sid,
            "duration_ms": round((time.perf_counter() - started) * 1000),
        }

    def _stream_codex_direct_runtime(
        self,
        llm: LLMClient,
        session_id: str,
        question: str,
        history: list[dict[str, str]],
        user_id: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Durably map Codex's own agent loop into Runtime events."""
        system = (
            "你是筋斗云的本地 Codex 助理。请直接回答用户问题。"
            "可以读取个人系统工作区中的文件来获取上下文，但当前是只读沙箱，"
            "不要修改文件或执行破坏性操作。回答使用中文，事实不确定时明确说明。"
        )
        messages = history[-8:] + [{
            "role": "user",
            "content": build_multimodal_content(question, attachments),
        }]
        adapter = ExternalAgentAdapter(self._runtime_store, llm)
        handle = adapter.start(
            owner_id=user_id,
            session_id=session_id,
            agent_id="codex-direct",
            system=system,
            messages=messages,
            metadata={"provider": "codex", "mode": "direct"},
        )
        first = True
        emitted_error = False
        for event in handle.events():
            ui_event = dict(event.ui_event)
            if first:
                first = False
                yield ui_event
                yield {"type": "progress", "message": "本地 Codex 正在处理…"}
                continue
            if ui_event.get("type") not in {"done"}:
                if ui_event.get("type") == "message":
                    ui_event = {"type": "token", "content": ui_event.get("content", "")}
                emitted_error = emitted_error or ui_event.get("type") == "error"
                yield ui_event
        result = handle.result()
        if result.outcome.value == "completed" and result.final_message:
            self._remember(
                session_id, question, result.final_message,
                user_id=user_id, runtime_user_entry_written=True,
            )
        elif result.outcome.value != "completed" and not emitted_error:
            yield {"type": "error", "message": "本地 Codex 响应失败，请稍后重试"}

    def _stream_deep_research_runtime(
        self,
        llm: LLMClient,
        session_id: str,
        question: str,
        history: list[dict[str, str]],
        user_id: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run one deep-research operation per owner/session at a time."""
        key = (user_id, session_id)
        with self._deep_research_lock:
            if key in self._deep_research_active:
                yield {"type": "error", "message": "该研究任务仍在执行，请等待当前任务完成后再重试。"}
                return
            self._deep_research_active.add(key)
        try:
            yield from self._run_deep_research_runtime(
                llm, session_id, question, history, user_id, attachments,
            )
        finally:
            with self._deep_research_lock:
                self._deep_research_active.discard(key)

    def _run_deep_research_runtime(
        self,
        llm: LLMClient,
        session_id: str,
        question: str,
        history: list[dict[str, str]],
        user_id: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run the fixed evidence workflow with durable Runtime events."""
        del history
        import re

        name_match = re.search(r"研究对象[:：]\s*(.+)", question)
        research_date = (
            question.split("研究日期：", 1)[1].splitlines()[0].strip()
            if "研究日期：" in question else ""
        )
        agent_tools = self.agent_registry.build_tool_registry(
            "investment-analyst", self.tools,
        )
        workflow = DeepResearchWorkflow(
            self._runtime_store,
            llm,
            agent_tools,
            normalize_result=_normalize_research_result,
            validate_result=_research_result_issues,
            preview_json=preview_json,
            select_latest_official_url=_select_latest_official_url,
            extract_official_report_period=_extract_official_report_period,
        )
        handle = workflow.start(
            owner_id=user_id,
            session_id=session_id,
            question=question,
            name=name_match.group(1).strip() if name_match else "",
            research_date=research_date,
            attachments=attachments,
        )
        emitted_error = False
        for event in handle.events():
            if event.ui_event.get("type") != "done":
                emitted_error = emitted_error or event.ui_event.get("type") == "error"
                yield event.ui_event
        result = handle.result()
        if result.outcome.value == "completed" and result.final_message:
            self._remember(
                session_id, question, result.final_message,
                user_id=user_id, runtime_user_entry_written=True,
            )
        elif result.outcome.value != "completed" and not emitted_error:
            yield {"type": "error", "message": result.error or "深度研究汇总失败，请稍后重试"}

    def resume_chat(
        self, session_id: str, decision: str = "approve", user_id: str = "default",
    ) -> Iterator[dict[str, Any]]:
        """Resume a paused graph after user confirmation.

        Args:
            session_id: The session ID of the paused graph.
            decision: User's decision: 'approve' or 'skip'.

        Yields:
            SSE events from the resumed graph execution.
        """
        started = time.perf_counter()
        pending = self._pending_confirms.get(session_id)
        runtime_operation = self._runtime_store.find_active(user_id, session_id)
        if runtime_operation is not None:
            yield from self._resume_runtime_chat(
                runtime_operation.operation_id, session_id, decision, user_id,
            )
            return
        if not pending:
            yield {"type": "error", "message": "no pending confirmation for this session"}
            yield {"type": "done", "session_id": session_id, "duration_ms": 0}
            return
        if pending.get("user_id") != user_id:
            yield {"type": "error", "message": "session not found or belongs to another user"}
            yield {"type": "done", "session_id": session_id, "duration_ms": 0}
            return
        self._pending_confirms.pop(session_id, None)

        graph_config = pending["config"]
        # Ensure the resumed graph has an event queue for real-time streaming
        if "event_queue" not in graph_config.get("configurable", {}):
            graph_config.setdefault("configurable", {})["event_queue"] = queue.Queue()
        session_llm = pending["session_llm"]
        user_id = pending["user_id"]

        logger.info("hitl: resuming session=%s decision=%s", session_id, decision)

        try:
            final_state = yield from self._stream_graph_events(
                Command(resume=decision), graph_config, emit_classify=False,
            )

            # Yield final answer
            final_answer = final_state.get("final_answer", "")
            # FINAL SAFETY NET: strip any leaked verification tags (same as normal path)
            if final_answer:
                final_answer = _strip_all_verification_tags(final_answer)
            if not final_answer:
                # Graceful degradation: no answer produced after resume
                final_answer = "抱歉，恢复会话后未能生成回复。请重新提问。"
            yield {"type": "token", "content": final_answer}

        except GraphInterrupt as gi:
            # Another confirmation needed — update pending confirms so session can recover
            interrupt_value = gi.args[0] if gi.args else {}
            pending_actions = interrupt_value.get("actions", [])
            logger.info(
                "hitl: second confirm_required session=%s actions=%d",
                session_id, len(pending_actions),
            )
            self._pending_confirms[session_id] = {
                "config": graph_config,
                "session_llm": session_llm,
                "user_id": user_id,
            }
            yield {
                "type": "confirm_required",
                "actions": pending_actions,
                "session_id": session_id,
            }
        except GeneratorExit:
            raise
        except Exception as err:
            logger.error("resume error: %s\n%s", err, traceback.format_exc())
            yield {"type": "error", "message": "恢复会话失败，请稍后重试"}
        finally:
            self._prune_checkpoints(session_id, keep_latest=False)
        yield {
            "type": "done",
            "session_id": session_id,
            "duration_ms": round((time.perf_counter() - started) * 1000),
        }

    def _resume_runtime_chat(
        self, operation_id: str, session_id: str, decision: str, user_id: str,
    ) -> Iterator[dict[str, Any]]:
        """Resume durable Runtime state after a process restart."""
        started = time.perf_counter()
        operation = self._runtime_store.load(user_id, operation_id)
        if operation is None:
            yield {"type": "error", "message": "operation not found or belongs to another user"}
            yield {"type": "done", "session_id": session_id, "duration_ms": 0}
            return
        try:
            agent_def = self.agent_registry.get(operation.agent_id)
            if agent_def is None:
                raise ValueError(f"Agent not found: {operation.agent_id}")
            agent_tools = self.agent_registry.build_tool_registry(operation.agent_id, self.tools)
            runtime = AgentRuntime(
                self._runtime_store,
                model=MatrixModelAdapter(self._get_llm(session_id, user_id=user_id)),
                tools=MatrixToolAdapter(
                    agent_tools,
                    session_id=session_id,
                    owner_id=user_id,
                    mode=str(operation.state.get("execution_policy", {}).get("mode", "read_only")),
                    allow_external_effects=bool(operation.state.get("execution_policy", {}).get("allow_external_effects", False)),
                ),
            )
            pending = operation.state.get("pending_tool_call", {})
            handle = runtime.resume(
                user_id, operation_id,
                ResumeInput(
                    kind="approval", decision=decision,
                    payload={"approval_id": pending.get("approval_id", "")},
                ),
            )
            events = list(handle.events())
            result = handle.result()
            for trace_event in handle.debug_trace():
                yield {
                    "type": "debug_trace",
                    "operation_id": operation_id,
                    "event": trace_event,
                }
            for event in events:
                if event.event_type.value == "tool_start":
                    yield {"type": "tool_call", "name": event.payload.get("name", ""), "operation_id": operation_id}
                elif event.event_type.value == "tool_end":
                    yield {"type": "tool_result", "name": event.payload.get("name", ""), "error": event.payload.get("error", ""), "operation_id": operation_id}
            if result.final_message:
                yield {"type": "token", "content": result.final_message}
                self._remember(session_id, "", result.final_message, user_id=user_id)
            if result.outcome.value == "suspended":
                yield {"type": "confirm_required", "actions": [result.suspension.payload if result.suspension else {}], "session_id": session_id}
            elif result.error:
                yield {"type": "error", "message": result.error}
        except GeneratorExit:
            raise
        except Exception as err:
            logger.error("runtime resume error: %s", err, exc_info=True)
            yield {"type": "error", "message": "恢复 Runtime 操作失败，请稍后重试"}
        yield {
            "type": "done",
            "session_id": session_id,
            "duration_ms": round((time.perf_counter() - started) * 1000),
        }

    # ---- Internal ----

    def _handle_working_memory(self, action: str, content: str, user_id: str = "default") -> dict:
        """Handle working_memory tool calls from the LLM.

        Used by the LLM to record key insights that survive context compression.
        Insights are isolated per user_id to prevent cross-user leakage.
        """
        if action == "add_insight" and content:
            insights = self._wm_insights.setdefault(user_id, [])
            insights.insert(0, content)
            self._wm_insights[user_id] = insights[:20]  # cap at 20 insights
            return {"ok": True, "recorded": content, "total_insights": len(self._wm_insights[user_id])}
        return {"ok": False, "error": f"Unknown action: {action}"}

    def _register_internal_tools(self) -> None:
        """Register context management tools (P0-P2: get_stored_data, working_memory).

        Extracted from __init__ to keep the constructor focused on wiring.
        """
        self.tools.register(ToolDefinition(
            name="get_stored_data",
            description=(
                "Retrieve full data that was externalized from context. "
                "Use when you see a __refId reference and need the complete data. "
                "Pass the refId from the __refId field."
            ),
            capabilities=["data_cache"],
            input_schema={
                "type": "object",
                "properties": {
                    "refId": {"type": "string", "description": "The reference ID from a __refId field"},
                },
                "required": ["refId"],
            },
            handler=make_get_stored_data_tool(self._ref_store),
        ))
        self.tools.register(ToolDefinition(
            name="working_memory",
            description=(
                "Record a key insight or finding that should survive context compression. "
                "Use this when you discover a critical piece of information (value, ID, "
                "constraint, decision) that future steps need to know."
            ),
            capabilities=["memory"],
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add_insight"], "description": "Always 'add_insight' to record a finding"},
                    "content": {"type": "string", "description": "The insight to record. Be specific: include values, IDs, names."},
                },
                "required": ["action", "content"],
            },
            handler=self._handle_working_memory,
        ))

    def _make_cfg_factory(self) -> Callable[[], dict[str, Any]]:
        """Create a cfg factory for agent-as-tool handlers.

        Returns a callable that produces a base cfg dict with service-level
        configuration. The handler will merge in task-specific fields (question,
        working_memory, etc.) before calling _run_domain_agent_react.
        """
        def factory() -> dict[str, Any]:
            return {
                "llm": self._default_llm,
                "pipeline_llm": self._pipeline_llm,
                "agent_registry": self.agent_registry,
                "full_tools": self.tools,
                "ref_store": self._ref_store,
                "lesson_store": self._lesson_store,
                "history": [],
                "attachments": [],
                "working_memory": {"pinned": "", "insights": []},
                "circuit_breaker": CircuitBreaker(),
                "question": "",
                "runtime_store": self._runtime_store,
                "execution_policy": ExecutionPolicy(),
                "user_id": "default",
            }
        return factory

    def _cleanup_stale_checkpoint(self, thread_id: str, call_id: str) -> None:
        """Delete stale checkpoints from a previous call to prevent reducer merge.

        If any checkpoint exists from a previous call with the same thread_id,
        we delete it. The call_id check ensures we only clean up when starting
        a genuinely new call (not when resuming an interrupted one).
        """
        try:
            conn = self._checkpoint_conn
            if conn is None:
                return
            row = conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if row and row[0] > 0:
                logger.info(
                    "cleanup_stale_checkpoint: removing %d stale checkpoints "
                    "thread_id=%s call_id=%s",
                    row[0], thread_id, call_id,
                )
                self._prune_checkpoints(thread_id, keep_latest=False)
        except Exception:
            # Best-effort cleanup; if it fails, the graph will still run
            pass

    def _prune_checkpoints(self, thread_id: str, keep_latest: bool = True) -> None:
        """Clean up checkpoints per thread to prevent unbounded growth.

        LangGraph's SqliteSaver writes a checkpoint after every node execution.
        Each user question generates 5-6 checkpoints. Over time this accumulates
        useless history.

        Two modes:
        - keep_latest=True (default): keeps only the latest checkpoint, used for
          HITL (interrupt/resume) where the paused state must be preserved.
        - keep_latest=False: deletes ALL checkpoints for this thread, used for
          normal call completion. This is critical because operator.add reducers
          on AgentState fields (agent_results, tool_results, tool_call_count)
          would otherwise merge stale checkpointed state into the next call,
          causing duplicate results and incorrect routing.
        """
        try:
            conn = self._checkpoint_conn
            if conn is None:
                return
            if keep_latest:
                # Delete all but the latest checkpoint for this thread
                conn.execute(
                    """DELETE FROM checkpoints
                       WHERE (thread_id, checkpoint_ns, checkpoint_id) IN (
                         SELECT thread_id, checkpoint_ns, checkpoint_id
                         FROM checkpoints
                         WHERE thread_id = ?
                         ORDER BY checkpoint_id DESC
                         LIMIT -1 OFFSET 1
                       )""",
                    (thread_id,),
                )
            else:
                # Delete all checkpoints for this thread
                conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?",
                    (thread_id,),
                )
            # Clean orphaned writes (no matching checkpoint)
            conn.execute(
                """DELETE FROM writes
                   WHERE (thread_id, checkpoint_ns, checkpoint_id) NOT IN (
                     SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints
                   )""",
            )
            conn.execute("PRAGMA optimize")
        except Exception:
            pass  # Pruning is best-effort; never fail the chat for it

    # ---- Stream event helpers (shared between stream_chat and resume_chat) ----

    def _stream_graph_events(
        self,
        graph_input: Any,
        graph_config: dict[str, Any],
        *,
        emit_classify: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Stream LangGraph events with common agent/tool/error emission.

        Args:
            graph_input: Initial state (for stream_chat) or Command(resume=...) (for resume_chat).
            graph_config: LangGraph config dict with configurable and thread_id.
            emit_classify: If True, emit a classify event when delegation_plan appears.

        Yields:
            SSE event dicts (classify, progress, tool_call, tool_result, agent_result, error, token).
        """
        emitted_tool_count = 0
        emitted_agent_count = 0
        classify_emitted = False
        _queue_emitted: set[tuple[str, str]] = set()
        final_state: dict[str, Any] = {}

        for event in self._compiled_graph.stream(
            graph_input, stream_mode="values", config=graph_config,
        ):
            yield from _drain_queue(graph_config["configurable"]["event_queue"], _queue_emitted)
            if not isinstance(event, dict):
                continue
            final_state = event

            # Emit classify event (stream_chat only)
            if emit_classify and not classify_emitted:
                delegation_plan = event.get("delegation_plan")
                if delegation_plan and len(delegation_plan) > 0:
                    classify_emitted = True
                    intent = "delegate" if len(delegation_plan) > 1 or (
                        delegation_plan and delegation_plan[0].get("agent_id") != "commander"
                    ) else "simple"
                    yield {"type": "classify", "intent": intent, "delegation_plan": delegation_plan}

            # Emit agent events
            agent_results = event.get("agent_results", [])
            if len(agent_results) > emitted_agent_count:
                emitted_agent_count = yield from self._emit_agent_events(
                    agent_results, emitted_agent_count,
                )

            # Emit tool events
            tool_results = event.get("tool_results", [])
            if len(tool_results) > emitted_tool_count:
                emitted_tool_count = yield from self._emit_tool_events(
                    tool_results, emitted_tool_count, _queue_emitted,
                )

            # Emit error
            error = event.get("error", "")
            if error:
                yield {"type": "error", "message": error}

        return final_state

    def _emit_agent_events(
        self, agent_results: list[dict], emitted_agent_count: int,
    ) -> int:
        """Yield agent_result events for newly emitted agents. Returns new count."""
        new_count = emitted_agent_count
        for i in range(emitted_agent_count, len(agent_results)):
            ar = agent_results[i]
            yield {
                "type": "agent_result",
                "agent_id": ar.get("agent_id", ""),
                "task": ar.get("task", ""),
                "content": ar.get("result", ""),
                "result": ar.get("result", "")[:500],
                "error": ar.get("error", ""),
            }
            new_count += 1
        return new_count

    def _emit_tool_events(
        self, tool_results: list[dict], emitted_tool_count: int,
        queue_emitted: set[tuple[str, str]],
    ) -> int:
        """Yield tool_call + tool_result events for new tools. Returns new count."""
        new_count = emitted_tool_count
        for i in range(emitted_tool_count, len(tool_results)):
            tr = tool_results[i]
            if tr.get("duplicate") or tr.get("name") == "_knowledge":
                continue
            tr_key = (tr.get("name", ""), json.dumps(tr.get("arguments", {}), sort_keys=True))
            if tr_key in queue_emitted:
                continue
            yield {
                "type": "tool_call",
                "name": tr.get("name", ""),
                "args": tr.get("arguments", {}),
            }
            yield {
                "type": "tool_result",
                "name": tr.get("name", ""),
                "result": tr.get("result"),
                "error": tr.get("error"),
                "elapsed_ms": tr.get("elapsed_ms"),
                "preview": preview_json(
                    tr.get("error", tr.get("result", {})),
                    limit=2000,
                ),
            }
            new_count += 1
        return new_count

    def _build_graph_config(
        self, sid: str, session_llm: LLMClient, history: list[dict],
        user_message: str = "", user_id: str = "default",
        attachments: list[dict[str, Any]] | None = None,
        agent_policy: ExecutionPolicy | None = None,
    ) -> dict[str, Any]:
        """Build the LangGraph config dict for a streaming session."""
        return {
            "configurable": {
                "llm": session_llm,
                "pipeline_llm": self._pipeline_llm,
                "agent_registry": self.agent_registry,
                "full_tools": self.tools,
                "trace": self.trace,
                "history": history,
                "event_queue": queue.Queue(),
                "ref_store": self._ref_store,
                "attachments": attachments or [],
                "working_memory": {
                    "pinned": user_message,
                    "insights": list(self._wm_insights.get(user_id, [])),
                },
                "circuit_breaker": CircuitBreaker(),
                "lesson_store": self._lesson_store,
                "user_id": user_id,
                "runtime_store": self._runtime_store,
                "execution_policy": agent_policy or ExecutionPolicy(),
            },
            "thread_id": sid,
        }

    def _finalize_stream(
        self, final_state: dict[str, Any], sid: str, text: str,
        session_llm: LLMClient, history: list[dict], user_id: str,
        runtime_user_entry_written: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Handle output after graph streaming completes: summarize or direct answer."""
        if final_state.get("needs_summary"):
            answer_parts: list[str] = []
            for event in self._stream_summarize(final_state, sid, text, session_llm, history):
                yield event
                if event["type"] == "token":
                    answer_parts.append(event["content"])
            answer = "".join(answer_parts)
            # ── FINAL SAFETY NET: strip any leaked verification tags ──
            # Same gate as the direct-answer path below.
            if answer:
                answer = _strip_all_verification_tags(answer)
            # ---- OUTPUT GUARD ----
            if answer and self._output_guard:
                result = self._output_guard.check(answer, user_id=user_id)
                if result.had_pii:
                    logger.warning("output_pii_detected: flags=%s session=%s", result.flags, sid)
                answer = result.sanitized
            # ---- END OUTPUT GUARD ----
            # ── Guard: if stream produced no answer (LLM returned empty) ──
            if not answer:
                logger.warning(
                    "_finalize_stream: needs_summary path produced empty answer, session=%s", sid,
                )
                answer = "抱歉，暂时无法生成回复。请稍后重试或换一种方式提问。"
                yield {"type": "token", "content": answer}
            if answer:
                self._remember(
                    sid, text, answer, user_id=user_id,
                    runtime_user_entry_written=runtime_user_entry_written,
                )
        else:
            final_answer = final_state.get("final_answer", "")
            # ── FINAL SAFETY NET: strip any leaked verification tags ----
            # This is the last gate before output reaches the user. Regardless
            # of which internal path produced this answer (ReAct, aggregate,
            # reflection revision, commander pass-through, error fallback),
            # strip ALL verification markup here so it can NEVER leak to UI.
            if final_answer:
                final_answer = _strip_all_verification_tags(final_answer)
            # ---- END SAFETY NET ----
            # ---- OUTPUT GUARD ----
            if final_answer and self._output_guard:
                result = self._output_guard.check(final_answer, user_id=user_id)
                if result.had_pii:
                    logger.warning("output_pii_detected: flags=%s session=%s", result.flags, sid)
                final_answer = result.sanitized
            # ---- END OUTPUT GUARD ----
            if not final_answer:
                # ── Graceful degradation: no answer produced at all ──
                final_answer = "抱歉，暂时无法生成回复。请稍后重试或换一种方式提问。"
            yield {"type": "token", "content": final_answer}
            self._remember(
                sid, text, final_answer, user_id=user_id,
                runtime_user_entry_written=runtime_user_entry_written,
            )

    def _handle_hitl_interrupt(
        self, gi: Any, sid: str, graph_config: dict[str, Any],
        session_llm: LLMClient, user_id: str,
    ) -> Iterator[dict[str, Any]]:
        """Handle GraphInterrupt: store pending state and yield HITL events."""
        interrupt_value = gi.args[0] if gi.args else {}
        pending_actions = interrupt_value.get("actions", [])
        logger.info(
            "hitl: confirm_required session=%s actions=%d",
            sid, len(pending_actions),
        )
        self._pending_confirms[sid] = {
            "config": graph_config,
            "session_llm": session_llm,
            "user_id": user_id,
        }
        self._prune_checkpoints(sid, keep_latest=True)
        yield {
            "type": "confirm_required",
            "actions": pending_actions,
            "session_id": sid,
        }

    def reload_skills(self) -> None:
        """Reload skills from disk (after CRUD)."""
        self.agent_registry.reload_skills()

    def _get_history(self, session_id: str, user_id: str = "default") -> list[dict[str, Any]]:
        """Return conversation history with layered user profile injected as context."""
        legacy_history = self.store.get_history(
            session_id, self.config.memory_max_turns, user_id=user_id,
        )
        runtime_history = self._get_runtime_history(session_id, user_id)
        has_branches = bool(self.store.get_branches(session_id, user_id=user_id))
        if runtime_history and not has_branches:
            history = _merge_runtime_history(
                legacy_history, runtime_history, self.config.memory_max_turns,
            )
        else:
            history = legacy_history
        formatted = self.store.get_profile_formatted(user_id)
        if formatted:
            history.insert(0, {"role": "system", "content": formatted})
        return history

    def _get_runtime_history(
        self, session_id: str, user_id: str,
    ) -> list[dict[str, Any]]:
        """Read Runtime user/assistant entries in chronological order."""
        if not hasattr(self._runtime_store, "list_session_entries"):
            return []
        entries: list[dict[str, Any]] = []
        per_role_limit = max(1, self.config.memory_max_turns)
        for entry_type in ("user", "assistant"):
            entries.extend(self._runtime_store.list_session_entries(
                user_id, session_id, entry_type=entry_type, limit=per_role_limit,
            ))
        entries.sort(key=lambda item: (item.get("created_at", 0), item.get("entry_id", "")))
        history: list[dict[str, Any]] = []
        for entry in entries:
            payload = entry.get("payload", {})
            content = payload.get("content") if isinstance(payload, dict) else None
            if isinstance(content, (str, list)):
                history.append({"role": entry["entry_type"], "content": content})
        return history[-self.config.memory_max_turns * 2:]

    def _backfill_runtime_history(
        self, user_id: str, session_id: str, history: list[dict[str, Any]],
    ) -> None:
        """Seed Runtime history once for sessions created before migration."""
        if self._get_runtime_history(session_id, user_id):
            return
        if self.store.get_branches(session_id, user_id=user_id):
            return
        for item in history:
            if item.get("role") in {"user", "assistant"}:
                self._runtime_store.append_session_entry(
                    user_id, session_id, item["role"], {"content": item.get("content", "")},
                )

    def _remember(
        self, session_id: str, question: str, answer: str,
        user_id: str = "default",
        runtime_user_entry_written: bool = False,
    ) -> None:
        if question:
            self.store.save_message(session_id, "user", question, user_id=user_id)
        self.store.save_message(session_id, "assistant", answer, user_id=user_id)
        if not runtime_user_entry_written and question:
            self._runtime_store.append_session_entry(
                user_id, session_id, "user", {"content": question},
            )
        self._runtime_store.append_session_entry(
            user_id, session_id, "assistant", {"content": answer},
        )
        self.store.update_title(session_id, question[:30].strip(), user_id=user_id)
        # Extract memories in background thread (non-blocking)
        threading.Thread(
            target=self._extract_memories,
            args=(question, answer, user_id),
            daemon=True,
        ).start()

    def _extract_memories(self, question: str, answer: str, user_id: str) -> None:
        """Extract key facts from conversation and store in user profile."""
        try:
            prompt = MEMORY_EXTRACTION_PROMPT.format(
                question=question[:500], answer=answer[:1000],
            )
            data = self._pipeline_llm.complete_json(
                prompt, [{"role": "user", "content": "Extract memories from this Q&A."}],
            )
            updated = False
            for mem in data.get("memories", []):
                key = mem["key"].strip()
                value = mem["value"].strip()
                mem_type = mem.get("type", "preference").strip()
                if key and value:
                    self.store.upsert_profile(user_id, key, value, memory_type=mem_type)
                    logger.debug("memory_upsert: user=%s key=%s type=%s", user_id, key, mem_type)
                    updated = True
            if updated and self.config.memory_sync_path:
                json_path = Path(self.config.memory_sync_path) / f"{user_id}.json"
                self.store.sync_profile_to_file(user_id, str(json_path))
        except Exception as exc:
            logger.warning("memory_extraction failed: %s", exc, exc_info=True)
            pass  # Memory extraction is best-effort

        # Run memory evolution (conflict resolution, consolidation, forgetting)
        try:
            report = self._evolution.evolve(user_id)
            if report.total_before != report.total_after:
                logger.info(
                    "memory_evolved: user=%s %s", user_id, str(report),
                )
        except Exception:
            pass  # Evolution is best-effort

    def _is_empty_tool_result(self, result: Any) -> bool:
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
            return False  # non-dict is likely actual data

        # Tool result entry: {"name": ..., "arguments": ..., "result": ...} or {"name": ..., "error": ...}
        # If this looks like a tool_result entry (has "name" and "result"/"error"), unwrap it.
        if "name" in result and ("result" in result or "error" in result):
            if result.get("error"):
                return True  # tool error → always empty
            return self._is_empty_tool_result(result.get("result"))

        # Plain result dict
        if result.get("error"):
            return True
        has_data = False
        for key, value in result.items():
            if key == "error":
                continue
            if isinstance(value, list):
                if len(value) > 0:
                    has_data = True
            elif isinstance(value, dict):
                if value:
                    has_data = True
            elif isinstance(value, (int, float)):
                if value != 0:
                    has_data = True
            elif value:
                has_data = True
        return not has_data

    def _stream_summarize(
        self, state: dict[str, Any], session_id: str, original_text: str, llm: LLMClient,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream the LLM summarization token by token via SSE.

        P0 guard: if ALL tool results are empty, skip LLM to prevent hallucination.
        """
        user_msg = state.get("user_message", original_text)
        tool_results = state.get("tool_results", [])
        attachments = state.get("attachments", [])

        # ── P0 guard: empty results → hardcoded message, no LLM call ──
        if tool_results and all(
            self._is_empty_tool_result(tr)
            for tr in tool_results
        ):
            msg = "抱歉，当前未能获取到相关数据。请检查查询条件后重试，或尝试使用其他关键词搜索。"
            yield {"type": "token", "content": msg}
            return msg

        system_prompt = f"""You are a helpful AI assistant. Answer the user's question using only the provided data.

Today is {_today_cn()}.

Rules:
- Use only the provided data, never fabricate
- Money is CNY unless stated otherwise, format large numbers with commas
- Keep answers concise and well-structured
- Reply in the same language as the user
- Use Markdown formatting: **bold** for key figures, bullet lists for breakdowns
- If the result contains an image URL, display it using ![description](URL) format
- Do NOT include execution process review, agent status tables, or step-by-step workflow
- If the user asks about "today" data but today is a weekend/holiday, first remind them the market is closed, then provide the latest available trading day data
- Your output is for the end user, not an internal log"""

        # Build conversation history context for multi-turn awareness
        history_context = ""
        if history:
            recent = history[-6:]  # last 3 turns
            lines = []
            for h in recent:
                role_label = "用户" if h["role"] == "user" else "助手"
                lines.append(f"[{role_label}]: {h['content'][:300]}")
            history_context = "对话历史：\n" + "\n".join(lines) + "\n\n"

        user_message_text = f"""User question: {user_msg}

Tool results:
{json.dumps(tool_results, ensure_ascii=False, indent=2)}

Please answer the user's question using only the provided data."""

        # Build multi-modal user message if attachments present
        if attachments:
            content_blocks: list[dict[str, Any]] = [
                {"type": "text", "text": history_context + user_message_text},
            ]
            for att in attachments:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{att['mime_type']};base64,{att['base64']}"},
                })
            user_message: str | list[dict[str, Any]] = content_blocks
        else:
            user_message = history_context + user_message_text

        full_answer: list[str] = []
        try:
            for token in llm.stream_complete(
                system_prompt, [{"role": "user", "content": user_message}]
            ):
                full_answer.append(token)
                yield {"type": "token", "content": token}
        except LLMError as err:
            logger.error("_stream_summarize LLM error: %s", err)
            yield {"type": "error", "message": "LLM 服务异常，请稍后重试"}
            full_answer = ["无法生成回答，请查看原始数据。"]
            yield {"type": "token", "content": full_answer[0]}
        except Exception as err:
            logger.error("_stream_summarize unexpected error: %s: %s", type(err).__name__, str(err)[:200])
            full_answer = ["无法生成回答，请查看原始数据。"]
            yield {"type": "token", "content": full_answer[0]}

        return "".join(full_answer).strip()


# ---- Module-level helpers ----


def _history_content_key(content: Any) -> str:
    """Compare text and multimodal history without including image bytes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif block.get("type") == "image_url":
                parts.append("[image]")
        return "".join(parts)
    return str(content)


def _merge_runtime_history(
    legacy: list[dict[str, Any]],
    runtime: list[dict[str, Any]],
    max_turns: int,
) -> list[dict[str, Any]]:
    """Overlay the Runtime suffix while retaining pre-migration messages."""
    if not runtime:
        return legacy[-max_turns * 2:]
    if not legacy:
        return runtime[-max_turns * 2:]

    overlap = 0
    max_overlap = min(len(legacy), len(runtime))
    for size in range(max_overlap, 0, -1):
        left = legacy[-size:]
        right = runtime[:size]
        if all(
            left[index].get("role") == right[index].get("role")
            and _history_content_key(left[index].get("content"))
            == _history_content_key(right[index].get("content"))
            for index in range(size)
        ):
            overlap = size
            break

    if overlap:
        merged = legacy[:-overlap] + runtime
    else:
        # A mixed session can have a Runtime suffix before its legacy mirror
        # is written (for example after a process interruption). If the roles
        # continue naturally, retain the legacy prefix; otherwise Runtime is
        # the only coherent transcript available.
        if legacy[-1].get("role") != runtime[0].get("role"):
            merged = legacy + runtime
        else:
            merged = runtime
    return merged[-max_turns * 2:]

def _build_default_registry(config: AgentConfig) -> AgentRegistry:
    """Build the default AgentRegistry with commander and domain agents."""
    registry = AgentRegistry(skills_base_dir=config.skills_base_dir)
    registry.register_all([
        COMMANDER,
        CODING_ASSISTANT,
        INVESTMENT_ANALYST,
        KNOWLEDGE_MANAGER,
        MEDIA_GENERATOR,
    ])
    return registry
