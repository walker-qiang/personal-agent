"""Read-only tools backed by the personal-os business/data layer."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
import uuid
from typing import Any

from ..base import ToolDefinition, tool_error
from ..principal import current_principal
from ..registry import ToolRegistry


def _get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    base_url = os.environ.get("PERSONAL_OS_API_URL", "http://127.0.0.1:7001").rstrip("/")
    query = urllib.parse.urlencode({key: value for key, value in params.items() if value not in (None, "")})
    url = f"{base_url}{path}"
    if query:
        url += f"?{query}"
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "personal-agent/personal-os-tools"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"error": f"personal-os tool request failed: {exc}"}
    if isinstance(payload, dict) and "error" in payload:
        return {"error": str(payload["error"])}
    return payload if isinstance(payload, dict) else {"result": payload}


def _post(path: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    base_url = os.environ.get("PERSONAL_OS_API_URL", "http://127.0.0.1:7001").rstrip("/")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "personal-agent/writeback-tools",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"error": f"personal-os writeback request failed: {exc}"}
    if isinstance(body, dict) and body.get("error"):
        return {
            "error": str(body["error"]),
            "error_code": body.get("error_code", "WRITEBACK_FAILED"),
        }
    return body if isinstance(body, dict) else {"result": body}


def writeback_prepare(
    operation: str,
    payload: dict[str, Any],
    idempotency_key: str = "",
) -> dict[str, Any]:
    owner_id, session_id, _, _ = current_principal()
    return _post(
        "/api/writeback/plan",
        {
            "operation": operation,
            "payload": payload,
            "owner_id": owner_id,
            "session_id": session_id,
            "request_id": "req_" + uuid.uuid4().hex,
            "idempotency_key": idempotency_key,
        },
    )


def writeback_execute(plan: dict[str, Any], plan_hash: str) -> dict[str, Any]:
    owner_id, session_id, mode, allow_external_effects = current_principal()
    if mode != "writeback" or not allow_external_effects:
        return {
            "error": "writeback execution is blocked outside approved writeback mode",
            "error_code": "APPROVAL_REQUIRED",
        }
    return _post(
        "/api/writeback/execute",
        {
            "plan": plan,
            "plan_hash": plan_hash,
            "owner_id": owner_id,
            "session_id": session_id,
            "request_id": "req_" + uuid.uuid4().hex,
        },
        timeout=60.0,
    )


def market_quote(code: str) -> dict[str, Any]:
    if not str(code).strip():
        return tool_error("personal_os.market_quote", "查询行情", "code is required")
    return _get("/api/tools/market/quote", {"code": code})


def financials(code: str, periods: int = 4) -> dict[str, Any]:
    if not str(code).strip():
        return tool_error("personal_os.financials", "查询财报", "code is required")
    return _get("/api/tools/market/financials", {"code": code, "num": periods})


def profile(code: str) -> dict[str, Any]:
    if not str(code).strip():
        return tool_error("personal_os.profile", "查询公司简况", "code is required")
    return _get("/api/tools/market/profile", {"code": code})


def dividend(code: str, years: int = 5) -> dict[str, Any]:
    if not str(code).strip():
        return tool_error("personal_os.dividend", "查询分红", "code is required")
    return _get("/api/tools/market/dividend", {"code": code, "years": years})


def valuation(code: str) -> dict[str, Any]:
    if not str(code).strip():
        return tool_error("personal_os.valuation", "计算估值", "code is required")
    return _get("/api/tools/market/valuation", {"code": code})


def peers(code: str) -> dict[str, Any]:
    if not str(code).strip():
        return tool_error("personal_os.peers", "查询同业估值", "code is required")
    return _get("/api/tools/market/peers", {"code": code})


def research_context(code: str = "", name: str = "") -> dict[str, Any]:
    if not str(code).strip() and not str(name).strip():
        return tool_error("personal_os.research_context", "读取研究上下文", "code or name is required")
    return _get("/api/tools/research/context", {"code": code, "name": name})


def information_search(query: str = "", limit: int = 10) -> dict[str, Any]:
    return _get("/api/tools/information/search", {"q": query, "limit": limit})


def web_fetch(url: str) -> dict[str, Any]:
    if not str(url).strip():
        return tool_error("personal_os.web_fetch", "抓取网页", "url is required")
    return _get("/api/tools/web/fetch", {"url": url})


def register_all(registry: ToolRegistry) -> None:
    registry.register(
        ToolDefinition(
            name="personal_os.market_quote",
            description="通过 personal-os 查询指定股票、ETF 或指数的最新行情和行情时间。只读。",
            input_schema={
                "type": "object",
                "properties": {"code": {"type": "string", "description": "市场代码，如 sz000858、hk00700、usVTI.AM"}},
                "required": ["code"],
            },
            handler=market_quote,
            capabilities=["market_data"],
        )
    )
    registry.register(
        ToolDefinition(
            name="personal_os.financials",
            description="通过 personal-os 查询指定股票或 ETF 的多期财务数据。只读，必须记录数据期间和 provider。",
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "periods": {"type": "integer"},
                },
                "required": ["code"],
            },
            handler=financials,
            capabilities=["financial_data", "source_provenance"],
        )
    )
    registry.register(
        ToolDefinition(
            name="personal_os.profile",
            description="通过 personal-os 查询公司主营、行业、官网、上市日期等简况。只读。",
            input_schema={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            handler=profile,
            capabilities=["company_profile", "source_provenance"],
        )
    )
    registry.register(
        ToolDefinition(
            name="personal_os.dividend",
            description="通过 personal-os 查询多年分红、除权除息和每股现金分红数据。只读。",
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "years": {"type": "integer"},
                },
                "required": ["code"],
            },
            handler=dividend,
            capabilities=["dividend_data", "source_provenance"],
        )
    )
    registry.register(
        ToolDefinition(
            name="personal_os.valuation",
            description="通过 personal-os 计算当前价格相对最新报告期 EPS 和每股净资产的估值指标。结果为估算值，包含报告期和计算口径。",
            input_schema={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            handler=valuation,
            capabilities=["valuation_data", "source_provenance"],
        )
    )
    registry.register(
        ToolDefinition(
            name="personal_os.peers",
            description="通过 personal-os 查询维护过的可比公司注册表，并批量计算当前估值。不得自行扩展同业名单。",
            input_schema={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            handler=peers,
            capabilities=["peer_comparison", "valuation_data", "source_provenance"],
        )
    )
    registry.register(
        ToolDefinition(
            name="personal_os.research_context",
            description="读取 personal-assets 中该标的已经保存的 schema v2 研究记录。只读。",
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "name": {"type": "string"},
                },
            },
            handler=research_context,
            capabilities=["research_context"],
        )
    )
    registry.register(
        ToolDefinition(
            name="personal_os.information_search",
            description="搜索 personal-os 已注册的高质量 RSS/Atom 信息源，返回标题、来源、摘要、发布时间和原文链接。只读。",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
            handler=information_search,
            capabilities=["web_search", "source_provenance"],
        )
    )
    registry.register(
        ToolDefinition(
            name="personal_os.web_fetch",
            description="通过 personal-os 抓取指定官方网页或公告页面正文。只读，必须提供完整 http(s) URL。",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            handler=web_fetch,
            capabilities=["web_fetch", "source_provenance"],
        )
    )
    registry.register(
        ToolDefinition(
            name="writeback.prepare",
            description=(
                "生成结构化 durable 写入计划，只校验和预览，不写文件、不提交 Git。"
                "当前只支持 finance.snapshot.create。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["finance.snapshot.create"]},
                    "payload": {"type": "object"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["operation", "payload"],
            },
            handler=writeback_prepare,
            capabilities=["writeback_plan"],
            recovery_policy="idempotent",
        )
    )
    registry.register(
        ToolDefinition(
            name="writeback.execute_plan",
            description=(
                "执行已经生成的 durable 写入计划。该工具有外部副作用，"
                "仅在 writeback 模式且通过 Runtime approval 后执行。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "plan": {"type": "object"},
                    "plan_hash": {"type": "string"},
                },
                "required": ["plan", "plan_hash"],
            },
            handler=writeback_execute,
            capabilities=["durable_writeback"],
            requires_approval=True,
            recovery_policy="idempotent",
            side_effect=True,
        )
    )
