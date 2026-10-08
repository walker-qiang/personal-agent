"""Structured stock v3 research application; no stock rules enter Runtime Core."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...llm.errors import LLMAuthError, LLMError, LLMRateLimitError, LLMTransientError

router = APIRouter()
logger = logging.getLogger("matrix.stock_research")

_MAX_ERROR_DETAIL = 1000


def _model_error_code(error: BaseException) -> str:
    if isinstance(error, LLMAuthError):
        return "provider_authentication"
    if isinstance(error, LLMRateLimitError):
        return "provider_rate_limit"
    if isinstance(error, LLMTransientError):
        return "provider_transient"
    if isinstance(error, LLMError):
        return "provider_error"
    return "unexpected_model_error"


def _safe_error_detail(error: BaseException) -> str:
    detail = str(error).strip() or error.__class__.__name__
    detail = re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[redacted]", detail)
    detail = re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "[redacted]", detail)
    return detail[:_MAX_ERROR_DETAIL]


def _limits() -> dict[str, int]:
    names = {
        "input_tokens": "MATRIX_STOCK_V3_INPUT_TOKENS",
        "output_tokens": "MATRIX_STOCK_V3_OUTPUT_TOKENS",
        "total_tokens": "MATRIX_STOCK_V3_TOTAL_TOKENS",
    }
    limits: dict[str, int] = {}
    for key, env in names.items():
        try:
            limits[key] = max(0, int(os.environ.get(env, "0")))
        except ValueError:
            limits[key] = 0
    return limits


def _client(request: Request) -> Any:
    return request.app.state.chat._get_llm(None)

def _usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    input_tokens = value.get("input_tokens")
    output_tokens = value.get("output_tokens")
    if (not isinstance(input_tokens, int) or isinstance(input_tokens, bool)
            or not isinstance(output_tokens, int) or isinstance(output_tokens, bool)
            or input_tokens < 0 or output_tokens < 0):
        return None
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _estimate_input_tokens(value: str) -> int:
    """Conservative admission estimate; provider usage remains authoritative."""
    ascii_units = sum(1 for char in value if ord(char) < 128)
    non_ascii_units = len(value) - ascii_units
    return (ascii_units + 3) // 4 + non_ascii_units * 2


def _structured_result(value: Any) -> bool:
    return (isinstance(value, dict)
            and isinstance(value.get("episode"), dict)
            and isinstance(value.get("thesis_patch"), list)
            and isinstance(value.get("evidence_candidates"), list))


def _valid_numeric_facts(value: Any, *, require_nonempty: bool) -> bool:
    if not isinstance(value, list) or (require_nonempty and not value):
        return False
    allowed = {"field", "value", "comparative_value", "period", "currency", "unit"}
    for fact in value:
        if not isinstance(fact, dict) or set(fact) - allowed:
            return False
        required = ("field", "value", "period", "currency", "unit")
        if not all(isinstance(fact.get(key), str) for key in required):
            return False
        if "comparative_value" in fact and not isinstance(fact["comparative_value"], str):
            return False
    return True


def _validate_portfolio_context(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("portfolio context must be structured")
    if set(value) - {"version", "target_holding", "transactions", "allocation"}:
        raise ValueError("portfolio context contains unauthorized fields")
    if not isinstance(value.get("version"), int) or value["version"] != 1:
        raise ValueError("portfolio context version is invalid")
    target = value.get("target_holding")
    if target is not None:
        if not isinstance(target, dict):
            raise ValueError("target holding must be structured")
        allowed = {
            "code", "name", "currency", "allocation_bucket", "as_of",
            "current_value_yuan", "cost_basis_yuan", "quantity",
            "current_weight_pct", "position_status", "transaction_count",
        }
        if set(target) - allowed or not all(
            isinstance(target.get(key), str)
            for key in ("code", "name", "currency", "allocation_bucket", "position_status")
        ):
            raise ValueError("target holding contains unauthorized fields")
    transactions = value.get("transactions")
    if not isinstance(transactions, list) or len(transactions) > 50:
        raise ValueError("portfolio transactions are invalid")
    transaction_fields = {
        "occurred_at", "type", "quantity", "unit_price_yuan",
        "amount_yuan", "fee_yuan", "tax_yuan", "currency",
    }
    for transaction in transactions:
        if not isinstance(transaction, dict) or set(transaction) - transaction_fields:
            raise ValueError("portfolio transaction contains unauthorized fields")
    allocation = value.get("allocation")
    if not isinstance(allocation, list) or len(allocation) > 20:
        raise ValueError("portfolio allocation is invalid")
    allocation_fields = {"allocation_bucket", "current_pct", "target_pct", "delta_pct"}
    for bucket in allocation:
        if not isinstance(bucket, dict) or set(bucket) - allocation_fields:
            raise ValueError("portfolio allocation contains unauthorized fields")


@router.get("/api/research/stock.v3/capabilities")
async def capabilities(request: Request) -> dict[str, Any]:
    limits = _limits()
    client = _client(request)
    return {
        "protocol": "stock-research-v3",
        "supported": bool(all(limits.values()) and limits["output_tokens"] <= limits["total_tokens"]
                          and request.app.state.config.llm_available
                          and callable(getattr(client, "complete_json_budgeted", None))),
        "model": getattr(client, "model", ""),
        "limits": limits,
        "max_new_documents": 5,
        "max_synthesis_calls": 1,
        "max_repair_calls": 1,
    }


@router.post("/api/research/stock.v3")
async def stock_research(request: Request):
    cap = await capabilities(request)
    if not cap["supported"]:
        return JSONResponse({"error": "stock v3 model budget is not configured"}, status_code=503)
    try:
        body = await request.json()
        plan, context, documents = body["plan"], body["context"], body["documents"]
        if not isinstance(plan, dict) or not isinstance(context, dict) or not isinstance(documents, list):
            raise ValueError("plan, context and documents must be structured")
        if plan.get("object_type") != "stock" or plan.get("model") != cap["model"]:
            raise ValueError("stock plan model or object type mismatch")
        if not plan.get("confirmed_at") or not plan.get("plan_id") or not plan.get("input_hash"):
            raise ValueError("confirmed plan, plan_id and input_hash are required")
        if documents != plan.get("documents") or context.get("question") != plan.get("question"):
            raise ValueError("research context or documents differ from the confirmed plan")
        budget = plan["budget"]
        if not isinstance(budget, dict) or any(
            not isinstance(budget.get(key), int) or budget[key] <= 0 or budget[key] > cap["limits"][key]
            for key in cap["limits"]
        ):
            raise ValueError("budget exceeds server limits or is missing")
        facts_only = plan.get("facts_only") is True
        if facts_only:
            if (set(context) - {"question", "subject", "calculated_metrics", "pending_questions"}
                    or set(context.get("subject", {})) != {"name"}):
                raise ValueError("numeric-only context must not include Thesis, notes or private source material")
            if context.get("calculated_metrics", []) != plan.get("calculated_metrics", []):
                raise ValueError("calculated metrics differ from the confirmed plan")
            if context.get("pending_questions", []) != plan.get("pending_questions", []):
                raise ValueError("pending questions differ from the confirmed plan")
        else:
            allowed_context = {
                "question", "subject", "thesis", "portfolio_context", "pending_questions",
            }
            if set(context) - allowed_context or set(context.get("subject", {})) != {"code", "name"}:
                raise ValueError("research context contains unauthorized fields")
            if not isinstance(context.get("thesis"), dict):
                raise ValueError("selected Thesis context is required")
            if context.get("pending_questions", []) != plan.get("pending_questions", []):
                raise ValueError("pending questions differ from the confirmed plan")
            if context.get("portfolio_context") != plan.get("portfolio_context"):
                raise ValueError("portfolio context differs from the confirmed plan")
            _validate_portfolio_context(context.get("portfolio_context"))
        if len(documents) > 5:
            raise ValueError("at most five documents are allowed")
        for doc in documents:
            if facts_only:
                facts = doc.get("numeric_facts")
                if set(doc) != {"numeric_facts"} or not _valid_numeric_facts(facts, require_nonempty=True):
                    raise ValueError("numeric-only document must contain only authorized structured values")
            else:
                if (not isinstance(doc, dict)
                        or set(doc) - {"source_id", "excerpt", "numeric_facts"}
                        or not isinstance(doc.get("source_id"), str)
                        or not doc["source_id"]):
                    raise ValueError("document excerpt DTO is invalid")
                excerpt = doc.get("excerpt", "")
                if not isinstance(excerpt, str):
                    raise ValueError("document excerpt DTO is invalid")
                if "numeric_facts" in doc and not _valid_numeric_facts(
                    doc["numeric_facts"], require_nonempty=False
                ):
                    raise ValueError("document numeric facts DTO is invalid")
                if not excerpt and not doc.get("numeric_facts"):
                    raise ValueError("document DTO has neither excerpt nor structured facts")
        system = (
            "Return only a JSON object with exactly these top-level keys: "
            '{"episode":{"summary":"","pending_questions":[]},"thesis_patch":[],'
            '"evidence_candidates":[]}. episode must be an object; the other two '
            "must be arrays. If documents is empty or plan.facts_only is true, leave both arrays empty and "
            "put the analysis and unresolved questions in episode. "
            "Propose only scoped changes with source id and fact id when evidence exists; "
            "never claim verification. "
            "Do not perform additional tool calls or external search; use only the "
            "verified documents, selected Thesis claims and portfolio context supplied by personal-os. "
            "Use portfolio context only to assess portfolio fit, concentration, cost basis and transaction history; "
            "distinguish those user-provided facts from company evidence and do not invent missing values. "
            "No trading actions or file writes."
        )
        if facts_only:
            system += (
                " API calculated_metrics are authoritative; do not calculate financial figures. "
                "Keep summary qualitative, without amounts, percentages or ratios. "
                "If citing a calculated value, use episode.numeric_claims as an array of "
                '{"metric_id":"the supplied metric id","value":"the exact supplied value"}. '
                "Do not claim a value for a not_computable metric."
            )
        # The application protocol has already validated the boundary. The model
        # receives only the explicit research DTO, never the original plan envelope.
        if facts_only:
            model_request = {
                "subject": context["subject"],
                "question": context["question"],
                "documents": documents,
                "calculated_metrics": context.get("calculated_metrics", []),
                "pending_questions": context.get("pending_questions", []),
            }
        else:
            model_request = {
                "subject": context["subject"],
                "question": context["question"],
                "thesis": context["thesis"],
                "portfolio_context": context["portfolio_context"],
                "documents": documents,
                "pending_questions": context.get("pending_questions", []),
            }
        messages = [{"role": "user", "content": json.dumps(
            model_request, ensure_ascii=False,
        )}]
        # Reserve framing explicitly, but avoid treating every UTF-8 byte as a
        # token. Provider-reported usage is checked again after each request.
        input_bound = (
            _estimate_input_tokens(system)
            + _estimate_input_tokens(messages[0]["content"])
            + 2048
        )
        output = budget["output_tokens"]
        if input_bound > budget["input_tokens"] or input_bound + output > budget["total_tokens"]:
            raise ValueError("synthesis exceeds cumulative token ceiling")
        client = _client(request)
        result, usage = await asyncio.to_thread(client.complete_json_budgeted, system, messages, output)
        calls = 1
        if not isinstance(result, dict):
            raise ValueError("synthesis result must be a JSON object")
        if not _structured_result(result):
            repair_system = (
                system + " Repair the draft into the required top-level structure and types. "
                "Preserve supported analysis in episode; use empty arrays when evidence "
                "is absent. Return only the corrected JSON object."
            )
            repair_input = json.dumps(
                {"request": model_request, "draft": result}, ensure_ascii=False,
            )
            repair_bound = (
                _estimate_input_tokens(repair_system)
                + _estimate_input_tokens(repair_input)
                + 2048
            )
            if _usage(usage) is None or repair_bound > budget["input_tokens"] or (
                input_bound + output + repair_bound + output > budget["total_tokens"]
            ):
                raise ValueError("repair requires known usage and remaining cumulative budget")
            result, repair_usage = await asyncio.to_thread(
                client.complete_json_budgeted,
                repair_system, [{"role": "user", "content": repair_input}], output,
            )
            calls = 2
            usage = {"synthesis": usage, "repair": repair_usage}
        actual = [_usage(usage)] if calls == 1 else [
            _usage(usage["synthesis"]), _usage(usage["repair"])]
        usage_complete = all(part is not None for part in actual)
        if not _structured_result(result):
            return JSONResponse(
                {"error": "v3 result incomplete after repair", "usage": usage,
                 "usage_complete": usage_complete, "model_calls": calls},
                status_code=422,
            )
        if usage_complete and (
            any(part["input_tokens"] > budget["input_tokens"]
                or part["output_tokens"] > output for part in actual)
            or sum(part["input_tokens"] + part["output_tokens"] for part in actual) > budget["total_tokens"]
        ):
            raise ValueError("provider usage exceeded confirmed budget; do not retry")
        return {"protocol": cap["protocol"], "plan_id": plan["plan_id"],
                "result": result, "usage": usage, "usage_complete": usage_complete,
                "model_calls": calls}
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except LLMError as exc:
        error_code = _model_error_code(exc)
        detail = _safe_error_detail(exc)
        logger.exception(
            "stock v3 model request failed plan_id=%s model=%s documents=%d "
            "facts_only=%s error_code=%s detail=%s",
            plan.get("plan_id", ""),
            cap.get("model", ""),
            len(documents) if isinstance(documents, list) else -1,
            facts_only,
            error_code,
            detail,
        )
        return JSONResponse(
            {
                "error": "stock research model request failed",
                "error_code": error_code,
                "detail": detail,
                "no_automatic_retry": True,
            },
            status_code=502,
        )
    except Exception as exc:
        error_code = _model_error_code(exc)
        detail = _safe_error_detail(exc)
        logger.exception(
            "stock v3 request failed unexpectedly plan_id=%s model=%s "
            "documents=%d facts_only=%s error_code=%s detail=%s",
            plan.get("plan_id", ""),
            cap.get("model", ""),
            len(documents) if isinstance(documents, list) else -1,
            facts_only,
            error_code,
            detail,
        )
        return JSONResponse(
            {
                "error": "stock research request failed",
                "error_code": error_code,
                "detail": detail,
                "no_automatic_retry": True,
            },
            status_code=502,
        )
