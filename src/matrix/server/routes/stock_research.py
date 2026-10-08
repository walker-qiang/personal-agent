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
_MAX_NEW_DOCUMENTS = 12
_MAX_SYNTHESIS_CALLS = 2
_MAX_REPAIR_CALLS = 1


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


_V3_RESULT_KEYS = {"episode", "thesis_patch", "evidence_candidates"}
_V3_EPISODE_KEYS = {"summary", "pending_questions", "numeric_claims"}
_V3_PATCH_KEYS = {
    "entity", "id", "field", "before", "after", "change_type",
    "reason", "evidence_refs",
}
_V3_EVIDENCE_KEYS = {"source_id", "fact_id", "limited_quote", "statement"}
_V3_CHANGE_TYPES = {"strengthened", "weakened", "falsified", "retired", "calibrated"}


def _v3_contract_prompt(*, facts_only: bool) -> str:
    numeric_claims = (
        ', "numeric_claims":[{"metric_id":"<metric id>","value":"<exact supplied value>"}]'
        if facts_only else ""
    )
    episode_example = (
        '{"summary":"<analysis>","pending_questions":["<question>"]'
        + numeric_claims
        + "}"
    )
    return (
        "The stock-research-v3 result contract is strict. Return exactly this JSON shape "
        "(no additional keys and never use patch_type, evidence_direction, verification, "
        "summary-as-fact, fact, or claims fields): "
        '{"episode":' + episode_example + ","
        '"thesis_patch":[{"entity":"claim","id":"<claim id>","field":"confidence",'
        '"before":"low","after":"medium","change_type":"strengthened",'
        '"reason":"<why this scoped change is supported>",'
        '"evidence_refs":["<source_id>:<fact_id>"]}],'
        '"evidence_candidates":[{"source_id":"<supplied source id>",'
        '"fact_id":"<stable fact id>",'
        '"limited_quote":"<exact contiguous quote from the supplied excerpt, <=300 chars>",'
        '"statement":"<what the quote establishes>"}]}. '
        "episode.summary may be empty, pending_questions must be an array of strings, "
        "and arrays may be empty. A thesis_patch requires a matching evidence_candidate "
        "and must use evidence_refs in the exact source_id:fact_id format. "
        "Only use entity=claim or entity=decision; only use the supplied claim ids and "
        "allowed Thesis fields. Do not assert that a source or fact is verified; "
        "personal-os performs verification and writeback checks."
    )


def _v3_result_errors(
    value: Any,
    documents: list[dict[str, Any]],
    *,
    facts_only: bool,
) -> list[str]:
    """Validate the complete v3 DTO before crossing the Agent boundary."""

    errors: list[str] = []
    if not isinstance(value, dict):
        return ["result must be a JSON object"]
    extra = set(value) - _V3_RESULT_KEYS
    missing = _V3_RESULT_KEYS - set(value)
    if extra:
        errors.append(f"result has unsupported keys: {sorted(extra)}")
    if missing:
        errors.append(f"result is missing keys: {sorted(missing)}")

    episode = value.get("episode")
    if not isinstance(episode, dict):
        errors.append("episode must be an object")
    else:
        episode_extra = set(episode) - _V3_EPISODE_KEYS
        if episode_extra:
            errors.append(f"episode has unsupported keys: {sorted(episode_extra)}")
        if "summary" in episode and not isinstance(episode["summary"], str):
            errors.append("episode.summary must be a string")
        pending = episode.get("pending_questions", [])
        if not isinstance(pending, list) or any(
            not isinstance(item, str) or not item.strip() for item in pending
        ):
            errors.append("episode.pending_questions must be an array of non-empty strings")
        numeric_claims = episode.get("numeric_claims")
        if numeric_claims is not None:
            if not facts_only:
                errors.append("episode.numeric_claims is only allowed for facts_only plans")
            elif not isinstance(numeric_claims, list) or any(
                not isinstance(item, dict)
                or set(item) != {"metric_id", "value"}
                or not isinstance(item["metric_id"], str)
                or not isinstance(item["value"], str)
                for item in numeric_claims
            ):
                errors.append(
                    "episode.numeric_claims must contain only metric_id and value strings"
                )

    source_excerpts = {
        str(document.get("source_id")): str(document.get("excerpt") or "")
        for document in documents
    }
    candidates = value.get("evidence_candidates")
    candidate_refs: set[str] = set()
    if not isinstance(candidates, list):
        errors.append("evidence_candidates must be an array")
    else:
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                errors.append(f"evidence_candidates[{index}] must be an object")
                continue
            if set(candidate) != _V3_EVIDENCE_KEYS:
                errors.append(
                    f"evidence_candidates[{index}] must use exactly "
                    f"{sorted(_V3_EVIDENCE_KEYS)}"
                )
                continue
            if not all(isinstance(candidate[key], str) and candidate[key].strip()
                       for key in _V3_EVIDENCE_KEYS):
                errors.append(f"evidence_candidates[{index}] contains an empty or non-string field")
                continue
            quote = candidate["limited_quote"]
            source_id = candidate["source_id"]
            if len(quote) > 300:
                errors.append(f"evidence_candidates[{index}].limited_quote exceeds 300 characters")
            if source_id not in source_excerpts:
                errors.append(f"evidence_candidates[{index}] references an unknown source_id")
            elif quote not in source_excerpts[source_id]:
                errors.append(
                    f"evidence_candidates[{index}].limited_quote is not an exact excerpt"
                )
            candidate_refs.add(f"{source_id}:{candidate['fact_id']}")

    patches = value.get("thesis_patch")
    if not isinstance(patches, list):
        errors.append("thesis_patch must be an array")
    else:
        for index, patch in enumerate(patches):
            if not isinstance(patch, dict):
                errors.append(f"thesis_patch[{index}] must be an object")
                continue
            if set(patch) != _V3_PATCH_KEYS:
                errors.append(
                    f"thesis_patch[{index}] must use exactly {sorted(_V3_PATCH_KEYS)}"
                )
                continue
            if patch["entity"] not in {"claim", "decision"}:
                errors.append(f"thesis_patch[{index}].entity is not a V3 entity")
            if not isinstance(patch["id"], str) or not isinstance(patch["field"], str):
                errors.append(f"thesis_patch[{index}] id and field must be strings")
            if not isinstance(patch["before"], str) or not isinstance(patch["after"], str):
                errors.append(f"thesis_patch[{index}] before and after must be strings")
            if patch["change_type"] not in _V3_CHANGE_TYPES:
                errors.append(f"thesis_patch[{index}].change_type is not a V3 change type")
            if not isinstance(patch["reason"], str) or not patch["reason"].strip():
                errors.append(f"thesis_patch[{index}].reason must be non-empty")
            refs = patch["evidence_refs"]
            if not isinstance(refs, list) or not refs or any(
                not isinstance(ref, str) or ref not in candidate_refs for ref in refs
            ):
                errors.append(
                    f"thesis_patch[{index}].evidence_refs must reference supplied candidates"
                )
    return errors


def _usage_parts(value: Any) -> list[dict[str, int]]:
    direct = _usage(value)
    if direct is not None:
        return [direct]
    if not isinstance(value, dict):
        return []
    parts: list[dict[str, int]] = []
    for nested in value.values():
        parts.extend(_usage_parts(nested))
    return parts


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
        "max_new_documents": _MAX_NEW_DOCUMENTS,
        "max_synthesis_calls": _MAX_SYNTHESIS_CALLS,
        "max_repair_calls": _MAX_REPAIR_CALLS,
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
        scope = plan.get("scope") if isinstance(plan.get("scope"), dict) else {}
        max_documents = scope.get("max_new_documents", 5)
        max_synthesis_calls = scope.get("max_synthesis_calls", 1)
        max_repair_calls = scope.get("max_repair_calls", 1)
        if (not isinstance(max_documents, int) or isinstance(max_documents, bool)
                or not 1 <= max_documents <= cap["max_new_documents"]):
            raise ValueError("research document limit is outside the server capability")
        if (not isinstance(max_synthesis_calls, int) or isinstance(max_synthesis_calls, bool)
                or not 1 <= max_synthesis_calls <= cap["max_synthesis_calls"]):
            raise ValueError("synthesis call limit is outside the server capability")
        if (not isinstance(max_repair_calls, int) or isinstance(max_repair_calls, bool)
                or not 0 <= max_repair_calls <= cap["max_repair_calls"]):
            raise ValueError("repair call limit is outside the server capability")
        if len(documents) > max_documents:
            raise ValueError(f"at most {max_documents} documents are allowed")
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
        system += " " + _v3_contract_prompt(facts_only=facts_only)
        if facts_only:
            system += (
                " API calculated_metrics are authoritative; do not calculate financial figures. "
                "Keep summary qualitative, without amounts, percentages or ratios. "
                "If citing a calculated value, use episode.numeric_claims as an array of "
                '{"metric_id":"the supplied metric id","value":"the exact supplied value"}. '
                "Do not claim a value for a not_computable metric."
            )
        evidence_system = system + (
            " First build an evidence map: identify the most decision-relevant new facts, "
            "contradictions, unchanged claims and unresolved questions. Keep every proposed "
            "change tied to the supplied source id and fact id."
        )
        synthesis_system = system + (
            " Act as the final decision synthesizer. Review the evidence draft supplied in "
            "the request, discard unsupported claims, preserve valid citations, and make the "
            "summary explicitly state what changed, what did not change, the strongest "
            "counter-evidence and the next monitoring question."
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
            _estimate_input_tokens(evidence_system if max_synthesis_calls > 1 and not facts_only else system)
            + _estimate_input_tokens(messages[0]["content"])
            + 2048
        )
        output = budget["output_tokens"]
        wants_second_synthesis = max_synthesis_calls > 1 and not facts_only
        second_reserve = 0
        if wants_second_synthesis:
            second_reserve = (
                _estimate_input_tokens(synthesis_system)
                + _estimate_input_tokens(json.dumps(
                    {"request": model_request, "evidence_draft": ""},
                    ensure_ascii=False,
                ))
                + output
                + 2048
            )
        if input_bound > budget["input_tokens"] or (
            input_bound + output + second_reserve > budget["total_tokens"]
        ):
            raise ValueError("synthesis exceeds cumulative token ceiling")
        client = _client(request)
        result, first_usage = await asyncio.to_thread(
            client.complete_json_budgeted,
            evidence_system if wants_second_synthesis else system,
            messages,
            output,
        )
        calls = 1
        usage: Any = first_usage
        if wants_second_synthesis:
            first_parts = _usage_parts(first_usage)
            if not first_parts:
                raise ValueError("second synthesis requires provider usage for the first call")
            draft_request = {
                "request": model_request,
                "evidence_draft": result,
            }
            second_messages = [{"role": "user", "content": json.dumps(
                draft_request, ensure_ascii=False,
            )}]
            second_bound = (
                _estimate_input_tokens(synthesis_system)
                + _estimate_input_tokens(second_messages[0]["content"])
                + 2048
            )
            consumed = sum(part["input_tokens"] + part["output_tokens"] for part in first_parts)
            if second_bound > budget["input_tokens"] or (
                consumed + second_bound + output > budget["total_tokens"]
            ):
                raise ValueError("second synthesis exceeds remaining cumulative token ceiling")
            result, second_usage = await asyncio.to_thread(
                client.complete_json_budgeted,
                synthesis_system,
                second_messages,
                output,
            )
            usage = {"evidence": first_usage, "synthesis": second_usage}
            calls = 2
        if not isinstance(result, dict):
            raise ValueError("synthesis result must be a JSON object")
        result_errors = _v3_result_errors(result, documents, facts_only=facts_only)
        if result_errors and max_repair_calls > 0:
            repair_system = (
                system + " Repair the draft into the exact stock-research-v3 contract. "
                "Preserve supported analysis in episode, discard unsupported patch or "
                "evidence fields, and use empty arrays when a supported citation cannot "
                "be produced. Return only the corrected JSON object."
            )
            repair_input = json.dumps(
                {"request": model_request, "draft": result}, ensure_ascii=False,
            )
            repair_bound = (
                _estimate_input_tokens(repair_system)
                + _estimate_input_tokens(repair_input)
                + 2048
            )
            usage_parts = _usage_parts(usage)
            consumed = sum(part["input_tokens"] + part["output_tokens"] for part in usage_parts)
            if not usage_parts or repair_bound > budget["input_tokens"] or (
                consumed + repair_bound + output > budget["total_tokens"]
            ):
                raise ValueError("repair requires known usage and remaining cumulative budget")
            result, repair_usage = await asyncio.to_thread(
                client.complete_json_budgeted,
                repair_system, [{"role": "user", "content": repair_input}], output,
            )
            calls += 1
            if isinstance(usage, dict) and _usage(usage) is None:
                usage["repair"] = repair_usage
            else:
                usage = {"synthesis": usage, "repair": repair_usage}
            result_errors = _v3_result_errors(result, documents, facts_only=facts_only)
        actual = _usage_parts(usage)
        usage_complete = len(actual) == calls
        if result_errors:
            return JSONResponse(
                {"error": "v3 result contract invalid after repair",
                 "details": result_errors[:20], "usage": usage,
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
