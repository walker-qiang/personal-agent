"""Structured stock v3 research application; no stock rules enter Runtime Core."""

from __future__ import annotations

import json
import os
import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


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


def _structured_result(value: Any) -> bool:
    return (isinstance(value, dict)
            and isinstance(value.get("episode"), dict)
            and isinstance(value.get("thesis_patch"), list)
            and isinstance(value.get("evidence_candidates"), list))


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
            if (set(context) - {"question", "subject", "calculated_metrics"}
                    or set(context.get("subject", {})) != {"name"}):
                raise ValueError("numeric-only context must not include Thesis, notes or private questions")
            if context.get("calculated_metrics", []) != plan.get("calculated_metrics", []):
                raise ValueError("calculated metrics differ from the confirmed plan")
        if len(documents) > 5:
            raise ValueError("at most five documents are allowed")
        for doc in documents:
            if not isinstance(doc, dict) or not isinstance(doc.get("permissions"), dict) or not doc.get("content_hash"):
                raise ValueError("document permissions or hash are invalid")
            if facts_only:
                facts = doc.get("numeric_facts")
                if (not doc["permissions"].get("external_model_facts")
                        or doc["permissions"].get("external_model_excerpt")
                        or doc.get("excerpt") or not isinstance(facts, list) or not facts
                        or any(not isinstance(fact, dict) or
                               not all(isinstance(fact.get(key), str) for key in
                                       ("field", "value", "period", "currency", "unit"))
                               for fact in facts)):
                    raise ValueError("numeric-only document must contain authorized structured values and no excerpt")
            elif (not doc["permissions"].get("external_model_excerpt")
                  or not isinstance(doc.get("excerpt"), str)):
                raise ValueError("document excerpt permission is invalid")
        system = (
            "Return only a JSON object with exactly these top-level keys: "
            '{"episode":{"summary":"","pending_questions":[]},"thesis_patch":[],'
            '"evidence_candidates":[]}. episode must be an object; the other two '
            "must be arrays. If documents is empty or plan.facts_only is true, leave both arrays empty and "
            "put the analysis and unresolved questions in episode. "
            "Propose only scoped changes with source id and fact id when evidence exists; "
            "never claim verification. "
            "No tools, external search, trading actions or file writes."
        )
        if facts_only:
            system += (
                " API calculated_metrics are authoritative; do not calculate financial figures. "
                "Keep summary qualitative, without amounts, percentages or ratios. "
                "If citing a calculated value, use episode.numeric_claims as an array of "
                '{"metric_id":"the supplied metric id","value":"the exact supplied value"}. '
                "Do not claim a value for a not_computable metric."
            )
        messages = [{"role": "user", "content": json.dumps(
            {"plan": plan, "context": context, "documents": documents}, ensure_ascii=False,
        )}]
        # Each UTF-8 byte is charged as a token, plus framing reserve.
        input_bound = len(system.encode()) + len(messages[0]["content"].encode()) + 2048
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
            repair_input = json.dumps({"plan": plan, "draft": result}, ensure_ascii=False)
            repair_bound = len(repair_system.encode()) + len(repair_input.encode()) + 2048
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
    except Exception:
        return JSONResponse({"error": "stock research request failed; do not automatically retry"},
                            status_code=502)
