"""Deterministic decision DTO projection; no model calls or investment thresholds."""

from __future__ import annotations

from datetime import date
from typing import Any

DECISION_VERSION = "stock-decision-v2"
LEGACY_DECISION_VERSION = "stock-decision-v1"
_ACTIONS = {"buy": "consider_buy", "hold": "wait", "sell": "avoid", "watch": "wait"}


def _new_money(action: Any) -> str:
    return _ACTIONS.get(action, "") if isinstance(action, str) else ""


def is_deterministic_decision(context: Any) -> bool:
    return isinstance(context, dict) and context.get("version") == DECISION_VERSION


def validate_decision_policy(
    policy: dict[str, Any], valuation_context: dict[str, Any] | None,
    market_context: dict[str, Any] | None,
) -> None:
    required = {"version", "objective", "required_metric_ids", "as_of",
                "valuation_methods", "review_triggers"}
    if not required <= set(policy) or set(policy) - required - {"latest_report_period"}:
        raise ValueError("decision-v2 requires program-owned method and review-trigger indexes")
    try:
        as_of = date.fromisoformat(policy["as_of"])
        latest = date.fromisoformat(policy["latest_report_period"]) if policy.get("latest_report_period") else None
    except (TypeError, ValueError):
        raise ValueError("decision policy dates must be ISO dates") from None
    if latest and latest > as_of:
        raise ValueError("latest report period cannot be after the decision date")
    computed = {
        item["id"] for item in (valuation_context or {}).get("metrics", [])
        if item["status"] == "computed"
    }
    methods = policy["valuation_methods"]
    if not isinstance(methods, list) or not 1 <= len(methods) <= 8:
        raise ValueError("decision methods must be a bounded list")
    seen: set[str] = set()
    for method in methods:
        if (not isinstance(method, dict)
                or set(method) != {"id", "name", "status", "metric_ids", "missing_inputs", "summary"}
                or any(not isinstance(method.get(key), str) or not method[key].strip()
                       or len(method[key]) > limit
                       for key, limit in (("id", 80), ("name", 40), ("summary", 480)))
                or method["id"] in seen or method["status"] not in {"ready", "blocked"}):
            raise ValueError("invalid or duplicate decision method")
        seen.add(method["id"])
        for key in ("metric_ids", "missing_inputs"):
            if (not isinstance(method[key], list) or len(method[key]) > 8
                    or any(not isinstance(item, str) or not item or len(item) > 80 for item in method[key])
                    or len(method[key]) != len(set(method[key]))):
                raise ValueError("decision method inputs must be unique bounded IDs")
        if not method["metric_ids"]:
            raise ValueError("decision methods require explicit metric inputs")
        missing = {item for item in method["metric_ids"] if item not in computed}
        if not market_context or market_context.get("status") != "available":
            missing.add("quote_price")
        if set(method["missing_inputs"]) != missing or (method["status"] == "ready") != (not missing):
            raise ValueError("method availability differs from supplied computed inputs")
    triggers = policy["review_triggers"]
    if not isinstance(triggers, list) or not 1 <= len(triggers) <= 8:
        raise ValueError("review triggers must be a bounded list")
    seen = set()
    for trigger in triggers:
        if (not isinstance(trigger, dict) or set(trigger) - {"id", "label", "period"}
                or not {"id", "label"} <= set(trigger)
                or any(not isinstance(trigger[key], str) or not trigger[key].strip()
                       or len(trigger[key]) > limit for key, limit in (("id", 80), ("label", 100)))
                or trigger["id"] in seen):
            raise ValueError("review triggers contain invalid or duplicate entries")
        seen.add(trigger["id"])
        if "period" in trigger:
            try:
                period = date.fromisoformat(trigger["period"])
            except (TypeError, ValueError):
                raise ValueError("review report period must be an ISO date") from None
            if not latest or period <= latest:
                raise ValueError("a review report must follow the latest available report")


def model_decision_index(policy: dict[str, Any]) -> dict[str, Any]:
    if not is_deterministic_decision(policy):
        return policy
    return {
        key: (
            [{k: v for k, v in method.items() if k != "summary"} for method in value]
            if key == "valuation_methods" else value
        )
        for key, value in policy.items()
    }


def decision_policy_prompt() -> str:
    return (
        " Use stock-decision-v2. The application owns metric selection/calculation, method "
        "availability, missing inputs, new_money, valuation.basis and review_trigger text. "
        "Do not return those derived fields. Select valuation.method_id from a ready method "
        "when any is ready; otherwise use a blocked method and status=unknown. "
        "Return valuation={status,method_id,uncertainty,rationale,evidence_refs,calculation_refs}. "
        "A ready method with a known valuation uses uncertainty=none. If business quality "
        "or evidence is too uncertain to judge value, status=unknown and uncertainty=business_uncertainty "
        "are allowed; explain the investment uncertainty, not an invented data gap. "
        "uncertainty=data_missing is only valid when every method is blocked. "
        "Cite all computed metric_ids required by the chosen ready method in calculation_refs. "
        "Method readiness does not prove fair value; low PE, high yield, historical rank "
        "or a peer discount alone is not a buy signal. Give a reasoned valuation and "
        "recommendation=buy|hold|sell|watch, considering risks and counter-evidence. "
        "Do not invent fair prices, earnings forecasts or valuation benchmarks. "
        "Inspect supplied report explanations before claiming causes are unknown, and "
        "check expense directions before describing a cost as a headwind. Distinguish "
        "absolute liabilities from leverage ratios and from interest-bearing debt. "
        "Background claims are unverified context, not established facts. "
        "buy requires known valuation and cited factual premises. existing_position still "
        "requires authorized holdings. Explain every required_metric_id exactly once in "
        "metric_assessments. Return reason and conditions in plain Chinese, without "
        "recomputing figures, repeating valuation categories or exposing internal field IDs. "
        "Use review_trigger_ids from the supplied index; do not invent dates or free-text "
        "review triggers. Do not request already-computed fields or optional inputs of "
        "a different method. Technical/protocol failure is not a watch recommendation."
    )


def apply_decision_schema(advice: dict[str, Any], policy: dict[str, Any]) -> None:
    properties = advice["properties"]
    for field in ("new_money", "review_trigger"):
        properties.pop(field)
        advice["required"].remove(field)
    advice["required"].append("review_trigger_ids")
    properties["review_trigger_ids"] = {
        "type": "array", "minItems": 1, "maxItems": len(policy["review_triggers"]), "uniqueItems": True,
        "items": {"type": "string", "enum": [item["id"] for item in policy["review_triggers"]]},
    }
    valuation = properties["valuation"]
    if "calculation_refs" not in valuation["properties"]:
        valuation["required"].append("calculation_refs")
        valuation["properties"]["calculation_refs"] = {
            "type": "array", "maxItems": 0, "items": {"type": "string"},
        }
    valuation["properties"].pop("basis")
    valuation["required"].remove("basis")
    ready = [item["id"] for item in policy["valuation_methods"] if item["status"] == "ready"]
    method_ids = ready or [item["id"] for item in policy["valuation_methods"]]
    valuation["required"].extend(["method_id", "uncertainty", "rationale"])
    valuation["properties"].update({
        "method_id": {"type": "string", "enum": method_ids},
        "uncertainty": {"type": "string", "enum": (
            ["none", "business_uncertainty"] if ready else ["data_missing"]
        )},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 240},
    })


def review_text(policy: dict[str, Any], ids: Any) -> str:
    if (not isinstance(ids, list) or not ids or len(ids) > len(policy["review_triggers"])
            or any(not isinstance(item, str) for item in ids) or len(ids) != len(set(ids))):
        return ""
    selected = set(ids)
    labels = [item["label"] for item in policy["review_triggers"] if item["id"] in selected]
    if len(labels) != len(selected):
        return ""
    return "；".join(labels) + "。"


def project_decision_result(result: Any, policy: dict[str, Any] | None) -> Any:
    if not is_deterministic_decision(policy) or not isinstance(result, dict):
        return result
    episode = result.get("episode")
    if not isinstance(episode, dict) or not isinstance(episode.get("investment_advice"), dict):
        return result
    advice = dict(episode["investment_advice"])
    # setdefault preserves conflicting model fields for rejection, never rewrites
    # an action, a valuation judgment, or a fabricated citation.
    advice.setdefault("new_money", _new_money(advice.get("recommendation")))
    advice.setdefault("review_trigger", review_text(policy, advice.get("review_trigger_ids")))
    if isinstance(advice.get("valuation"), dict):
        valuation = dict(advice["valuation"])
        method = next((item for item in policy["valuation_methods"]
                       if item["id"] == valuation.get("method_id")), None)
        if method:
            valuation.setdefault("basis", method["summary"])
        advice["valuation"] = valuation
    return {**result, "episode": {**episode, "investment_advice": advice}}


def decision_policy_errors(advice: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    valuation = advice.get("valuation")
    if not isinstance(valuation, dict):
        return ["decision-v2 requires structured valuation reasoning"]
    methods = policy["valuation_methods"]
    method = next((item for item in methods if item["id"] == valuation.get("method_id")), None)
    if (not method or not isinstance(valuation.get("rationale"), str)
            or not valuation["rationale"].strip() or len(valuation["rationale"]) > 240):
        return ["decision-v2 requires a supplied method_id and substantive rationale"]
    ready = any(item["status"] == "ready" for item in methods)
    status, uncertainty = valuation.get("status"), valuation.get("uncertainty")
    errors = []
    if method["status"] != "ready":
        if ready or status != "unknown" or uncertainty != "data_missing":
            errors.append("use a ready method; only fully blocked methods permit data_missing/unknown")
    elif status == "unknown":
        if uncertainty != "business_uncertainty":
            errors.append("ready method inputs cannot be treated as missing; explain business_uncertainty")
    elif uncertainty != "none":
        errors.append("a known valuation must use uncertainty=none")
    refs = valuation.get("calculation_refs")
    if (isinstance(refs, list) and all(isinstance(item, str) for item in refs)
            and len(refs) != len(set(refs))):
        errors.append("valuation calculation references must be unique")
    if method["status"] == "ready" and (
        not isinstance(refs, list) or any(item not in refs for item in method["metric_ids"])
    ):
        errors.append("valuation must cite every computed input of the selected method")
    if valuation.get("basis") != method["summary"]:
        errors.append("valuation.basis is program-owned and cannot be replaced by model prose")
    review = review_text(policy, advice.get("review_trigger_ids"))
    if not review or advice.get("review_trigger") != review:
        errors.append("review_trigger_ids must select supplied future/event triggers without invented dates")
    if advice.get("new_money") != _new_money(advice.get("recommendation")):
        errors.append("new_money is program-derived from the recommendation")
    return errors
