"""Deterministic decision DTO projection; no model calls or investment thresholds."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import re
from typing import Any
from .stock_decision_tools import (
    apply_decision_analysis_schema, decision_analysis_errors,
    decision_condition_text, validate_decision_tools, decision_analysis_text,
)

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
    documents: list[dict[str, Any]] | None = None, code: str = "",
) -> None:
    required = {"version", "objective", "required_metric_ids", "as_of",
                "valuation_methods", "review_triggers"}
    if not required <= set(policy) or set(policy) - required - {
        "latest_report_period", "capital_return_facts", "analysis_profile", "reported_explanations", "comparisons",
        "decision_tools",
    }:
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
    validate_capital_return_facts(policy, documents or [], code)
    validate_prepared_evidence(policy, documents or [])
    validate_comparisons(policy, code)
    validate_decision_tools(policy, valuation_context, market_context)


def validate_prepared_evidence(policy: dict[str, Any], documents: list[dict[str, Any]]) -> None:
    profile = policy.get("analysis_profile")
    if profile is None and not policy.get("reported_explanations"):
        return
    if any(not isinstance(item, dict) for item in documents):
        raise ValueError("prepared evidence sources must be objects")
    by_id = {item.get("source_id"): item for item in documents}
    if profile is not None and (
        not isinstance(profile, dict) or set(profile) != {"kind", "source_id"}
        or profile.get("kind") not in ("general", "banking", "utility")
        or not isinstance(profile.get("source_id"), str) or profile["source_id"] not in by_id
    ):
        raise ValueError("analysis profile must identify a prepared official source")
    explanations = policy.get("reported_explanations", [])
    if not isinstance(explanations, list) or len(explanations) > 3:
        raise ValueError("reported explanations must be bounded")
    seen: set[str] = set()
    for item in explanations:
        if (not isinstance(item, dict)
                or set(item) != {"id", "kind", "source_id", "period", "metric_ids", "quote"}
                or not isinstance(item.get("quote"), str) or not 1 <= len(item["quote"]) <= 1500
                or item.get("kind") not in ("issuer_explanation", "issuer_reported_adjustment")
                or not isinstance(item.get("source_id"), str) or item["source_id"] not in by_id
                or item.get("period") != policy.get("latest_report_period")):
            raise ValueError("reported explanation identity or period is invalid")
        source = by_id[item["source_id"]]
        expected = "reported-" + hashlib.sha256((item["source_id"] + "|" + item["quote"]).encode()).hexdigest()[:20]
        if (item["id"] != expected or expected in seen or item["quote"] not in source.get("excerpt", "")
                or source.get("report_period") != item["period"]):
            raise ValueError("reported explanation must be an exact original source span")
        if item["kind"] == "issuer_reported_adjustment" and not any(
            marker in re.sub(r"\s+", "", item["quote"]).lower()
            for marker in ("剔除", "扣除该影响", "excludingtheimpact")
        ):
            raise ValueError("issuer adjustment requires an explicit original disclosure qualifier")
        seen.add(expected)
        ids = item["metric_ids"]
        if (not isinstance(ids, list) or not ids or any(not isinstance(metric, str) for metric in ids)
                or len(ids) != len(set(ids))
                or any(metric not in ("revenue_yoy_pct", "net_profit_yoy_pct", "operating_cash_flow_yoy_pct") for metric in ids)):
            raise ValueError("reported explanation metric IDs are invalid")


def validate_comparisons(policy: dict[str, Any], code: str) -> None:
    comparisons = policy.get("comparisons", [])
    if not isinstance(comparisons, list) or len(comparisons) > 3:
        raise ValueError("financial comparisons must be bounded")
    definitions = {
        "peer_cash_conversion_ttm": ("ttm_cash_to_profit", "ratio", "same_period_currency_reported_ttm"),
        "peer_net_margin_ytd": ("net_margin_pct", "percent", "same_period_type_currency_reported_ytd"),
        "peer_operating_margin_ytd": ("operating_margin_pct", "percent", "same_period_type_currency_reported_ytd"),
    }
    seen: set[str] = set()
    for item in comparisons:
        if (not isinstance(item, dict) or set(item) - {
            "id", "metric_id", "status", "basis", "period", "target", "peers", "median", "position", "reason",
        } or not {"id", "metric_id", "status", "basis", "period", "peers"} <= set(item)
                or not isinstance(item.get("id"), str) or item["id"] not in definitions or item["id"] in seen
                or item.get("status") not in ("comparable", "not_comparable", "not_applicable")):
            raise ValueError("financial comparison identity or status is invalid")
        seen.add(item["id"])
        metric, unit, basis = definitions[item["id"]]
        if item["metric_id"] != metric or item["basis"] != basis:
            raise ValueError("financial comparison formula differs from its index")
        if item["status"] != "comparable":
            if item["peers"] or any(field in item for field in ("target", "median", "position")) or not item.get("reason"):
                raise ValueError("unavailable comparisons cannot contain a ranking")
            continue
        try:
            if date.fromisoformat(item["period"]) > date.fromisoformat(policy["as_of"]):
                raise ValueError("financial comparison period cannot be in the future")
        except (TypeError, ValueError):
            raise ValueError("financial comparison period is invalid") from None
        peers, target = item.get("peers"), item.get("target")
        if not isinstance(peers, list) or not 2 <= len(peers) <= 3 or not isinstance(target, dict):
            raise ValueError("financial comparisons require a target and two supported peers")
        currency, period_type = target.get("currency"), target.get("period_type")
        codes: set[str] = set()
        numbers: list[Decimal] = []
        for value in [target, *peers]:
            if (not isinstance(value, dict) or set(value) != {
                "code", "value", "unit", "currency", "period", "period_type", "verification",
            } or not isinstance(value.get("code"), str) or value["code"] in codes
                    or not isinstance(value.get("value"), str) or len(value["value"]) > 64
                    or not re.fullmatch(r"-?\d+(?:\.\d+)?", value["value"])
                    or value.get("unit") != unit or value.get("period") != item["period"]
                    or currency not in ("CNY", "HKD", "USD") or value.get("currency") != currency
                    or not isinstance(value.get("period_type"), str) or not value["period_type"] or len(value["period_type"]) > 20
                    or not isinstance(value.get("verification"), str) or len(value["verification"]) > 40
                    or metric != "ttm_cash_to_profit" and value["period_type"] != period_type):
                raise ValueError("financial comparison values differ in period, currency, unit or fiscal basis")
            codes.add(value["code"])
            numbers.append(Decimal(value["value"]))
        if target["code"] != code:
            raise ValueError("financial comparison target differs from the confirmed subject")
        try:
            with localcontext() as decimal_context:
                decimal_context.prec = 100
                ordered = sorted(numbers[1:])
                middle = len(ordered) // 2
                median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
                actual = Decimal(item["median"])
                if not actual.is_finite() or abs(actual - median) > Decimal("0.00005"):
                    raise ValueError("financial comparison median differs from its inputs")
                position = "lower" if numbers[0] < median else "higher" if numbers[0] > median else "equal"
                if item.get("position") != position:
                    raise ValueError("financial comparison direction differs from its inputs")
        except (KeyError, InvalidOperation, TypeError):
            raise ValueError("financial comparison median is invalid") from None


def valuation_calculation_ids(valuation: dict[str, Any] | None, policy: dict[str, Any] | None) -> set[str]:
    ids = {item["id"] for item in (valuation or {}).get("metrics", []) if item["status"] == "computed"}
    ids.update(item["id"] for item in (policy or {}).get("comparisons", []) if item["status"] == "comparable")
    return ids


def comparison_claim_errors(advice: dict[str, Any], policy: dict[str, Any], summary: str) -> list[str]:
    if "comparisons" not in policy:
        return []
    valuation = advice.get("valuation", {})
    if not isinstance(valuation, dict):
        return []
    assessments = advice.get("metric_assessments")
    assessments = assessments if isinstance(assessments, list) else []
    text = " ".join([advice["reason"], str(summary), str(valuation.get("rationale") or ""),
                     decision_analysis_text(advice),
                     *(item["explanation"] for item in assessments
                       if isinstance(item, dict) and isinstance(item.get("explanation"), str))])
    refs = valuation.get("calculation_refs", [])
    known = {item["id"]: item for item in policy["comparisons"]}
    errors = []
    pattern = r"(?:(当前|最新|目前|当期)[^。；;，,\n]{0,6})?(现金转换|现金转化|现金利润比|净利率|营业利润率)[^。；;，,\n]{0,12}?(弱于|强于|低于|高于|优于|不及|落后于|逊于)(?:同业|同行)"
    for temporal, name, direction in re.findall(pattern, text):
        comparison_id = {"净利率": "peer_net_margin_ytd", "营业利润率": "peer_operating_margin_ytd"}.get(
            name, "peer_cash_conversion_ttm",
        )
        record = known.get(comparison_id)
        if not record or record["status"] != "comparable" or comparison_id not in refs:
            errors.append("cross-company claim requires a cited same-basis comparison: " + comparison_id)
        elif temporal and record["period"] != policy.get("latest_report_period"):
            errors.append("a historical common period cannot establish a current peer ranking: " + comparison_id)
        elif record["position"] != ("higher" if direction in ("强于", "高于", "优于") else "lower"):
            errors.append("cross-company claim contradicts the computed comparison direction: " + comparison_id)
    return errors


def _repurchase_quote_values(quote: str) -> tuple[str, str, str]:
    compact = re.sub(r"\s+", "", quote)
    match = re.match(r"截至(20\d{2})年(\d{1,2})月(\d{1,2})日", compact)
    if not match:
        raise ValueError("capital return quote must identify its progress date")
    observed = date(int(match[1]), int(match[2]), int(match[3])).isoformat()
    shares = re.search(r"累计回购股份数量为([\d,]+)股", compact)
    not_started = any(word in compact for word in ("暂未实施本次回购", "尚未实施本次回购", "未实施本次回购"))
    if shares:
        count = int(shares[1].replace(",", ""))
        if not_started and count > 0:
            raise ValueError("capital return quote contains conflicting execution states")
        return observed, str(count), "in_progress" if count else "not_started"
    if not_started:
        return observed, "0", "not_started"
    raise ValueError("capital return quote does not establish an execution state")


def validate_capital_return_facts(policy: dict[str, Any], documents: list[dict[str, Any]], code: str) -> None:
    facts = policy.get("capital_return_facts", [])
    if not isinstance(facts, list) or len(facts) > 12:
        raise ValueError("capital return facts must be bounded")
    if not facts:
        return
    by_id = {item["source_id"]: item.get("excerpt", "") for item in documents}
    seen: set[str] = set()
    for fact in facts:
        if (not isinstance(fact, dict) or set(fact) != {
            "event_id", "scheme_ref", "as_of", "status", "executed_shares",
            "source_id", "quote", "superseded", "summary",
        } or any(not isinstance(fact.get(key), str) or not fact[key] or len(fact[key]) > limit
                 for key, limit in (("event_id", 80), ("scheme_ref", 20), ("as_of", 10),
                                    ("status", 20), ("executed_shares", 20), ("source_id", 80),
                                    ("quote", 240), ("summary", 160)))):
            raise ValueError("capital return fact metadata is invalid")
        event_id = "repurchase-" + hashlib.sha256((code + "|" + fact["scheme_ref"]).encode()).hexdigest()[:20]
        if fact["event_id"] != event_id or event_id in seen:
            raise ValueError("capital return event identity differs from target and scheme")
        seen.add(event_id)

        def validate_span(span: dict[str, Any]) -> tuple[str, str, str]:
            if (not isinstance(span, dict) or set(span) != {"source_id", "quote"}
                    or not isinstance(span["source_id"], str) or not isinstance(span["quote"], str)
                    or not span["quote"] or len(span["quote"]) > 240):
                raise ValueError("capital return source span is invalid")
            source = by_id.get(span["source_id"], "")
            if not source or span["quote"] not in source:
                raise ValueError("capital return quote is not in an authorized source")
            compact = re.sub(r"\s+", "", source)
            refs = set(re.findall(r"回购(?:股份)?报告书.{0,32}?公告编号[：:](20\d{2}[-－]\d{1,6})", compact))
            refs = {ref.replace("－", "-") for ref in refs}
            if refs != {fact["scheme_ref"]}:
                raise ValueError("capital return quote does not identify the same repurchase scheme")
            return _repurchase_quote_values(span["quote"])

        observed, shares, status = validate_span({"source_id": fact["source_id"], "quote": fact["quote"]})
        if (observed != fact["as_of"] or shares != fact["executed_shares"] or status != fact["status"]
                or observed > policy["as_of"]):
            raise ValueError("capital return state differs from dated source figures")
        obsolete = fact["superseded"]
        if not isinstance(obsolete, list) or len(obsolete) > 12:
            raise ValueError("capital return superseded spans must be bounded")
        for span in obsolete:
            prior_date, prior_shares, _ = validate_span(span)
            if prior_date >= observed or int(prior_shares) > int(shares):
                raise ValueError("capital return ordering is inconsistent")


def capital_return_ranges(
    documents: list[dict[str, Any]], policy: dict[str, Any] | None,
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, list[tuple[int, int]]]]:
    current: dict[str, list[tuple[int, int]]] = {}
    obsolete: dict[str, list[tuple[int, int]]] = {}
    if not is_deterministic_decision(policy):
        return current, obsolete
    by_id = {item["source_id"]: item.get("excerpt", "") for item in documents}
    for fact in policy.get("capital_return_facts", []):
        for target, spans in ((current, [fact]), (obsolete, fact["superseded"])):
            for span in spans:
                text, quote, start = by_id.get(span["source_id"], ""), span["quote"], 0
                while quote:
                    index = text.find(quote, start)
                    if index < 0:
                        break
                    target.setdefault(span["source_id"], []).append((index, index + len(quote)))
                    start = index + len(quote)
    for item in policy.get("reported_explanations", []):
        text, quote = by_id.get(item["source_id"], ""), item["quote"]
        index = text.find(quote)
        if index >= 0:
            current.setdefault(item["source_id"], []).append((index, index + len(quote)))
    return current, obsolete


def model_decision_index(policy: dict[str, Any]) -> dict[str, Any]:
    if not is_deterministic_decision(policy):
        return policy
    result = {
        key: (
            [{k: v for k, v in method.items() if k != "summary"} for method in value]
            if key == "valuation_methods" else value
        )
        for key, value in policy.items()
    }
    if "capital_return_facts" in result:
        result["capital_return_facts"] = [
            {**{key: value for key, value in fact.items() if key not in {"quote", "superseded"}},
             "superseded_source_ids": sorted({span["source_id"] for span in fact["superseded"]})}
            for fact in policy["capital_return_facts"]
        ]
    if "reported_explanations" in result:
        result["reported_explanations"] = [
            {key: value for key, value in item.items() if key != "quote"}
            for item in result["reported_explanations"]
        ]
    if "comparisons" in result:
        compact = []
        for item in result["comparisons"]:
            value = dict(item)
            if item["status"] == "comparable":
                target = item["target"]
                value.update({
                    "unit": target["unit"], "currency": target["currency"],
                    "period_type": target["period_type"],
                    "target": {"code": target["code"], "value": target["value"]},
                    "peers": [{"code": peer["code"], "value": peer["value"]} for peer in item["peers"]],
                    "verification": ("official_reconciled" if all(
                        member["verification"] == "official_reconciled" for member in [target, *item["peers"]]
                    ) else "provider_unverified"),
                })
            compact.append(value)
        result["comparisons"] = compact
    if result.get("decision_tools") is not None:
        tools = result["decision_tools"]
        result["decision_tools"] = {
            **tools,
            "cases": [{key: value for key, value in case.items() if key != "summary"} for case in tools["cases"]],
            "conditions": [{key: condition[key] for key in ("id", "case_id", "label")}
                           for condition in tools["conditions"]],
        }
    return result


def decision_policy_prompt() -> str:
    return (
        " Use stock-decision-v2. The application owns metric selection/calculation, method "
        "availability, missing inputs, new_money, valuation.basis and review_trigger text. "
        "Do not return those derived fields. Select valuation.method_id from a ready method "
        "when any is ready; otherwise use a blocked method and status=unknown. "
        "Return valuation={status,method_id,uncertainty,rationale,evidence_refs,calculation_refs}. "
        "Valuation direction and business uncertainty are independent: a known valuation "
        "may use uncertainty=none or business_uncertainty. If business quality "
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
        "reported_explanations identify original issuer disclosures, including issuer-reported "
        "adjustments; they are not script calculations or normalized/free cash flow. Inspect "
        "their prioritized evidence before requesting another read of the same report. "
        "Cross-company cash conversion and margin claims must use comparisons with status=comparable "
        "and cite their id in calculation_refs. A common historical period is not the latest-period ranking. "
        "Comparison position is versus the supplied peer median, not every peer or the entire industry. "
        "For analysis_profile.kind=banking, operating cash flow and cash/profit ratios are not "
        "general corporate quality signals; use disclosed credit/capital evidence without inventing ratios. "
        "capital_return_facts are dated current states for distinct repurchase schemes; "
        "cite their current source_id. Superseded progress is not a current execution state, "
        "and an approved plan is not an implemented or completed plan. "
        "buy requires known valuation and cited factual premises. existing_position still "
        "requires authorized holdings. Explain every required_metric_id exactly once in "
        "metric_assessments. Return reason and conditions in plain Chinese, without "
        "recomputing figures, repeating valuation categories or exposing internal field IDs. "
        "Use review_trigger_ids from the supplied index; do not invent dates or free-text "
        "review triggers. Do not request already-computed fields or optional inputs of "
        "a different method. Technical/protocol failure is not a watch recommendation. "
        "decision_tools indexes computed sensitivities and conditions. In decision_analysis explain "
        "cases from two distinct available tools (all if fewer), give current_price_reasoning and "
        "matching condition_ids; do not return conditions prose. Explain how each case affects the "
        "action. Cases are conditional, not forecasts, worst-case bounds, return hurdles or fair prices; "
        "inherit input verification limits. Peer parity retains its common-period basis. Earnings yield "
        "is not expected return. Deterioration alone does not make value unknowable: weigh price against "
        "stress; if still unknown, name a specific unresolved_premise. Such a decision remains limited."
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
    apply_decision_analysis_schema(advice, policy)


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
    if policy.get("decision_tools") is not None and isinstance(advice.get("decision_analysis"), dict):
        advice.setdefault("conditions", decision_condition_text(policy, advice["decision_analysis"].get("condition_ids")))
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
    elif uncertainty not in ("none", "business_uncertainty"):
        errors.append("a known valuation cannot claim missing method inputs")
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
    errors.extend(decision_analysis_errors(advice, policy))
    return errors
