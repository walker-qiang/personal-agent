"""OS-authorized, single-request synthesis transport. Never retries or repairs."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from .analysis import AnalysisInputs
from .advice import AdviceInputs, VERSION as ADVICE_VERSION
from .preparation import ResearchCapabilities
from ..llm.deepseek import DeepSeekClient
from ..llm.http import post_json_raw_once


ENDPOINT = "https://api.deepseek.com/chat/completions"
ENCODING_VERSION = "deepseek-recipe:0.2.0:DeepseekV41Encoding"


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def count_input(payload: dict[str, Any], tokenizer_path: str, expected_hash: str) -> int:
    import deepseek_recipe as recipe

    if recipe.__version__ != "0.2.0":
        raise ValueError("unapproved provider encoding version")
    raw = Path(tokenizer_path).read_bytes()
    if _digest(raw) != expected_hash:
        raise ValueError("tokenizer digest differs from the approved revision")
    options = recipe.ConversionOptions().with_model_alias(
        "deepseek-flash", recipe.ModelVersion.DeepseekV41Flash,
    )
    converted = recipe.ChatCompletionRequest(payload).convert(options)
    tokenizer = recipe.Tokenizer.from_str(raw.decode("utf-8"))
    return len(recipe.DeepseekV41Encoding().with_tokenizer(tokenizer).encode(converted.conversation))


def prepare(value: dict[str, Any]) -> dict[str, Any]:
    auth, context, caps = value["authorization"], value["context"], value["capabilities"]
    if auth.get("advice_protocol") != ADVICE_VERSION:
        raise ValueError("legacy output contracts are read-only; prepare the program-owned advice protocol")
    if auth["provider"] != "deepseek" or auth["model"] != "deepseek-flash":
        raise ValueError("provider/model differs from the controlled synthesis scope")
    if auth["encoding_version"] != ENCODING_VERSION:
        raise ValueError("unapproved prompt encoding")
    if not any(
        target["research_id"] == context["plan"]["plan_id"]
        and target["context_hash"] == context["context_hash"]
        for target in auth["targets"]
    ):
        raise ValueError("context is outside the approved target list")
    capabilities = ResearchCapabilities.from_payload(
        caps, expected_contract_digest=auth["contract_digest"],
    )
    AnalysisInputs.from_authorized_payload(
        capabilities, context, expected_plan_id=context["plan"]["plan_id"],
        expected_instrument_id=context["instrument"]["instrument_id"],
        expected_context_hash=context["context_hash"],
        external_authorization=auth["permission"],
    )
    packet = value["decision_facts"]
    advice = AdviceInputs.from_packet(
        capabilities, packet, expected_advice_digest=auth["advice_contract_digest"],
        expected_context_hash=context["context_hash"],
        expected_instrument_id=context["instrument"]["instrument_id"],
        expected_packet_hash=value["decision_facts_hash"],
    )
    prompt = advice.model_request(research_focus=context["plan"].get("focus", ""))
    # Truncation is forbidden at this boundary: admission covers the exact payload.
    client = DeepSeekClient(api_key="", model=auth["model"], max_message_chars=0)
    payload = client._build_payload(
        prompt["system"], prompt["messages"],
        tools=[{
            "name": "return_json", "description": "Return the requested structured research result.",
            "input_schema": advice.output_schema(),
        }],
        tool_choice="required", max_tokens=auth["max_output_tokens"],
        temperature=0.0, thinking={"type": "disabled"},
    )
    count = count_input(payload, auth["tokenizer_path"], auth["tokenizer_sha256"])
    # Explicit allowance for provider-side framing; it is reserved, not a tokenizer estimate.
    bound = count + 1024
    if bound > auth["max_input_tokens"]:
        raise ValueError(f"input budget exceeded: encoded={count}, admission_bound={bound}")
    body = _encoded(payload)
    prepared = {
        "endpoint": ENDPOINT, "model": auth["model"], "body": body.decode("utf-8"),
        "body_sha256": _digest(body), "input_tokens": count, "input_token_bound": bound,
        "max_output_tokens": auth["max_output_tokens"], "encoding_version": ENCODING_VERSION,
        "tokenizer_sha256": auth["tokenizer_sha256"],
    }
    prepared.update(advice_protocol=ADVICE_VERSION, decision_facts_hash=packet["packet_hash"])
    return prepared


def execute(value: dict[str, Any]) -> dict[str, Any]:
    from dotenv import load_dotenv
    from ..config import load_config

    auth, prepared = value["authorization"], value["request"]
    if auth.get("advice_protocol") != ADVICE_VERSION:
        raise ValueError("legacy output contracts are read-only; do not dispatch a legacy request")
    if auth.get("advice_protocol", "") != prepared.get("advice_protocol", ""):
        raise ValueError("archived advice protocol changed")
    load_dotenv(Path.cwd() / ".env", override=False)
    config = load_config()
    if config.deepseek_base_url.rstrip("/") != "https://api.deepseek.com":
        raise ValueError("configured endpoint differs from the authorized official provider")
    if not config.deepseek_api_key:
        raise ValueError("configured provider credential is missing")
    body = prepared["body"].encode("utf-8")
    if (
        prepared["endpoint"] != ENDPOINT
        or prepared["model"] != auth["model"]
        or _digest(body) != prepared["body_sha256"]
    ):
        raise ValueError("archived request binding changed")
    payload = json.loads(body)
    if auth.get("advice_protocol"):
        if payload["tools"][0]["function"]["parameters"]["properties"]["packet_hash"] != {"const": prepared["decision_facts_hash"]}:
            raise ValueError("advice schema changed its frozen packet")
    if (
        payload["model"] != auth["model"]
        or payload["max_tokens"] != auth["max_output_tokens"]
        or payload.get("thinking") != {"type": "disabled"}
        or payload.get("stream", False)
        or payload.get("tool_choice") != "required"
    ):
        raise ValueError("request changed its authorized inference settings")
    count = count_input(payload, auth["tokenizer_path"], auth["tokenizer_sha256"])
    if count != prepared["input_tokens"] or count + 1024 != prepared["input_token_bound"]:
        raise ValueError("archived token admission changed")
    if prepared["input_token_bound"] > auth["max_input_tokens"]:
        raise ValueError("request exceeds its authorized input budget")
    status, raw, failure = post_json_raw_once(
        ENDPOINT, body,
        {"authorization": f"Bearer {config.deepseek_api_key}", "content-type": "application/json"},
        auth["timeout_seconds"],
    )
    return {
        "http_status": status, "body_base64": base64.b64encode(raw).decode("ascii"),
        "transport_error": failure, "request_sha256": prepared["body_sha256"],
    }


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "execute"):
        raise ValueError("choose prepare or execute; authorization is supplied by OS")
    value = json.load(sys.stdin)
    result = prepare(value) if sys.argv[1] == "prepare" else execute(value)
    sys.stdout.buffer.write(_encoded(result))


if __name__ == "__main__":
    main()
