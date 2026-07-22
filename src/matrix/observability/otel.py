"""OpenTelemetry-standardized tracing primitives.

Implements the core OTel concepts:
- Span: a unit of work with standard fields (trace_id, span_id, parent_span_id,
  name, kind, start/end time, status, attributes, events)
- TraceContext: W3C Trace Context propagation (traceparent header format)
- SpanExporter: OTLP/JSON export format for external backends (Jaeger, Zipkin, etc.)

This module is dependency-free (no opentelemetry-sdk required) so it works
in any environment.  The data model follows the OTLP spec:
https://opentelemetry.io/docs/specs/otel/protocol/

Design goals:
1. Existing TraceStore.record() callers keep working — they emit dicts that
   get normalized into OTel Spans internally.
2. _trace_span() now returns an OTel Span with proper trace context.
3. Spans can be exported to any OTLP-compatible backend via HTTP.
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterator

logger = logging.getLogger("matrix.observability.otel")

# ---- Constants (OTLP spec) ---------------------------------------------------

# Span kinds
SPAN_KIND_INTERNAL = 0
SPAN_KIND_SERVER = 1
SPAN_KIND_CLIENT = 2
SPAN_KIND_PRODUCER = 3
SPAN_KIND_CONSUMER = 4

# Status codes
STATUS_UNSET = 0
STATUS_OK = 1
STATUS_ERROR = 2

# Trace/Span ID formats (W3C)
TRACE_ID_LEN = 32  # 16 bytes hex
SPAN_ID_LEN = 16   # 8 bytes hex


def _generate_trace_id() -> str:
    """Generate a W3C-compliant trace ID (16 bytes hex, all zeros invalid)."""
    while True:
        tid = uuid.uuid4().hex
        if tid != "0" * TRACE_ID_LEN:
            return tid


def _generate_span_id() -> str:
    """Generate a W3C-compliant span ID (8 bytes hex, all zeros invalid)."""
    while True:
        sid = uuid.uuid4().hex[:SPAN_ID_LEN]
        if sid != "0" * SPAN_ID_LEN:
            return sid


# ---- Span model --------------------------------------------------------------

@dataclass
class Span:
    """OpenTelemetry Span: a single unit of work in a trace.

    Fields follow the OTLP Span format:
    https://opentelemetry.io/docs/specs/otel/protocol/#span
    """
    trace_id: str                          # 32-char hex, shared across a trace
    span_id: str                           # 16-char hex, unique within a trace
    parent_span_id: str = ""               # 16-char hex, empty for root span
    name: str = ""                         # span name (e.g. "react_llm")
    kind: int = SPAN_KIND_INTERNAL         # OTel SpanKind enum
    start_time_unix_nano: int = 0          # epoch nanoseconds
    end_time_unix_nano: int = 0            # epoch nanoseconds (0 = still open)

    status_code: int = STATUS_UNSET         # OTel StatusCode enum
    status_message: str = ""

    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    # Resource attributes (service.name, etc.)
    resource: dict[str, str] = field(default_factory=dict)

    # Convenience: elapsed in milliseconds (computed from start/end)
    @property
    def elapsed_ms(self) -> float:
        if self.end_time_unix_nano == 0:
            return 0.0
        return round((self.end_time_unix_nano - self.start_time_unix_nano) / 1e6, 3)

    @property
    def is_open(self) -> bool:
        return self.end_time_unix_nano == 0

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute (OTel convention: dot.namespaced keys)."""
        if value is not None:
            self.attributes[key] = value

    def add_event(self, name: str, **attributes: Any) -> None:
        """Add a timed event to the span."""
        self.events.append({
            "name": name,
            "time_unix_nano": _now_nano(),
            "attributes": dict(attributes),
        })

    def set_status(self, code: int, message: str = "") -> None:
        self.status_code = code
        if message:
            self.status_message = message

    def to_otlp_dict(self) -> dict[str, Any]:
        """Convert to OTLP/JSON Span format for export."""
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id or None,
            "name": self.name,
            "kind": self.kind,
            "startTimeUnixNano": str(self.start_time_unix_nano),
            "endTimeUnixNano": str(self.end_time_unix_nano),
            "status": {
                "code": self.status_code,
                "message": self.status_message or None,
            },
            "attributes": [
                {"key": k, "value": _to_any_value(v)}
                for k, v in self.attributes.items()
            ],
            "events": [
                {
                    "name": e["name"],
                    "timeUnixNano": str(e["time_unix_nano"]),
                    "attributes": [
                        {"key": k, "value": _to_any_value(v)}
                        for k, v in e.get("attributes", {}).items()
                    ],
                }
                for e in self.events
            ],
        }


# ---- TraceContext (W3C) ------------------------------------------------------

@dataclass
class TraceContext:
    """W3C Trace Context for propagation across service boundaries.

    Format: version-trace_id-parent_id-trace_flags
    Example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01

    Spec: https://www.w3.org/TR/trace-context/
    """
    trace_id: str
    span_id: str
    trace_flags: int = 0  # bit 0 = sampled

    def to_traceparent(self) -> str:
        """Serialize to W3C traceparent header value."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    @classmethod
    def from_traceparent(cls, header: str) -> TraceContext | None:
        """Parse a W3C traceparent header. Returns None if invalid."""
        if not header:
            return None
        parts = header.strip().split("-")
        if len(parts) < 4:
            return None
        trace_id = parts[1]
        span_id = parts[2]
        if len(trace_id) != TRACE_ID_LEN or len(span_id) != SPAN_ID_LEN:
            return None
        try:
            flags = int(parts[3], 16)
        except ValueError:
            return None
        return cls(trace_id=trace_id, span_id=span_id, trace_flags=flags)

    @classmethod
    def from_span(cls, span: Span) -> TraceContext:
        return cls(trace_id=span.trace_id, span_id=span.span_id)


# ---- Tracer (span factory) ---------------------------------------------------

class Tracer:
    """Creates spans with proper trace context propagation.

    Usage:
        tracer = Tracer(service_name="matrix-agent")
        with tracer.start_span("react_llm", session_id="s1") as span:
            span.set_attribute("agent_id", "commander")
            ... do work ...
        # span is now ended and recorded

    For child spans:
        with tracer.start_span("parent") as parent:
            with tracer.start_span("child", parent=parent) as child:
                ...
    """

    def __init__(
        self,
        service_name: str = "matrix-agent",
        service_version: str = "",
        sampler_ratio: float = 1.0,
    ):
        self._service_name = service_name
        self._resource: dict[str, str] = {
            "service.name": service_name,
        }
        if service_version:
            self._resource["service.version"] = service_version
        self._sampler_ratio = min(1.0, max(0.0, sampler_ratio))
        self._spans: list[Span] = []
        self._current_span: Span | None = None

    def start_span(
        self,
        name: str,
        *,
        parent: Span | None = None,
        trace_context: TraceContext | None = None,
        kind: int = SPAN_KIND_INTERNAL,
        **attributes: Any,
    ) -> Span:
        """Start a new span.

        Parent resolution order:
        1. Explicit `parent` Span
        2. Explicit `trace_context` (from incoming headers)
        3. Current active span (if any)
        4. New root span

        Args:
            name: Span name (e.g. "react_llm", "tool_call")
            parent: Parent span (for child spans)
            trace_context: Incoming trace context (W3C traceparent)
            kind: OTel SpanKind (INTERNAL, SERVER, CLIENT, etc.)
            **attributes: Initial span attributes
        """
        # Resolve trace context
        if parent:
            trace_id = parent.trace_id
            parent_span_id = parent.span_id
        elif trace_context:
            trace_id = trace_context.trace_id
            parent_span_id = trace_context.span_id
        elif self._current_span:
            trace_id = self._current_span.trace_id
            parent_span_id = self._current_span.span_id
        else:
            trace_id = _generate_trace_id()
            parent_span_id = ""

        span_id = _generate_span_id()

        # Sampling decision
        if self._sampler_ratio < 1.0:
            sampled = random.random() < self._sampler_ratio
        else:
            sampled = True

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time_unix_nano=_now_nano(),
            resource=dict(self._resource),
        )

        # Add standard attributes
        span.set_attribute("session.id", attributes.pop("session_id", ""))
        span.set_attribute("agent.id", attributes.pop("agent_id", ""))
        for k, v in attributes.items():
            span.set_attribute(k, v)

        if not sampled:
            span.trace_id = "0" * TRACE_ID_LEN  # mark as not sampled

        self._spans.append(span)
        return span

    def end_span(self, span: Span) -> None:
        """End a span and finalize it."""
        span.end_time_unix_nano = _now_nano()

    @property
    def spans(self) -> list[Span]:
        """All completed spans."""
        return [s for s in self._spans if not s.is_open]

    @property
    def all_spans(self) -> list[Span]:
        """All spans (including open ones)."""
        return list(self._spans)

    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a given trace."""
        return [s for s in self._spans if s.trace_id == trace_id]

    def clear(self) -> None:
        """Clear all spans (after export)."""
        self._spans.clear()


# ---- Span context manager ---------------------------------------------------

class SpanContext:
    """Context manager for a span's lifecycle.

    Created by Tracer.start_span() — not intended for direct instantiation.
    """
    def __init__(self, tracer: Tracer, span: Span):
        self._tracer = tracer
        self._span = span
        self._previous: Span | None = None

    def __enter__(self) -> Span:
        self._previous = self._tracer._current_span
        self._tracer._current_span = self._span
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self._span.set_status(STATUS_ERROR, str(exc_val)[:200])
            self._span.add_event("exception", type=exc_type.__name__, message=str(exc_val)[:200])
        self._tracer.end_span(self._span)
        self._tracer._current_span = self._previous


# Extend Tracer with context manager support
def _tracer_start_span(self, name: str, **kw) -> SpanContext:
    """Start a span and return a context manager."""
    span = self.start_span(name, **kw)
    return SpanContext(self, span)

Tracer.start_span_ctx = _tracer_start_span


# ---- OTLP Exporter ----------------------------------------------------------

class OTLPExporter:
    """Exports completed spans in OTLP/JSON format.

    Can export to:
    - stdout (for debugging)
    - HTTP endpoint (for OTLP receivers like Jaeger, Tempo, etc.)
    - In-memory buffer (for testing)

    Usage:
        exporter = OTLPExporter(endpoint="http://localhost:4318/v1/traces")
        exporter.export(tracer.spans)

    Or for testing:
        exporter = OTLPExporter()
        exporter.export(tracer.spans)
        data = exporter.get_buffered()
    """

    def __init__(
        self,
        endpoint: str = "",
        service_name: str = "matrix-agent",
        timeout: float = 5.0,
    ):
        self._endpoint = endpoint
        self._service_name = service_name
        self._timeout = timeout
        self._buffer: list[dict[str, Any]] = []

    def export(self, spans: list[Span]) -> bool:
        """Export spans in OTLP/JSON format.

        Returns True if export succeeded (or buffered), False on error.
        """
        if not spans:
            return True

        payload = self._build_otlp_request(spans)

        if not self._endpoint:
            # Buffer in memory (for testing/debugging)
            self._buffer.append(payload)
            return True

        # HTTP export
        try:
            import urllib.request
            req = urllib.request.Request(
                self._endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=self._timeout)
            return True
        except Exception as e:
            logger.warning("otlp_export_failed: %s", e)
            return False

    def _build_otlp_request(self, spans: list[Span]) -> dict[str, Any]:
        """Build OTLP/HTTP JSON request body.

        Format:
        https://opentelemetry.io/docs/specs/otel/protocol/json/
        """
        # Group spans by resource
        resource_groups: dict[str, list[Span]] = {}
        for span in spans:
            key = json.dumps(span.resource, sort_keys=True)
            resource_groups.setdefault(key, []).append(span)

        scope_spans = []
        for _key, group in resource_groups.items():
            scope_spans.append({
                "scope": {"name": "matrix-agent"},
                "spans": [s.to_otlp_dict() for s in group],
                "resource": {
                    "attributes": [
                        {"key": k, "value": _to_any_value(v)}
                        for k, v in group[0].resource.items()
                    ],
                },
            })

        return {
            "resourceSpans": scope_spans,
        }

    def get_buffered(self) -> list[dict[str, Any]]:
        """Get buffered exports (for testing)."""
        return list(self._buffer)

    def clear_buffer(self) -> None:
        self._buffer.clear()


# ---- Helpers ----------------------------------------------------------------

def _now_nano() -> int:
    """Current Unix timestamp in nanoseconds."""
    return int(time.time() * 1e9)


def _to_any_value(value: Any) -> dict[str, Any]:
    """Convert a Python value to OTLP AnyValue format."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_to_any_value(v) for v in value]}}
    if isinstance(value, dict):
        return {"kvlistValue": {
            "values": [
                {"key": k, "value": _to_any_value(v)}
                for k, v in value.items()
            ]
        }}
    return {"stringValue": str(value)}


def normalize_legacy_event(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize a legacy trace event dict to OTel span attributes.

    Maps old field names to OTel semantic conventions:
    - session_id → session.id
    - agent_id → agent.id
    - tool_name → tool.name
    - elapsed_ms → (used for span duration)
    - arguments → attributes
    - result → attributes (truncated)
    """
    attrs: dict[str, Any] = {}

    # Direct attribute mappings
    if event.get("session_id"):
        attrs["session.id"] = event["session_id"]
    if event.get("agent_id"):
        attrs["agent.id"] = event["agent_id"]
    if event.get("tool_name"):
        attrs["tool.name"] = event["tool_name"]
    if event.get("node_name"):
        attrs["node.name"] = event["node_name"]

    # Arguments → attributes
    args = event.get("arguments")
    if args and isinstance(args, dict):
        for k, v in args.items():
            attrs[f"args.{k}"] = v
    elif args and isinstance(args, str):
        attrs["args"] = args[:500]

    # Result (truncated)
    result = event.get("result") or event.get("result_preview")
    if result:
        attrs["result.preview"] = str(result)[:500]

    # Error
    if event.get("error"):
        attrs["error.message"] = str(event["error"])[:500]

    # Path (for HTTP trace events)
    if event.get("path"):
        attrs["http.path"] = event["path"]

    return attrs


def event_to_span_kind(event_type: str) -> int:
    """Map legacy event_type to OTel SpanKind."""
    if event_type in ("tool_call", "react_tool"):
        return SPAN_KIND_CLIENT  # calling an external tool
    if event_type in ("span_start", "span_end"):
        return SPAN_KIND_INTERNAL
    if event_type in ("http_request",):
        return SPAN_KIND_SERVER
    return SPAN_KIND_INTERNAL


def event_to_status(event: dict[str, Any]) -> tuple[int, str]:
    """Map legacy event to OTel StatusCode and message."""
    ok = event.get("ok")
    if ok is False or event.get("error"):
        return STATUS_ERROR, str(event.get("error", ""))[:200]
    if ok is True:
        return STATUS_OK, ""
    return STATUS_UNSET, ""
