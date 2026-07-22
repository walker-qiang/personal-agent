"""Observability: tracing, events, metrics.

Exposes both the legacy TraceStore (SQLite event store) and the new
OTel-standardized tracing primitives (Span, Tracer, OTLPExporter).
"""

from .trace import TraceStore, TraceLogger
from .otel import (
    Span,
    Tracer,
    TraceContext,
    OTLPExporter,
    SpanContext,
    # Constants
    SPAN_KIND_INTERNAL,
    SPAN_KIND_SERVER,
    SPAN_KIND_CLIENT,
    SPAN_KIND_PRODUCER,
    SPAN_KIND_CONSUMER,
    STATUS_UNSET,
    STATUS_OK,
    STATUS_ERROR,
    # Helpers
    normalize_legacy_event,
    event_to_span_kind,
    event_to_status,
)

__all__ = [
    "TraceStore",
    "TraceLogger",
    "Span",
    "Tracer",
    "TraceContext",
    "OTLPExporter",
    "SpanContext",
    "SPAN_KIND_INTERNAL",
    "SPAN_KIND_SERVER",
    "SPAN_KIND_CLIENT",
    "SPAN_KIND_PRODUCER",
    "SPAN_KIND_CONSUMER",
    "STATUS_UNSET",
    "STATUS_OK",
    "STATUS_ERROR",
    "normalize_legacy_event",
    "event_to_span_kind",
    "event_to_status",
]
