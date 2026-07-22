"""Tests for OpenTelemetry-standardized tracing primitives."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from matrix.observability.otel import (
    Span,
    Tracer,
    TraceContext,
    OTLPExporter,
    SpanContext,
    SPAN_KIND_INTERNAL,
    SPAN_KIND_SERVER,
    SPAN_KIND_CLIENT,
    STATUS_UNSET,
    STATUS_OK,
    STATUS_ERROR,
    normalize_legacy_event,
    event_to_span_kind,
    event_to_status,
    _to_any_value,
    _generate_trace_id,
    _generate_span_id,
    TRACE_ID_LEN,
    SPAN_ID_LEN,
)
from matrix.observability.trace import TraceStore


# ---- Fixtures ---------------------------------------------------------------

@pytest.fixture
def tracer():
    return Tracer(service_name="test-agent")


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "otel_test.db"
    return TraceStore(db_path)


# ---- Span model -------------------------------------------------------------

class TestSpan:
    def test_default_values(self):
        span = Span(
            trace_id="0" * TRACE_ID_LEN,
            span_id="0" * SPAN_ID_LEN,
        )
        assert span.kind == SPAN_KIND_INTERNAL
        assert span.status_code == STATUS_UNSET
        assert span.attributes == {}
        assert span.events == []
        assert span.is_open is True
        assert span.elapsed_ms == 0.0

    def test_set_attribute(self):
        span = Span(trace_id="x" * TRACE_ID_LEN, span_id="y" * SPAN_ID_LEN)
        span.set_attribute("session.id", "s1")
        span.set_attribute("tool.name", "web_search")
        assert span.attributes["session.id"] == "s1"
        assert span.attributes["tool.name"] == "web_search"

    def test_set_attribute_none_ignored(self):
        span = Span(trace_id="x" * TRACE_ID_LEN, span_id="y" * SPAN_ID_LEN)
        span.set_attribute("key", None)
        assert "key" not in span.attributes

    def test_add_event(self):
        span = Span(trace_id="x" * TRACE_ID_LEN, span_id="y" * SPAN_ID_LEN)
        span.add_event("tool.called", tool="web_search")
        assert len(span.events) == 1
        assert span.events[0]["name"] == "tool.called"
        assert span.events[0]["attributes"]["tool"] == "web_search"

    def test_set_status(self):
        span = Span(trace_id="x" * TRACE_ID_LEN, span_id="y" * SPAN_ID_LEN)
        span.set_status(STATUS_ERROR, "something failed")
        assert span.status_code == STATUS_ERROR
        assert span.status_message == "something failed"

    def test_elapsed_ms(self):
        span = Span(
            trace_id="x" * TRACE_ID_LEN,
            span_id="y" * SPAN_ID_LEN,
            start_time_unix_nano=1_000_000_000,
            end_time_unix_nano=1_001_500_000,
        )
        assert span.elapsed_ms == 1.5

    def test_to_otlp_dict(self):
        span = Span(
            trace_id="abcdef1234567890abcdef1234567890",
            span_id="1234567890abcdef",
            parent_span_id="abcdef1234567890",
            name="react_llm",
            kind=SPAN_KIND_INTERNAL,
            start_time_unix_nano=1_000_000_000,
            end_time_unix_nano=1_001_000_000,
        )
        span.set_attribute("session.id", "s1")
        span.add_event("started")

        d = span.to_otlp_dict()
        assert d["traceId"] == "abcdef1234567890abcdef1234567890"
        assert d["spanId"] == "1234567890abcdef"
        assert d["parentSpanId"] == "abcdef1234567890"
        assert d["name"] == "react_llm"
        assert d["kind"] == SPAN_KIND_INTERNAL
        assert d["startTimeUnixNano"] == "1000000000"
        assert d["endTimeUnixNano"] == "1001000000"
        assert len(d["attributes"]) == 1
        assert d["attributes"][0]["key"] == "session.id"
        assert d["attributes"][0]["value"]["stringValue"] == "s1"
        assert len(d["events"]) == 1
        assert d["events"][0]["name"] == "started"

    def test_to_otlp_dict_null_parent(self):
        span = Span(
            trace_id="x" * TRACE_ID_LEN,
            span_id="y" * SPAN_ID_LEN,
        )
        d = span.to_otlp_dict()
        assert d["parentSpanId"] is None  # empty → null


# ---- TraceContext (W3C) -----------------------------------------------------

class TestTraceContext:
    def test_to_traceparent(self):
        ctx = TraceContext(
            trace_id="abcdef1234567890abcdef1234567890",
            span_id="1234567890abcdef",
            trace_flags=1,
        )
        tp = ctx.to_traceparent()
        assert tp == "00-abcdef1234567890abcdef1234567890-1234567890abcdef-01"

    def test_from_traceparent(self):
        header = "00-abcdef1234567890abcdef1234567890-1234567890abcdef-01"
        ctx = TraceContext.from_traceparent(header)
        assert ctx is not None
        assert ctx.trace_id == "abcdef1234567890abcdef1234567890"
        assert ctx.span_id == "1234567890abcdef"
        assert ctx.trace_flags == 1

    def test_from_traceparent_invalid(self):
        assert TraceContext.from_traceparent("") is None
        assert TraceContext.from_traceparent("invalid") is None
        assert TraceContext.from_traceparent("00-short-short-01") is None

    def test_from_traceparent_no_flags(self):
        header = "00-abcdef1234567890abcdef1234567890-1234567890abcdef"
        ctx = TraceContext.from_traceparent(header)
        assert ctx is None  # Missing flags part → invalid

    def test_from_span(self):
        span = Span(trace_id="a" * TRACE_ID_LEN, span_id="b" * SPAN_ID_LEN)
        ctx = TraceContext.from_span(span)
        assert ctx.trace_id == "a" * TRACE_ID_LEN
        assert ctx.span_id == "b" * SPAN_ID_LEN


# ---- Tracer -----------------------------------------------------------------

class TestTracer:
    def test_start_root_span(self, tracer):
        span = tracer.start_span("root", session_id="s1")
        assert span.name == "root"
        assert span.parent_span_id == ""
        assert span.attributes["session.id"] == "s1"
        assert span.is_open is True
        assert span.trace_id != "0" * TRACE_ID_LEN  # sampled

    def test_start_child_span(self, tracer):
        parent = tracer.start_span("parent")
        child = tracer.start_span("child", parent=parent)
        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id

    def test_start_span_with_context(self, tracer):
        ctx = TraceContext(
            trace_id="a" * TRACE_ID_LEN,
            span_id="b" * SPAN_ID_LEN,
        )
        span = tracer.start_span("server", trace_context=ctx)
        assert span.trace_id == "a" * TRACE_ID_LEN
        assert span.parent_span_id == "b" * SPAN_ID_LEN

    def test_end_span(self, tracer):
        span = tracer.start_span("test")
        assert span.is_open is True
        tracer.end_span(span)
        assert span.is_open is False
        assert span.end_time_unix_nano > span.start_time_unix_nano

    def test_spans_property_only_completed(self, tracer):
        s1 = tracer.start_span("s1")
        tracer.end_span(s1)
        s2 = tracer.start_span("s2")  # still open
        completed = tracer.spans
        assert len(completed) == 1
        assert completed[0].name == "s1"

    def test_all_spans_includes_open(self, tracer):
        s1 = tracer.start_span("s1")
        tracer.end_span(s1)
        s2 = tracer.start_span("s2")
        all_spans = tracer.all_spans
        assert len(all_spans) == 2

    def test_get_trace(self, tracer):
        s1 = tracer.start_span("root")
        s2 = tracer.start_span("child", parent=s1)
        tracer.end_span(s1)
        tracer.end_span(s2)
        trace_spans = tracer.get_trace(s1.trace_id)
        assert len(trace_spans) == 2

    def test_clear(self, tracer):
        s = tracer.start_span("test")
        tracer.end_span(s)
        assert len(tracer.spans) == 1
        tracer.clear()
        assert len(tracer.spans) == 0

    def test_resource_attributes(self, tracer):
        span = tracer.start_span("test")
        assert "service.name" in span.resource
        assert span.resource["service.name"] == "test-agent"

    def test_sampler_ratio_0(self):
        tracer = Tracer(service_name="test", sampler_ratio=0.0)
        span = tracer.start_span("test")
        # All spans should be marked as not sampled (trace_id all zeros)
        assert span.trace_id == "0" * TRACE_ID_LEN

    def test_sampler_ratio_1(self):
        tracer = Tracer(service_name="test", sampler_ratio=1.0)
        span = tracer.start_span("test")
        assert span.trace_id != "0" * TRACE_ID_LEN


# ---- SpanContext (context manager) ------------------------------------------

class TestSpanContext:
    def test_context_manager_ends_span(self, tracer):
        with tracer.start_span_ctx("test") as span:
            assert span.is_open is True
        assert span.is_open is False

    def test_context_manager_exception(self, tracer):
        with pytest.raises(ValueError):
            with tracer.start_span_ctx("test") as span:
                raise ValueError("test error")
        assert span.status_code == STATUS_ERROR
        assert "test error" in span.status_message
        assert len(span.events) == 1
        assert span.events[0]["name"] == "exception"

    def test_context_manager_nesting(self, tracer):
        with tracer.start_span_ctx("parent") as parent:
            assert tracer._current_span is parent
            with tracer.start_span_ctx("child") as child:
                assert tracer._current_span is child
                assert child.trace_id == parent.trace_id
                assert child.parent_span_id == parent.span_id
            # After child exits, current should be parent again
            assert tracer._current_span is parent
        # After parent exits, current should be None
        assert tracer._current_span is None


# ---- OTLP Exporter ----------------------------------------------------------

class TestOTLPExporter:
    def test_buffer_mode(self):
        exporter = OTLPExporter()
        tracer = Tracer(service_name="test")
        s1 = tracer.start_span("s1")
        tracer.end_span(s1)

        result = exporter.export(tracer.spans)
        assert result is True
        buffered = exporter.get_buffered()
        assert len(buffered) == 1
        assert "resourceSpans" in buffered[0]

    def test_empty_export(self):
        exporter = OTLPExporter()
        assert exporter.export([]) is True
        assert len(exporter.get_buffered()) == 0

    def test_otlp_format(self):
        exporter = OTLPExporter()
        tracer = Tracer(service_name="test-agent")
        s = tracer.start_span("test", session_id="s1")
        tracer.end_span(s)

        exporter.export(tracer.spans)
        data = exporter.get_buffered()[0]

        assert "resourceSpans" in data
        rs = data["resourceSpans"][0]
        assert rs["scope"]["name"] == "matrix-agent"
        assert len(rs["spans"]) == 1
        assert rs["spans"][0]["name"] == "test"
        assert "service.name" in str(rs["resource"]["attributes"])

    def test_clear_buffer(self):
        exporter = OTLPExporter()
        tracer = Tracer()
        s = tracer.start_span("test")
        tracer.end_span(s)
        exporter.export(tracer.spans)
        assert len(exporter.get_buffered()) == 1
        exporter.clear_buffer()
        assert len(exporter.get_buffered()) == 0


# ---- AnyValue conversion ----------------------------------------------------

class TestAnyValue:
    def test_bool(self):
        assert _to_any_value(True) == {"boolValue": True}
        assert _to_any_value(False) == {"boolValue": False}

    def test_int(self):
        assert _to_any_value(42) == {"intValue": "42"}

    def test_float(self):
        assert _to_any_value(3.14) == {"doubleValue": 3.14}

    def test_string(self):
        assert _to_any_value("hello") == {"stringValue": "hello"}

    def test_list(self):
        result = _to_any_value([1, "two"])
        assert "arrayValue" in result
        assert len(result["arrayValue"]["values"]) == 2

    def test_dict(self):
        result = _to_any_value({"key": "value"})
        assert "kvlistValue" in result
        assert result["kvlistValue"]["values"][0]["key"] == "key"

    def test_fallback(self):
        class Custom:
            def __str__(self):
                return "custom"
        result = _to_any_value(Custom())
        assert result == {"stringValue": "custom"}


# ---- Legacy event normalization ---------------------------------------------

class TestLegacyNormalization:
    def test_normalize_legacy_event(self):
        event = {
            "session_id": "s1",
            "agent_id": "commander",
            "tool_name": "web_search",
            "node_name": "react",
            "arguments": {"query": "test"},
            "result": "search results here",
            "error": None,
        }
        attrs = normalize_legacy_event(event)
        assert attrs["session.id"] == "s1"
        assert attrs["agent.id"] == "commander"
        assert attrs["tool.name"] == "web_search"
        assert attrs["node.name"] == "react"
        assert attrs["args.query"] == "test"
        assert "result.preview" in attrs

    def test_normalize_legacy_error(self):
        event = {"error": "timeout", "ok": False}
        attrs = normalize_legacy_event(event)
        assert attrs["error.message"] == "timeout"

    def test_normalize_legacy_http_path(self):
        event = {"path": "/api/chat"}
        attrs = normalize_legacy_event(event)
        assert attrs["http.path"] == "/api/chat"

    def test_event_to_span_kind(self):
        assert event_to_span_kind("tool_call") == SPAN_KIND_CLIENT
        assert event_to_span_kind("http_request") == SPAN_KIND_SERVER
        assert event_to_span_kind("span_start") == SPAN_KIND_INTERNAL
        assert event_to_span_kind("unknown") == SPAN_KIND_INTERNAL

    def test_event_to_status(self):
        code, msg = event_to_status({"ok": True})
        assert code == STATUS_OK
        code, msg = event_to_status({"ok": False, "error": "failed"})
        assert code == STATUS_ERROR
        assert "failed" in msg
        code, msg = event_to_status({})
        assert code == STATUS_UNSET


# ---- TraceStore OTel integration --------------------------------------------

class TestTraceStoreOTel:
    def test_start_and_end_span(self, store):
        span = store.start_span("test_span", session_id="s1")
        assert span.name == "test_span"
        assert span.attributes["session.id"] == "s1"
        assert span.is_open is True

        store.end_span(span)
        assert span.is_open is False
        assert span.status_code == STATUS_OK  # defaults to OK

    def test_record_span_to_db(self, store):
        span = store.start_span("db_test", session_id="s1", agent_id="commander")
        span.set_attribute("tool.name", "web_search")
        span.add_event("tool.started")
        store.end_span(span)

        # Query it back
        results = store.query_spans(session_id="s1")
        assert len(results) == 1
        assert results[0]["name"] == "db_test"
        assert results[0]["session_id"] == "s1"
        assert results[0]["elapsed_ms"] > 0

    def test_query_spans_by_trace_id(self, store):
        # Use explicit parent to ensure both spans share the same trace_id
        s1 = store.start_span("span1", session_id="s1")
        s2 = store.start_span("span2", session_id="s1", parent=s1)
        store.end_span(s1)
        store.end_span(s2)

        results = store.query_spans(trace_id=s1.trace_id)
        assert len(results) == 2

    def test_query_spans_empty(self, store):
        results = store.query_spans(session_id="nonexistent")
        assert len(results) == 0

    def test_record_span_attributes_persisted(self, store):
        span = store.start_span("attr_test", session_id="s1")
        span.set_attribute("custom.attr", "value123")
        span.set_attribute("numeric.attr", 42)
        store.end_span(span)

        results = store.query_spans(session_id="s1")
        assert len(results) == 1
        attrs = results[0]["attributes"]
        assert attrs["custom.attr"] == "value123"
        assert attrs["numeric.attr"] == 42

    def test_record_span_events_persisted(self, store):
        span = store.start_span("event_test", session_id="s1")
        span.add_event("event1", detail="something happened")
        store.end_span(span)

        results = store.query_spans(session_id="s1")
        assert len(results) == 1
        events = results[0]["events"]
        assert len(events) == 1
        assert events[0]["name"] == "event1"

    def test_legacy_record_still_works(self, store):
        """Legacy record() method should still work alongside OTel spans."""
        store.record({
            "session_id": "s1",
            "event_type": "tool_call",
            "node_name": "react",
            "agent_id": "commander",
            "tool_name": "web_search",
            "ok": True,
            "elapsed_ms": 123.4,
            "ts": "2026-07-22T10:00:00Z",
        })
        results = store.query(session_id="s1")
        assert len(results) == 1
        assert results[0]["event_type"] == "tool_call"

    def test_otlp_export_buffered(self, store):
        """When OTLP export is disabled, spans are not exported."""
        span = store.start_span("export_test", session_id="s1")
        store.end_span(span)

        # With otlp_export=False (default), buffer should be empty
        exports = store.export_otlp()
        assert len(exports) == 0  # exporter not configured to buffer

    def test_otlp_export_enabled(self, tmp_path):
        """When OTLP export is enabled, spans are buffered for export."""
        store = TraceStore(
            tmp_path / "otel_export_test.db",
            otlp_export=True,
        )
        span = store.start_span("export_test", session_id="s1")
        store.end_span(span)

        exports = store.export_otlp()
        assert len(exports) == 1
        assert "resourceSpans" in exports[0]


# ---- Config integration -----------------------------------------------------

class TestOTelConfig:
    def test_config_has_otel_fields(self):
        from matrix.config import AgentConfig
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(AgentConfig)}
        assert "otel_exporter_endpoint" in fields
        assert "otel_export" in fields
        assert fields["otel_export"].default is False

    def test_env_var_otlp_endpoint(self, monkeypatch):
        from matrix.config import load_config
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
        monkeypatch.setenv("OTEL_EXPORT", "true")
        monkeypatch.setenv("JWT_SECRET", "test-secret")
        monkeypatch.chdir("/Users/liqiang/code/personal-system/personal-agent")
        config = load_config()
        assert config.otel_exporter_endpoint == "http://localhost:4318/v1/traces"
        assert config.otel_export is True

    def test_env_var_otlp_disabled_by_default(self, monkeypatch):
        from matrix.config import load_config
        monkeypatch.setenv("JWT_SECRET", "test-secret")
        monkeypatch.chdir("/Users/liqiang/code/personal-system/personal-agent")
        config = load_config()
        assert config.otel_export is False
        assert config.otel_exporter_endpoint == ""


# ---- ID generation ----------------------------------------------------------

class TestIDGeneration:
    def test_trace_id_length(self):
        tid = _generate_trace_id()
        assert len(tid) == TRACE_ID_LEN

    def test_span_id_length(self):
        sid = _generate_span_id()
        assert len(sid) == SPAN_ID_LEN

    def test_trace_id_not_all_zeros(self):
        tid = _generate_trace_id()
        assert tid != "0" * TRACE_ID_LEN

    def test_span_id_not_all_zeros(self):
        sid = _generate_span_id()
        assert sid != "0" * SPAN_ID_LEN

    def test_unique_trace_ids(self):
        ids = {_generate_trace_id() for _ in range(100)}
        assert len(ids) == 100  # all unique (with overwhelming probability)

    def test_unique_span_ids(self):
        ids = {_generate_span_id() for _ in range(100)}
        assert len(ids) == 100
