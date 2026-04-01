from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult


def _ns_to_iso8601(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    return str(value)


def _serialize_span(span: ReadableSpan) -> dict[str, Any]:
    parent_span_id = None
    if span.parent is not None:
        parent_span_id = f"{span.parent.span_id:016x}"

    return {
        "name": span.name,
        "trace_id": f"{span.context.trace_id:032x}",
        "span_id": f"{span.context.span_id:016x}",
        "parent_span_id": parent_span_id,
        "start_time": _ns_to_iso8601(span.start_time),
        "end_time": _ns_to_iso8601(span.end_time),
        "attributes": _json_safe(dict(span.attributes or {})),
        "status": {
            "status_code": str(span.status.status_code),
            "description": span.status.description,
        },
        "events": [
            {
                "name": event.name,
                "timestamp": _ns_to_iso8601(event.timestamp),
                "attributes": _json_safe(dict(event.attributes or {})),
            }
            for event in span.events
        ],
        "resource": _json_safe(dict(span.resource.attributes or {})),
        "instrumentation_scope": {
            "name": getattr(span.instrumentation_scope, "name", ""),
            "version": getattr(span.instrumentation_scope, "version", ""),
        },
    }


class InMemoryTraceExporter(SpanExporter):
    """Collect completed spans in memory so the app can persist them per session."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._spans_by_trace_id: dict[str, list[dict[str, Any]]] = {}

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            for span in spans:
                trace_id = f"{span.context.trace_id:032x}"
                self._spans_by_trace_id.setdefault(trace_id, []).append(_serialize_span(span))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def pop_trace(self, trace_id: str) -> list[dict[str, Any]]:
        with self._lock:
            spans = self._spans_by_trace_id.pop(trace_id, [])
        spans.sort(key=lambda item: (item.get("start_time") or "", item.get("span_id") or ""))
        return spans


_exporter = InMemoryTraceExporter()
_initialized = False


def setup_tracing() -> None:
    global _initialized
    if _initialized:
        return

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_exporter))
    trace.set_tracer_provider(provider)
    _initialized = True


def get_tracer(name: str):
    setup_tracing()
    return trace.get_tracer(name)


def start_session_trace(session_id: str, app_trace_id: str):
    """
    Create a root span for a single agent turn without attaching it as the
    current context for the whole streaming response lifecycle.
    """
    setup_tracing()
    tracer = get_tracer("chargepoint.session")
    span = tracer.start_span(
        "warranty_agent.turn",
        attributes={
            "session.id": session_id,
            "app.trace_id": app_trace_id,
        },
    )
    return span


def finish_session_trace(span) -> None:
    if span is not None:
        span.end()


from contextlib import contextmanager


@contextmanager
def start_span(name: str, attributes: dict[str, Any] | None = None):
    tracer = get_tracer("chargepoint")
    with tracer.start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            span.set_attribute(key, _json_safe(value))
        yield span


@contextmanager
def start_linked_span(name: str, parent_span, attributes: dict[str, Any] | None = None):
    tracer = get_tracer("chargepoint")
    parent_context = None
    if parent_span is not None:
        parent_context = trace.set_span_in_context(parent_span)
    with tracer.start_as_current_span(
        name,
        context=parent_context,
    ):
        for key, value in (attributes or {}).items():
            span.set_attribute(key, _json_safe(value))
        yield span


def pop_serialized_trace(trace_id: str | None) -> list[dict[str, Any]]:
    if not trace_id:
        return []
    return _exporter.pop_trace(trace_id)
