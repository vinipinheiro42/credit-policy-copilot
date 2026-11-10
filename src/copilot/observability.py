"""Structured JSON logs with a request id, and OpenTelemetry tracing.

Tracing exporter is chosen with COPILOT_OTEL_EXPORTER: "none" (default; spans still created so tests can
capture them), "console" (spans printed as JSON to stdout) or "otlp" (OTLP/HTTP, endpoint from the standard
OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_TRACES_ENDPOINT variables; needs the `otel` extra).
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_configured_tracing = False
TRACER_NAME = "credit-policy-copilot"


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def set_request_id(rid: str | None) -> contextvars.Token:
    return _request_id.set(rid)


def reset_request_id(token: contextvars.Token) -> None:
    _request_id.reset(token)


def get_request_id() -> str | None:
    return _request_id.get()


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Extra structured fields go in ``extra={"fields": {...}}``."""

    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or get_request_id(),
        }
        span = trace.get_current_span().get_span_context()
        if span.is_valid:
            doc["trace_id"] = f"{span.trace_id:032x}"
            doc["span_id"] = f"{span.span_id:016x}"
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            doc.update(fields)
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    root = logging.getLogger("copilot")
    root.setLevel(level.upper())
    if any(getattr(h, "_copilot", False) for h in root.handlers):
        return
    h = logging.StreamHandler(sys.stderr)
    h._copilot = True  # type: ignore[attr-defined]
    h.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    root.addHandler(h)
    root.propagate = False


def configure_tracing(exporter: str = "none", service_name: str = TRACER_NAME) -> None:
    """Idempotent. Installs a TracerProvider unless the application already installed one."""
    global _configured_tracing
    if _configured_tracing:
        return
    _configured_tracing = True
    if exporter == "none":
        # keep whatever provider exists (tests install an in-memory one); otherwise spans are no-ops
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        raise ValueError(f"unknown COPILOT_OTEL_EXPORTER={exporter!r} (none|console|otlp)")
    trace.set_tracer_provider(provider)


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)
