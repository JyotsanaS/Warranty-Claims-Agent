"""
Langfuse client — singleton + per-request context variables.
Uses the Langfuse v3+ API (get_client / start_as_current_observation).
"""
from __future__ import annotations

import logging
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Per-request context — set by the runner, read by the gateway
_trace_id_var: ContextVar[str | None] = ContextVar("lf_trace_id", default=None)
_span_id_var: ContextVar[str | None] = ContextVar("lf_span_id", default=None)

_initialized: bool = False


def get_langfuse():
    """
    Return the shared Langfuse v3+ client (via get_client()), or None if
    credentials are missing / Langfuse is unavailable.

    On first call, initialises the Langfuse singleton and also exports the
    credentials into os.environ so LiteLLM's 'langfuse' callback can find them.
    """
    global _initialized
    try:
        from app.config import settings
        if not settings.langfuse_secret_key or not settings.langfuse_public_key:
            return None

        if not _initialized:
            import os
            # LiteLLM's Langfuse callback reads from os.environ, not from Settings.
            os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
            os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_base_url)
            # Initialise the Langfuse v3+ singleton.
            from langfuse import Langfuse
            Langfuse(
                secret_key=settings.langfuse_secret_key,
                public_key=settings.langfuse_public_key,
                host=settings.langfuse_base_url,
            )
            _initialized = True
            logger.info("Langfuse client initialised (host=%s)", settings.langfuse_base_url)

        from langfuse import get_client
        return get_client()

    except Exception as exc:
        logger.warning("Langfuse init failed — tracing disabled: %s", exc)
        return None


def set_trace_context(trace_id: str, span_id: str | None = None) -> None:
    """Called by the runner to set the current trace and optional parent span."""
    _trace_id_var.set(trace_id)
    _span_id_var.set(span_id)


def get_trace_context() -> tuple[str | None, str | None]:
    """Called by the gateway to retrieve (trace_id, span_id)."""
    return _trace_id_var.get(), _span_id_var.get()
