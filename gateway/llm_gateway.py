"""
LLM Gateway — single module owning all model interactions.
No other module imports a provider SDK directly.
"""
from __future__ import annotations

import logging
from typing import Any

import os

import litellm

logger = logging.getLogger(__name__)
litellm.suppress_debug_info = True

# Enable Langfuse as a LiteLLM callback — captures prompts, responses, token counts.
# No-ops gracefully if Langfuse is not configured.
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

# LiteLLM's Langfuse callback reads credentials from os.environ, not from the
# pydantic Settings object.  Populate them here so tracing works.
try:
    from app.config import settings as _settings
    if _settings.langfuse_secret_key:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", _settings.langfuse_secret_key)
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", _settings.langfuse_public_key)
        os.environ.setdefault("LANGFUSE_HOST", _settings.langfuse_base_url)
except Exception:
    pass


def _langfuse_metadata(name: str) -> dict:
    """Build the metadata dict that associates a LiteLLM call with the current Langfuse trace."""
    from gateway.langfuse_client import get_trace_context
    trace_id, span_id = get_trace_context()
    if not trace_id:
        return {}
    meta: dict = {"trace_id": trace_id, "generation_name": name}
    if span_id:
        meta["parent_observation_id"] = span_id
    return meta


def _api_key() -> str:
    from app.config import settings
    return settings.groq_api_key


def _fast_model() -> str:
    from app.config import settings
    return settings.llm_fast_model


def _main_model() -> str:
    from app.config import settings
    return settings.llm_main_model


def _vision_model() -> str:
    from app.config import settings
    return settings.llm_vision_model


def fast_llm(
    messages: list[dict],
    temperature: float = 0.0,
    json_mode: bool = False,
    **kwargs: Any,
) -> str:
    """Call the fast model (routing, extractors, query reformulation). Returns text."""
    extra: dict[str, Any] = {}
    if json_mode:
        extra["response_format"] = {"type": "json_object"}
    try:
        resp = litellm.completion(
            model=_fast_model(),
            messages=messages,
            temperature=temperature,
            api_key=_api_key(),
            metadata=_langfuse_metadata("fast_llm"),
            **extra,
            **kwargs,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.error("fast_llm failed: %s", exc)
        return ""


def main_llm(
    messages: list[dict],
    temperature: float = 0.3,
    **kwargs: Any,
) -> str:
    """Call the main model (agent_respond, empathy, claim_decision). Returns text."""
    try:
        resp = litellm.completion(
            model=_main_model(),
            messages=messages,
            temperature=temperature,
            api_key=_api_key(),
            metadata=_langfuse_metadata("main_llm"),
            **kwargs,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.error("main_llm failed: %s", exc)
        return ""


def vision_llm(
    messages: list[dict],
    temperature: float = 0.0,
    **kwargs: Any,
) -> str:
    """Call the vision model (vision_analysis only). Returns text."""
    try:
        resp = litellm.completion(
            model=_vision_model(),
            messages=messages,
            temperature=temperature,
            api_key=_api_key(),
            metadata=_langfuse_metadata("vision_llm"),
            **kwargs,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.error("vision_llm failed: %s", exc)
        return ""


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed texts using the configured embedding model (local SentenceTransformer)."""
    from app.config import settings
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.llm_embedding_model)
    return model.encode(texts, normalize_embeddings=True).tolist()
