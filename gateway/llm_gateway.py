"""
LLM Gateway — single module owning all model interactions.
No other module imports a provider SDK directly.
"""
from __future__ import annotations

import logging
import json
from typing import Any

import litellm

from observability.tracing import start_span

logger = logging.getLogger(__name__)
litellm.suppress_debug_info = True


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"... <truncated {len(value) - limit} chars>"


def _safe_dump(value: Any, limit: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        text = str(value)
    return _truncate(text, limit)


def _summarize_messages(messages: list[dict], limit: int) -> str:
    summary: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, list):
            content_summary: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    content_summary.append({"type": type(item).__name__})
                    continue
                item_type = item.get("type", "unknown")
                if item_type == "text":
                    content_summary.append(
                        {
                            "type": "text",
                            "text": _truncate(str(item.get("text", "")), min(limit, 600)),
                        }
                    )
                elif item_type == "image_url":
                    url = ""
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        url = str(image_url.get("url", ""))
                    content_summary.append(
                        {
                            "type": "image_url",
                            "url_prefix": url[:80],
                            "url_length": len(url),
                        }
                    )
                else:
                    content_summary.append({"type": item_type, "value": _safe_dump(item, 400)})
            summary.append({"role": role, "content": content_summary})
        else:
            summary.append(
                {
                    "role": role,
                    "content": _truncate(str(content or ""), min(limit, 1000)),
                }
            )
    return _safe_dump(summary, limit)


def _should_log_payloads() -> bool:
    from app.config import settings
    return settings.log_http_bodies


def _payload_log_limit() -> int:
    from app.config import settings
    return settings.log_http_body_max_chars


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
    model = _fast_model()
    with start_span(
        "llm.fast",
        {
            "llm.model": model,
            "llm.temperature": temperature,
            "llm.json_mode": json_mode,
            "llm.message_count": len(messages),
        },
    ) as span:
        try:
            if _should_log_payloads():
                span.set_attribute(
                    "llm.prompt", _summarize_messages(messages, _payload_log_limit())
                )
            resp = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                api_key=_api_key(),
                **extra,
                **kwargs,
            )
            text = resp.choices[0].message.content or ""
            span.set_attribute("llm.response_length", len(text))
            if _should_log_payloads():
                span.set_attribute("llm.response", _truncate(text, _payload_log_limit()))
            if resp.usage:
                span.set_attribute("llm.tokens.prompt", resp.usage.prompt_tokens or 0)
                span.set_attribute("llm.tokens.completion", resp.usage.completion_tokens or 0)
                span.set_attribute("llm.tokens.total", resp.usage.total_tokens or 0)
            return text
        except Exception as exc:
            span.record_exception(exc)
            logger.error("fast_llm failed: %s", exc)
            return ""


def main_llm(
    messages: list[dict],
    temperature: float = 0.3,
    **kwargs: Any,
) -> str:
    """Call the main model (agent_respond, empathy, claim_decision). Returns text."""
    model = _main_model()
    with start_span(
        "llm.main",
        {
            "llm.model": model,
            "llm.temperature": temperature,
            "llm.message_count": len(messages),
        },
    ) as span:
        try:
            if _should_log_payloads():
                span.set_attribute(
                    "llm.prompt", _summarize_messages(messages, _payload_log_limit())
                )
            resp = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                api_key=_api_key(),
                **kwargs,
            )
            text = resp.choices[0].message.content or ""
            span.set_attribute("llm.response_length", len(text))
            if _should_log_payloads():
                span.set_attribute("llm.response", _truncate(text, _payload_log_limit()))
            if resp.usage:
                span.set_attribute("llm.tokens.prompt", resp.usage.prompt_tokens or 0)
                span.set_attribute("llm.tokens.completion", resp.usage.completion_tokens or 0)
                span.set_attribute("llm.tokens.total", resp.usage.total_tokens or 0)
            return text
        except Exception as exc:
            span.record_exception(exc)
            logger.error("main_llm failed: %s", exc)
            return ""


def vision_llm(
    messages: list[dict],
    temperature: float = 0.0,
    **kwargs: Any,
) -> str:
    """Call the vision model (vision_analysis only). Returns text."""
    model = _vision_model()
    if _should_log_payloads():
        logger.info(
            "vision_llm request model=%s temperature=%s kwargs=%s messages=%s",
            model,
            temperature,
            _safe_dump(kwargs, _payload_log_limit()),
            _summarize_messages(messages, _payload_log_limit()),
        )
    with start_span(
        "llm.vision",
        {
            "llm.model": model,
            "llm.temperature": temperature,
            "llm.message_count": len(messages),
        },
    ) as span:
        try:
            resp = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                api_key=_api_key(),
                **kwargs,
            )
            text = resp.choices[0].message.content or ""
            span.set_attribute("llm.response_length", len(text))
            if resp.usage:
                span.set_attribute("llm.tokens.prompt", resp.usage.prompt_tokens or 0)
                span.set_attribute("llm.tokens.completion", resp.usage.completion_tokens or 0)
                span.set_attribute("llm.tokens.total", resp.usage.total_tokens or 0)
            if _should_log_payloads():
                logger.info(
                    "vision_llm response model=%s body=%s",
                    model,
                    _truncate(text, _payload_log_limit()),
                )
            return text
        except Exception as exc:
            span.record_exception(exc)
            if _should_log_payloads():
                logger.error(
                    "vision_llm provider error model=%s error=%s",
                    model,
                    repr(exc),
                )
            logger.error("vision_llm failed for model %s: %r", model, exc)
            return ""


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed texts using the configured embedding model (local SentenceTransformer)."""
    from app.config import settings
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.llm_embedding_model)
    return model.encode(texts, normalize_embeddings=True).tolist()
