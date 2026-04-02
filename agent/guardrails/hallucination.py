"""
Hallucination guardrail using Guardrails AI GroundedAIHallucination validator.

Fail-open: if the validator itself errors (misconfiguration, network issue, etc.),
the check is skipped and a warning is emitted. The claim flow is never blocked
by a guardrail failure.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _fast_llm_callable(prompt: str, **kwargs) -> str:
    """Adapter: wraps fast_llm to the single-string-prompt signature Guardrails expects."""
    from gateway.llm_gateway import fast_llm
    return fast_llm([{"role": "user", "content": prompt}])


def is_grounded(value: str, reference: str, query: str) -> bool:
    """
    Returns True if value is grounded in reference for the given query.
    Returns False if hallucination is detected.

    Fail-open: unexpected validator errors return True and emit a warning
    so the claim flow is never blocked by a guardrail failure.
    """
    if not value or not reference:
        return True

    try:
        from guardrails import Guard
        from guardrails.hub import GroundedAIHallucination
        from guardrails.errors import ValidationError

        guard = Guard().use(
            GroundedAIHallucination,
            llm_callable=_fast_llm_callable,
            on_fail="exception",
        )
        guard.validate(
            value,
            metadata={
                "query": query,
                "sources": [reference],
            },
        )
        return True

    except Exception as exc:
        exc_name = type(exc).__name__
        # Guardrails raises ValidationError (or FailedReaskError) on detected hallucination
        if "Validation" in exc_name or "FailedReask" in exc_name or "PassResult" in exc_name:
            try:
                from guardrails.errors import ValidationError as GValidationError
                if isinstance(exc, GValidationError):
                    logger.info("Hallucination detected by guardrail: %s", exc)
                    return False
            except ImportError:
                pass
            # If we can't import the specific type, treat "Validation" in name as failure
            if "Validation" in exc_name or "FailedReask" in exc_name:
                logger.info("Hallucination detected by guardrail: %s", exc)
                return False

        logger.warning("Hallucination guardrail error (fail-open): %s: %s", exc_name, exc)
        return True
