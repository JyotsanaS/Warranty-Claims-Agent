"""
Coverage service owning policy query reformulation and coverage assessment.
"""
from __future__ import annotations

import json
import logging

from gateway.llm_gateway import fast_llm
from agent.prompt_store import get_prompt
from models.state import PolicyCoverage
from observability.tracing import start_span

logger = logging.getLogger(__name__)


def reformulate_queries(component: str, statement: str) -> list[str]:
    with start_span(
        "coverage.reformulate_queries",
        {"claim.component": component, "claim.statement": statement},
    ) as span:
        text = f"Component: {component}\nIssue: {statement}"
        messages = [
            {"role": "system", "content": get_prompt("coverage_service", "reformulation_system")},
            {"role": "user", "content": text},
        ]
        raw = fast_llm(messages, json_mode=True)
        try:
            data = json.loads(raw)
            queries = data.get("queries", [])
            if isinstance(queries, list) and queries:
                result = [str(q) for q in queries[:3]]
                span.set_attribute("coverage.query_count", len(result))
                return result
        except (json.JSONDecodeError, TypeError) as exc:
            span.record_exception(exc)

        fallback = [
            f"{component} warranty coverage",
            f"{component} {statement}",
        ]
        span.set_attribute("coverage.query_count", len(fallback))
        return fallback


def assess_coverage(component: str, statement: str, chunks: list[dict]) -> dict:
    with start_span(
        "coverage.assess",
        {
            "claim.component": component,
            "claim.statement": statement,
            "coverage.chunk_count": len(chunks),
        },
    ) as span:
        if not chunks:
            result = PolicyCoverage(
                covered=False,
                exclusion_reason="No relevant policy sections found for this component.",
            ).model_dump()
            span.set_attribute("coverage.covered", False)
            return result

        context = "\n---\n".join(chunk["text"] for chunk in chunks)
        user_content = (
            f"Component: {component}\n"
            f"Customer statement: {statement}\n\n"
            f"Policy sections:\n{context}"
        )
        messages = [
            {"role": "system", "content": get_prompt("coverage_service", "coverage_system")},
            {"role": "user", "content": user_content},
        ]
        raw = fast_llm(messages, json_mode=True)
        try:
            data = json.loads(raw)
            result = PolicyCoverage(**data).model_dump()
        except Exception as exc:
            span.record_exception(exc)
            logger.warning("Coverage assessment parse failed for %s", component)
            result = PolicyCoverage(
                covered=False,
                exclusion_reason="Unable to assess coverage due to a processing error.",
            ).model_dump()
            span.set_attribute("coverage.covered", False)
            return result

        # Hallucination check — only on approvals; rejections are conservative by default
        if result.get("covered"):
            from agent.guardrails.hallucination import is_grounded
            grounded = is_grounded(
                value=raw,
                reference=context,
                query=f"{component}: {statement}",
            )
            if not grounded:
                logger.warning(
                    "Coverage hallucination detected for %s — retrying with stricter prompt", component
                )
                stricter_messages = [
                    {
                        "role": "system",
                        "content": get_prompt("coverage_service", "coverage_system")
                        + "\n\nIMPORTANT: Only cite policy clauses that appear verbatim "
                        "in the provided policy sections. Do not infer or paraphrase.",
                    },
                    messages[1],
                ]
                raw_retry = fast_llm(stricter_messages, json_mode=True)
                try:
                    data_retry = json.loads(raw_retry)
                    result_retry = PolicyCoverage(**data_retry).model_dump()
                except Exception:
                    result_retry = None

                if result_retry and result_retry.get("covered"):
                    grounded_retry = is_grounded(
                        value=raw_retry,
                        reference=context,
                        query=f"{component}: {statement}",
                    )
                else:
                    grounded_retry = bool(result_retry and not result_retry.get("covered"))

                if grounded_retry and result_retry:
                    result = result_retry
                    span.set_attribute("coverage.hallucination_retry_passed", True)
                else:
                    result = PolicyCoverage(
                        covered=False,
                        exclusion_reason=(
                            "Coverage not grounded in retrieved policy. Manual review required."
                        ),
                    ).model_dump()
                    span.set_attribute("coverage.hallucination_fallback", True)
                    logger.warning(
                        "Coverage hallucination fallback applied for %s — downgraded to not covered",
                        component,
                    )

        span.set_attribute("coverage.covered", bool(result.get("covered")))
        return result
