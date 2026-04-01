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
            span.set_attribute("coverage.covered", bool(result.get("covered")))
            return result
        except Exception as exc:
            span.record_exception(exc)
            logger.warning("Coverage assessment parse failed for %s", component)
            result = PolicyCoverage(
                covered=False,
                exclusion_reason="Unable to assess coverage due to a processing error.",
            ).model_dump()
            span.set_attribute("coverage.covered", False)
            return result
