"""
Policy checker node — for each UserClaim without coverage, runs:
  1. Query reformulation (fast_llm)
  2. Parallel Pinecone retrieval
  3. Coverage assessment (fast_llm)
"""
from __future__ import annotations

import json
import logging

from gateway.llm_gateway import fast_llm
from models.state import AgentState, PolicyCoverage
from rag.retriever import retrieve

logger = logging.getLogger(__name__)

# ── Query reformulation ───────────────────────────────────────────────────────

_REFORMULATION_SYSTEM = """\
You are a query reformulation agent for a vector database search.
Given a customer's claim description, generate 2–3 concise search queries
that would retrieve the most relevant warranty policy sections.

Return JSON only:
{"queries": ["query 1", "query 2", "query 3"]}
"""


def _reformulate_queries(component: str, statement: str) -> list[str]:
    text = f"Component: {component}\nIssue: {statement}"
    messages = [
        {"role": "system", "content": _REFORMULATION_SYSTEM},
        {"role": "user", "content": text},
    ]
    raw = fast_llm(messages, json_mode=True)
    try:
        data = json.loads(raw)
        queries = data.get("queries", [])
        if isinstance(queries, list) and queries:
            return [str(q) for q in queries[:3]]
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: simple queries
    return [
        f"{component} warranty coverage",
        f"{component} {statement}",
    ]


# ── Coverage assessment ───────────────────────────────────────────────────────

_COVERAGE_SYSTEM = """\
You are a warranty policy analyst. Given retrieved policy sections and a customer's claim,
determine whether the claim is covered.

Return JSON only:
{
  "covered": <true|false>,
  "coverage_months": <integer or null>,
  "exclusion_reason": "<reason if excluded, else null>",
  "policy_clauses": ["§2", "§3.1"]
}

Rules:
- "covered" = true only if the component is explicitly listed in the coverage table AND
  no applicable exclusion applies.
- Set "exclusion_reason" when "covered" is false and an exclusion is relevant.
- List all relevant section references in "policy_clauses".
- If the policy context is empty or ambiguous, set "covered" to false with
  "exclusion_reason" = "Insufficient policy context to determine coverage."
"""


def _assess_coverage(component: str, statement: str, chunks: list[dict]) -> dict:
    if not chunks:
        return PolicyCoverage(
            covered=False,
            exclusion_reason="No relevant policy sections found for this component.",
        ).model_dump()

    context = "\n---\n".join(c["text"] for c in chunks)
    user_content = (
        f"Component: {component}\n"
        f"Customer statement: {statement}\n\n"
        f"Policy sections:\n{context}"
    )
    messages = [
        {"role": "system", "content": _COVERAGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    raw = fast_llm(messages, json_mode=True)
    try:
        data = json.loads(raw)
        return PolicyCoverage(**data).model_dump()
    except Exception:
        logger.warning("Coverage assessment parse failed for %s", component)
        return PolicyCoverage(
            covered=False,
            exclusion_reason="Unable to assess coverage due to a processing error.",
        ).model_dump()


# ── Main node ─────────────────────────────────────────────────────────────────

def policy_checker_node(state: AgentState) -> dict:
    user_claims: list[dict] = list(state.get("user_claims", []))
    all_chunks: list[dict] = []
    all_clauses: list[str] = []

    for i, claim in enumerate(user_claims):
        # Skip claims that already have coverage assessed
        if claim.get("policy_coverage") is not None:
            continue

        component = claim.get("component", "")
        statement = claim.get("verbatim_statement", "") or component

        # 1. Reformulate queries
        queries = _reformulate_queries(component, statement)

        # 2. Retrieve chunks for each query, deduplicate by text
        seen_texts: set[str] = set()
        chunks: list[dict] = []
        for q in queries:
            for chunk in retrieve(q):
                if chunk["text"] not in seen_texts:
                    seen_texts.add(chunk["text"])
                    chunks.append(chunk)
        chunks.sort(key=lambda c: c["score"], reverse=True)
        chunks = chunks[:5]  # top 5 after dedup
        all_chunks.extend(chunks)

        # 3. Assess coverage
        coverage = _assess_coverage(component, statement, chunks)
        user_claims[i] = dict(claim)
        user_claims[i]["policy_coverage"] = coverage

        # Collect clause references
        all_clauses.extend(coverage.get("policy_clauses", []))

    policy_context = [c["text"] for c in all_chunks]
    # Deduplicate clauses
    seen: set[str] = set()
    unique_clauses: list[str] = []
    for cl in all_clauses:
        if cl not in seen:
            seen.add(cl)
            unique_clauses.append(cl)

    return {
        "user_claims": user_claims,
        "policy_context": policy_context,
        "policy_clauses": unique_clauses,
    }
