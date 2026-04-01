"""
Policy checker node — thin state adapter around coverage and retrieval services.
"""
from __future__ import annotations

from agent.services.coverage_service import assess_coverage, reformulate_queries
from agent.tools.retrieval_tool import retrieve_policy_chunks
from models.state import AgentState


def _retrieve_deduped_chunks(queries: list[str]) -> list[dict]:
    seen_texts: set[str] = set()
    chunks: list[dict] = []
    for query in queries:
        for chunk in retrieve_policy_chunks(query):
            if chunk["text"] in seen_texts:
                continue
            seen_texts.add(chunk["text"])
            chunks.append(chunk)
    chunks.sort(key=lambda item: item["score"], reverse=True)
    return chunks[:5]


# ── Main node ─────────────────────────────────────────────────────────────────

def policy_checker_node(state: AgentState) -> dict:
    user_claims: list[dict] = list(state.get("user_claims", []))
    all_chunks: list[dict] = []
    all_clauses: list[str] = []

    if not user_claims:
        last_user = next(
            (m["content"] for m in reversed(state.get("messages", [])) if m.get("role") == "user"),
            "",
        )
        if last_user:
            queries = reformulate_queries("policy inquiry", last_user)
            all_chunks = _retrieve_deduped_chunks(queries)
        return {
            "user_claims": user_claims,
            "policy_context": [c["text"] for c in all_chunks],
            "policy_clauses": [],
        }

    for i, claim in enumerate(user_claims):
        # Skip claims that already have coverage assessed
        if claim.get("policy_coverage") is not None:
            continue

        component = claim.get("component", "")
        statement = claim.get("verbatim_statement", "") or component

        queries = reformulate_queries(component, statement)
        chunks = _retrieve_deduped_chunks(queries)
        all_chunks.extend(chunks)

        coverage = assess_coverage(component, statement, chunks)
        user_claims[i] = dict(claim)
        user_claims[i]["policy_coverage"] = coverage
        if coverage.get("covered") is False:
            user_claims[i]["claim_verdict"] = "rejected"

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
