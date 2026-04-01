"""
Retrieval tool adapter for policy chunk lookup.
"""
from __future__ import annotations

from rag.retriever import retrieve


def retrieve_policy_chunks(query: str, top_k: int | None = None) -> list[dict]:
    """Return relevant policy chunks for a search query."""
    return retrieve(query, top_k=top_k)
