"""
RAG retriever: embeds a query with the local SentenceTransformer model
and queries Pinecone for the closest policy chunks.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Lazy singletons — initialised on first retrieval call
_embed_model = None
_pinecone_index = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        from app.config import settings
        logger.info("Loading embedding model %s …", settings.llm_embedding_model)
        _embed_model = SentenceTransformer(settings.llm_embedding_model)
    return _embed_model


def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        from pinecone import Pinecone
        from app.config import settings
        pc = Pinecone(api_key=settings.pinecone_api_key)
        _pinecone_index = pc.Index(settings.pinecone_index_name)
    return _pinecone_index


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """
    Embed *query* and return the top matching policy chunks from Pinecone.

    Each returned dict has keys: text, section, score.
    Returns an empty list if Pinecone is unavailable or no chunks meet the
    similarity threshold — callers should treat retrieval failure as non-fatal.
    """
    from app.config import settings

    if top_k is None:
        top_k = settings.rag_top_k

    try:
        model = _get_embed_model()
        index = _get_index()

        embedding: list[float] = model.encode(query).tolist()
        results = index.query(
            vector=embedding,
            top_k=top_k,
            namespace=settings.pinecone_namespace,
            include_metadata=True,
        )

        chunks = []
        for match in results.matches:
            if match.score >= settings.rag_similarity_threshold:
                chunks.append(
                    {
                        "text": match.metadata.get("text", ""),
                        "section": match.metadata.get("section_title", ""),
                        "score": round(match.score, 4),
                    }
                )
        logger.debug("RAG: %d chunks above threshold for query %r", len(chunks), query[:60])
        return chunks

    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return []
