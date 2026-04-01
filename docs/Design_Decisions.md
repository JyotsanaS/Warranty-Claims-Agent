# Design Decisions

## Chunking Strategy

We use a section-aware, structure-preserving chunking approach designed for policy documents where formatting conveys meaning.

- Split on section headers (H2+) to maintain semantic boundaries
- Treat tables as atomic units to prevent structural corruption
- Apply recursive chunking within sections for coherent, size-bounded chunks
- Add ~10–20% overlap to reduce boundary context loss
- Store section context in metadata (not duplicated in text) This is useful if we need to retrieve more context later on or do some way of filtering.

This ensures chunks are semantically complete, retrieval-friendly, and aligned with document structure, which is critical for reliable RAG performance.

## Embedding Model Selection

For retrieval, we selected BGE-M3 as embedding model.

Why:

- Strong MTEB retrieval performance
- Handles structured + semi-structured content well
- Supports long context (~8k tokens) while remaining efficient at our chunk sizes
- Good balance of quality, latency, and infra cost

We avoided larger models (e.g., Qwen3 8B) due to VRAM constraints. We will revisit if recall proves insufficient.

### Future Considerations

We will evaluate retrieval performance based on:

- Context recall across sections
- Handling of edge cases (tables, lists, cross-references)

If current performance is insufficient, we may:

- Explore higher-capacity embedding models
- Introduce hybrid retrieval (dense + keyword)
- Revisit chunk size or overlap parameters
