# Design Decisions

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

## Chunking Strategy

We use a section-aware, structure-preserving chunking approach designed for policy documents where formatting conveys meaning.

- Split on section headers (H2+) to maintain semantic boundaries and avoid mixing multiple policy intents within a single chunk
- Treat tables as atomic units (≤~1600 chars) to prevent structural corruption, since partial tables significantly degrade embedding quality
- Apply recursive chunking within sections (~400–500 chars) to ensure each chunk represents a single coherent unit aligned with retrieval granularity
- Add ~10–20% overlap (~60–90 chars) to reduce boundary context loss without introducing excessive duplication or index bloat
- Store section context in metadata (not duplicated in text) to enable flexible retrieval strategies (e.g., context expansion, filtering) while keeping embeddings focused on content

These choices are optimized for BGE-M3, which performs best on mid-sized, structure-preserving inputs; chunk size is the primary lever (↑ for recall, ↓ for precision) before considering heavier models given current VRAM constraints.

This ensures chunks are semantically complete, retrieval-friendly, and aligned with document structure, which is critical for reliable RAG performance.
