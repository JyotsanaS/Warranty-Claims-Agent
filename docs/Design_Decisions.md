# Design Decisions

## Chunking Strategy

We adopted a section-aware, structure-preserving chunking approach tailored for policy documents. Instead of naïve fixed-size splitting, the pipeline:

- Splits documents by section headings (H2+), ensuring semantic boundaries are respected
- Preserves tables as atomic units, avoiding row-level fragmentation
- Applies recursive chunking within each section to maintain coherence
- Introduces controlled overlap (≈10–20%) between chunks to reduce context loss
- Avoids repeating section titles in the first chunk, while maintaining section context through metadata

This design ensures that each chunk is contextually meaningful, retrieval-friendly, and structurally aligned with the original document.

## Embedding Model Selection

For retrieval, we selected BGE-M3 as the primary embedding model.

**Rationale:**

- Strong performance on MTEB benchmarks for retrieval tasks
- Well-suited for structured and semi-structured text (e.g., sections, lists, tables)
- Supports longer context windows (~8k tokens) while performing optimally on mid-sized chunks
- Provides a good balance between performance, latency, and resource requirements

We did not choose larger models such as Qwen3-Embedding due to current VRAM constraints. However, this remains a future option if higher recall or semantic fidelity is required.

## Future Considerations

We will evaluate retrieval performance based on:

- Context recall across sections
- Handling of edge cases (tables, lists, cross-references)

If current performance is insufficient, we may:

- Explore higher-capacity embedding models
- Introduce hybrid retrieval (dense + keyword)
- Revisit chunk size or overlap parameters
