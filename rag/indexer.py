from __future__ import annotations

import os
import time
import argparse
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from custom_chunking import PolicyChunk, create_policy_chunks

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "voltedge-warranty")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "warranty_policy_v1")
EMBEDDING_MODEL = os.environ.get("LLM_EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIMENSIONS = int(os.environ.get("LLM_EMBEDDING_DIMENSIONS", "1024"))

POLICY_FILE = Path(__file__).parent.parent / "data" / "sample-policy.md"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


def _get_or_create_index(pc: Pinecone) -> None:
    existing = {idx.name for idx in pc.list_indexes()}
    if PINECONE_INDEX_NAME in existing:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists — skipping creation.")
        return

    print(
        f"Creating serverless index '{PINECONE_INDEX_NAME}' "
        f"({EMBEDDING_DIMENSIONS}d, cosine, {PINECONE_CLOUD}/{PINECONE_REGION})..."
    )
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSIONS,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        print("  waiting for index to become ready...")
        time.sleep(2)
    print("Index ready.")


def _build_vectors(chunks: List[PolicyChunk], model: SentenceTransformer) -> list:
    texts = [chunk.text for chunk in chunks]
    print(f"Embedding {len(texts)} chunks with '{EMBEDDING_MODEL}'...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        metadata = {
            "policy_title": chunk.policy_title,
            "section_title": chunk.section_title,
            "chunk_index_in_section": chunk.chunk_index_in_section,
            "block_type": chunk.block_type,
            "text": chunk.text,
            "token_count": len(chunk.text.split()),
        }
        vectors.append({
            "id": chunk.chunk_id,
            "values": embedding.tolist(),
            "metadata": metadata,
        })
    return vectors


def _upsert_batched(index, vectors: list, batch_size: int = 100) -> None:
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)
        print(f"  upserted {min(i + batch_size, len(vectors))}/{len(vectors)} vectors")


def run_indexer(force: bool = False) -> None:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    _get_or_create_index(pc)
    index = pc.Index(PINECONE_INDEX_NAME)

    if not force:
        stats = index.describe_index_stats()
        ns = stats.namespaces.get(PINECONE_NAMESPACE)
        existing_count = ns.vector_count if ns else 0
        if existing_count > 0:
            print(
                f"Namespace '{PINECONE_NAMESPACE}' already has {existing_count} vectors. "
                "Pass --force to re-index."
            )
            return

    chunks = create_policy_chunks(
        file_path=str(POLICY_FILE),
        text_chunk_size=450,
        text_overlap=75,
        table_max_chars=1600,
    )
    print(f"Chunks produced by policy parser: {len(chunks)}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = _build_vectors(chunks, model)
    _upsert_batched(index, vectors)

    print(f"\nDone. {len(vectors)} vectors indexed into '{PINECONE_INDEX_NAME}' / '{PINECONE_NAMESPACE}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index VoltEdge policy into Pinecone.")
    parser.add_argument("--force", action="store_true", help="Re-index even if namespace is non-empty.")
    args = parser.parse_args()
    run_indexer(force=args.force)
