"""
Deterministic RAG Context Evaluator

This evaluator measures retrieval quality against ground-truth policy sections
stored in the sample dataset.

Metrics:
  - context_recall: GT sections retrieved / total GT sections
  - context_precision: relevant retrieved chunks / total retrieved chunks

Usage:
    uv run python llm_as_judge/rag_context_eval.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from dotenv import dotenv_values

from agent.nodes.policy_checker import _retrieve_deduped_chunks
from app.config import settings
from rag import retriever


ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "data" / "scenarios" / "dataset.csv"
ENV_FILE = ROOT / ".env"


def load_scenarios() -> list[dict]:
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            if row.get("rag_gt_sections", "").strip():
                rows.append(row)
        return rows


def parse_query_plan(raw: str) -> list[str]:
    return [item.strip() for item in raw.split("||") if item.strip()]


def parse_gt_sections(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(";") if item.strip()]


def section_aliases(section: str) -> list[str]:
    aliases = [section.lower()]
    if section == "1. Normal Use Defined":
        aliases.append("1. general limited warranty")
        aliases.append("general limited warranty")
        aliases.append("warranty period")
    elif section == "2. Main Logic Board":
        aliases.append("main logic board")
    elif section == "2. LED Display / Touchscreen":
        aliases.append("led display / touchscreen")
        aliases.append("led display")
        aliases.append("touchscreen")
        aliases.append("dead pixels")
    elif section == "2. External Housing/Casing":
        aliases.append("external housing/casing")
        aliases.append("external housing")
        aliases.append("casing")
        aliases.append("cosmetic scratches/fading")
        aliases.append("scratches")
    elif section == "3. Cosmetic Damage":
        aliases.append("cosmetic damage")
        aliases.append("cosmetic damage exclusion")
    elif section == "4. Claim Validation Requirements":
        aliases.append("4. claim validation requirements")
        aliases.append("claim validation requirements")
        aliases.append("proof of purchase")
        aliases.append("serial number")
        aliases.append("invoice")
    return aliases


def chunk_matches_section(chunk: dict, section: str) -> bool:
    lowered_text = chunk.get("text", "").lower()
    lowered_section = chunk.get("section", "").lower()
    return any(
        alias in lowered_text or alias in lowered_section
        for alias in section_aliases(section)
    )


def configure_live_retrieval() -> None:
    if not ENV_FILE.exists():
        raise RuntimeError(f"Missing env file: {ENV_FILE}")

    env = dotenv_values(ENV_FILE)
    required = [
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "PINECONE_NAMESPACE",
        "LLM_EMBEDDING_MODEL",
    ]
    missing = [name for name in required if not env.get(name)]
    if missing:
        raise RuntimeError(f"Missing Pinecone config in .env: {', '.join(missing)}")

    settings.pinecone_api_key = env["PINECONE_API_KEY"]
    settings.pinecone_index_name = env["PINECONE_INDEX_NAME"]
    settings.pinecone_namespace = env["PINECONE_NAMESPACE"]
    settings.llm_embedding_model = env["LLM_EMBEDDING_MODEL"]
    settings.rag_top_k = 5
    settings.rag_similarity_threshold = 0.35

    retriever._embed_model = None
    retriever._pinecone_index = None


def evaluate_scenario(row: dict) -> dict:
    scenario = row["scenario"].strip()
    session_id = row["session_id"].strip()
    query_plan = parse_query_plan(row["rag_query_plan"])
    gt_sections = parse_gt_sections(row["rag_gt_sections"])

    chunks = _retrieve_deduped_chunks(query_plan, component=scenario)
    retrieved_sections = [chunk.get("section", "") for chunk in chunks]

    matched_sections = [
        section for section in gt_sections
        if any(chunk_matches_section(chunk, section) for chunk in chunks)
    ]
    relevant_chunk_count = sum(
        1 for chunk in chunks
        if any(chunk_matches_section(chunk, section) for section in gt_sections)
    )

    context_recall = len(matched_sections) / len(gt_sections) if gt_sections else 0.0
    context_precision = relevant_chunk_count / len(chunks) if chunks else 0.0

    return {
        "session_id": session_id,
        "scenario": scenario,
        "query_plan": query_plan,
        "gt_sections": gt_sections,
        "retrieved_sections": retrieved_sections,
        "retrieved_chunk_count": len(chunks),
        "relevant_chunk_count": relevant_chunk_count,
        "matched_gt_sections": matched_sections,
        "missed_gt_sections": [section for section in gt_sections if section not in matched_sections],
        "context_recall": round(context_recall, 4),
        "context_precision": round(context_precision, 4),
    }


def run() -> None:
    configure_live_retrieval()
    scenarios = load_scenarios()

    print(f"\n{'=' * 70}")
    print("  Deterministic RAG Context Evaluation")
    print(f"  Scenarios   : {len(scenarios)}")
    print("  Retrieval   : live Pinecone + existing policy checker dedupe path")
    print(f"{'=' * 70}\n")

    results: list[dict] = []
    for i, row in enumerate(scenarios, start=1):
        result = evaluate_scenario(row)
        results.append(result)

        print(f"[{i}/{len(scenarios)}] {result['scenario']}")
        print(f"      session_id         : {result['session_id']}")
        print(f"      matched_gt_sections: {len(result['matched_gt_sections'])}/{len(result['gt_sections'])}")
        print(f"      retrieved_chunks   : {result['retrieved_chunk_count']}")
        print(f"      relevant_chunks    : {result['relevant_chunk_count']}")
        print(f"      context_recall     : {result['context_recall']:.2f}")
        print(f"      context_precision  : {result['context_precision']:.2f}")
        if result["missed_gt_sections"]:
            print(f"      missed_sections    : {', '.join(result['missed_gt_sections'])}")
        print()

    avg_recall = sum(item["context_recall"] for item in results) / len(results) if results else 0.0
    avg_precision = sum(item["context_precision"] for item in results) / len(results) if results else 0.0

    print(f"{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Scenarios evaluated : {len(results)}")
    print(f"  Avg context_recall  : {avg_recall:.2f}")
    print(f"  Avg context_precision: {avg_precision:.2f}")
    print(f"{'=' * 70}\n")

    out_path = Path(__file__).parent / "rag_context_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "avg_context_recall": round(avg_recall, 4),
                "avg_context_precision": round(avg_precision, 4),
                "scenarios_evaluated": len(results),
                "scenarios": results,
            },
            f,
            indent=2,
        )
    print(f"  Results saved to: {out_path}\n")


if __name__ == "__main__":
    run()
