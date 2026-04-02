"""
Integration tests for policy retrieval using the live Pinecone setup.

These tests use the same Pinecone index/namespace configured in the repo's
`.env` and exercise the real retrieval path in `rag.retriever`.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
from dotenv import dotenv_values

from agent.nodes import policy_checker as mod
from agent.tools.retrieval_tool import retrieve_policy_chunks
from app.config import settings
from rag import retriever


REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_CSV = REPO_ROOT / "data" / "scenarios" / "voltedge_chatbot_policy_scenarios.csv"
ENV_FILE = REPO_ROOT / ".env"


def _load_scenarios() -> list[dict]:
    with SCENARIO_CSV.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


SCENARIOS = _load_scenarios()


def _query_plan_for_header(header: str) -> list[str]:
    plans = {
        "Logic board failure within warranty": [
            "main logic board replacement unit stopped turning on",
            "general limited warranty installed by certified technician warranty period",
            "claim validation requirements invoice serial number receipt",
        ],
        "Display dead pixels above threshold": [
            "led display touchscreen dead pixels more than 5",
            "general limited warranty device 8 months warranty period",
            "claim validation requirements serial number photo",
        ],
        "Cosmetic scratches on casing": [
            "external housing casing scratches faded cosmetic damage",
            "standard exclusions cosmetic damage scratches discoloration",
        ],
    }
    return plans[header]


def _section_aliases(raw: str) -> list[str]:
    aliases: list[str] = []
    for item in raw.split(";"):
        section = item.strip()
        if not section:
            continue
        aliases.append(section.lower())
        if section == "1. Normal Use Defined":
            aliases.append("1. general limited warranty")
        elif section == "2. Main Logic Board":
            aliases.append("main logic board")
        elif section == "2. LED Display / Touchscreen":
            aliases.append("led display / touchscreen")
            aliases.append("dead pixels")
        elif section == "2. External Housing/Casing":
            aliases.append("external housing/casing")
            aliases.append("cosmetic scratches/fading")
        elif section == "3. Cosmetic Damage":
            aliases.append("cosmetic damage")
        elif section == "4. Claim Validation Requirements":
            aliases.append("4. claim validation requirements")
            aliases.append("proof of purchase")
            aliases.append("serial number")
    return aliases


def _is_expected_chunk(chunk_text: str, raw_expected_sections: str) -> bool:
    lowered = chunk_text.lower()
    return any(alias in lowered for alias in _section_aliases(raw_expected_sections))


def _fake_assess_coverage(component: str, statement: str, chunks: list[dict]) -> dict:
    lowered = statement.lower()
    if "scratch" in lowered or "faded" in lowered:
        return {
            "covered": False,
            "exclusion_reason": "Cosmetic damage is excluded.",
            "policy_clauses": [chunk["section"] for chunk in chunks],
        }
    return {
        "covered": True,
        "coverage_months": 24,
        "policy_clauses": [chunk["section"] for chunk in chunks],
    }


@pytest.fixture
def live_pinecone_setup(monkeypatch):
    if not ENV_FILE.exists():
        pytest.skip(f"Missing env file: {ENV_FILE}")

    env = dotenv_values(ENV_FILE)
    required = [
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "PINECONE_NAMESPACE",
        "LLM_EMBEDDING_MODEL",
    ]
    missing = [name for name in required if not env.get(name)]
    if missing:
        pytest.skip(f"Missing Pinecone config in .env: {', '.join(missing)}")

    monkeypatch.setattr(settings, "pinecone_api_key", env["PINECONE_API_KEY"])
    monkeypatch.setattr(settings, "pinecone_index_name", env["PINECONE_INDEX_NAME"])
    monkeypatch.setattr(settings, "pinecone_namespace", env["PINECONE_NAMESPACE"])
    monkeypatch.setattr(settings, "llm_embedding_model", env["LLM_EMBEDDING_MODEL"])
    monkeypatch.setattr(settings, "rag_top_k", 5)
    monkeypatch.setattr(settings, "rag_similarity_threshold", 0.35)

    retriever._embed_model = None
    retriever._pinecone_index = None
    yield
    retriever._embed_model = None
    retriever._pinecone_index = None


class TestLivePineconeRetriever:
    @pytest.mark.integration
    @pytest.mark.parametrize("scenario", SCENARIOS, ids=[row["scenario_header"] for row in SCENARIOS])
    def test_retrieve_policy_chunks_hits_expected_sections(self, live_pinecone_setup, scenario):
        chunks = retrieve_policy_chunks(_query_plan_for_header(scenario["scenario_header"])[0], top_k=5)

        assert chunks, "live Pinecone retrieval should return chunks for the scenario query"
        assert any(
            _is_expected_chunk(chunk["text"], scenario["policy sections needed to check validity"])
            for chunk in chunks
        )

    @pytest.mark.integration
    def test_retrieve_deduped_chunks_returns_unique_top_five(self, live_pinecone_setup):
        queries = [
            "main logic board replacement unit stopped turning on",
            "general limited warranty installed by certified technician warranty period",
            "claim validation requirements invoice serial number receipt",
        ]

        chunks = mod._retrieve_deduped_chunks(queries)

        assert chunks
        assert len(chunks) <= 5
        assert len({chunk["text"] for chunk in chunks}) == len(chunks)
        assert chunks == sorted(chunks, key=lambda item: item["score"], reverse=True)


@pytest.mark.integration
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[row["scenario_header"] for row in SCENARIOS])
def test_policy_checker_uses_live_pinecone_context_for_scenarios(
    monkeypatch, live_pinecone_setup, scenario
):
    scenario_header = scenario["scenario_header"]
    query_plan = _query_plan_for_header(scenario_header)

    monkeypatch.setattr(mod, "reformulate_queries", lambda component, statement: list(query_plan))
    monkeypatch.setattr(mod, "assess_coverage", _fake_assess_coverage)

    state = {
        "messages": [{"role": "user", "content": scenario["user query"]}],
        "user_claims": [
            {
                "component": scenario_header,
                "verbatim_statement": scenario["user query"],
                "policy_coverage": None,
            }
        ],
    }

    result = mod.policy_checker_node(state)

    assert result["policy_context"], "policy checker should return live Pinecone context"
    assert any(
        _is_expected_chunk(chunk_text, scenario["policy sections needed to check validity"])
        for chunk_text in result["policy_context"]
    )

    if scenario["gt"] == "rejected":
        assert any("cosmetic damage" in chunk.lower() for chunk in result["policy_context"])
        assert result["user_claims"][0]["claim_verdict"] == "rejected"
        assert result["user_claims"][0]["policy_coverage"]["covered"] is False
    else:
        assert result["user_claims"][0]["policy_coverage"]["covered"] is True

    if scenario_header == "Logic board failure within warranty":
        assert any("main logic board" in chunk.lower() for chunk in result["policy_context"])
        assert any("serial number" in chunk.lower() for chunk in result["policy_context"])

    if scenario_header == "Display dead pixels above threshold":
        assert any("dead pixels" in chunk.lower() for chunk in result["policy_context"])
