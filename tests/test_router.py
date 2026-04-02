"""
Tests for agent/nodes/router.py

Structure
---------
1. Pure unit tests — _build_recent_conversation, _build_structured_memory
   No LLM calls, no mocking required.

2. Short-circuit / fallback tests — router_node with mocked fast_llm
   Covers JSON parse failure, invalid intent, missing keys, and bypass paths.

3. Integration tests — router_node with real fast_llm
   Validates actual intent classification for all 11 intents.
   Requires a valid .env with GROQ_API_KEY.
"""
from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from agent.nodes.router import (
    _build_recent_conversation,
    _build_structured_memory,
    router_node,
)

_ALL_INTENTS = {
    "greeting", "out_of_scope", "escalation", "cancellation", "status_query",
    "frustration", "claim", "policy", "issue", "evidence", "clarification",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_state(**overrides) -> dict:
    base: dict = {
        "session_id": "test",
        "trace_id": "trace-1",
        "turn_count": 1,
        "messages": [{"role": "user", "content": "hello"}],
        "claim_items": [],
        "active_claim_index": 0,
        "user_claims": [],
        "pending_switch_confirmation": False,
        "awaiting_feedback": False,
        "intent": "",
        "router_confidence": 0.0,
    }
    base.update(overrides)
    return base


def _llm_response(intent: str, confidence: float = 0.9) -> str:
    return json.dumps({"intent": intent, "confidence": confidence})


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Pure unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildRecentConversation:
    def test_empty_messages(self):
        assert _build_recent_conversation([]) == "None"

    def test_system_messages_excluded(self):
        messages = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "hello"},
        ]
        result = _build_recent_conversation(messages)
        assert "system" not in result.lower()
        assert "USER: hello" in result

    def test_basic_turn(self):
        messages = [
            {"role": "user", "content": "my charger is broken"},
            {"role": "assistant", "content": "I can help with that."},
        ]
        result = _build_recent_conversation(messages)
        assert "USER: my charger is broken" in result
        assert "ASSISTANT: I can help with that." in result

    def test_max_turns_limits_history(self):
        # 10 user+assistant pairs, max_turns=4 → only last 8 messages
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"msg {i}"})
            messages.append({"role": "assistant", "content": f"reply {i}"})
        result = _build_recent_conversation(messages, max_turns=4)
        assert "msg 6" in result
        assert "msg 9" in result
        assert "msg 0" not in result   # too old

    def test_blank_content_excluded(self):
        messages = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "   "},
            {"role": "user", "content": "hello"},
        ]
        result = _build_recent_conversation(messages)
        lines = result.strip().split("\n")
        assert len(lines) == 1
        assert "hello" in lines[0]

    def test_returns_none_string_when_all_blank(self):
        messages = [{"role": "user", "content": ""}]
        assert _build_recent_conversation(messages) == "None"


class TestBuildStructuredMemory:
    def test_no_claims(self):
        state = make_state(claim_items=[], user_claims=[])
        result = _build_structured_memory(state)
        assert "Active claim: none" in result
        assert "Known user claims: none" in result

    def test_active_claim_present(self):
        state = make_state(
            claim_items=[{
                "component": "charger",
                "incident_type": "malfunction",
                "claim_status": "open",
                "has_image": True,
                "has_receipt": False,
            }],
            active_claim_index=0,
        )
        result = _build_structured_memory(state)
        assert "component=charger" in result
        assert "incident_type=malfunction" in result
        assert "has_image=True" in result
        assert "has_receipt=False" in result

    def test_active_claim_index_out_of_range(self):
        state = make_state(
            claim_items=[{"component": "charger", "claim_status": "open"}],
            active_claim_index=5,
        )
        result = _build_structured_memory(state)
        assert "Active claim: none" in result

    def test_pending_switch_confirmation_shown(self):
        state = make_state(pending_switch_confirmation=True)
        result = _build_structured_memory(state)
        assert "Pending context switch confirmation: True" in result

    def test_user_claims_listed(self):
        state = make_state(user_claims=[
            {
                "component": "cable",
                "assertion_type": "cosmetic",
                "claim_verdict": "rejected",
                "policy_coverage": {"covered": False},
            }
        ])
        result = _build_structured_memory(state)
        assert "component=cable" in result
        assert "verdict=rejected" in result
        assert "covered=False" in result

    def test_user_claim_unknown_coverage(self):
        # policy_coverage=None → covered should show "unknown"
        state = make_state(user_claims=[
            {"component": "adapter", "assertion_type": "malfunction", "policy_coverage": None}
        ])
        result = _build_structured_memory(state)
        assert "covered=unknown" in result

    def test_max_claims_capped_at_three(self):
        claims = [
            {"component": f"part_{i}", "assertion_type": "malfunction", "policy_coverage": None}
            for i in range(6)
        ]
        state = make_state(user_claims=claims)
        result = _build_structured_memory(state)
        assert "part_0" in result
        assert "part_2" in result
        assert "part_3" not in result   # exceeds max_claims=3


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Short-circuit and fallback tests (fast_llm mocked)
# ═══════════════════════════════════════════════════════════════════════════════

class TestShortCircuits:
    def test_awaiting_feedback_bypasses_llm(self):
        state = make_state(awaiting_feedback=True)
        with patch("agent.nodes.router.fast_llm") as mock_llm:
            result = router_node(state)
        mock_llm.assert_not_called()
        assert result["intent"] == "feedback_response"
        assert result["router_confidence"] == 1.0

    def test_pending_switch_confirmation_bypasses_llm(self):
        state = make_state(pending_switch_confirmation=True)
        with patch("agent.nodes.router.fast_llm") as mock_llm:
            result = router_node(state)
        mock_llm.assert_not_called()
        assert result["intent"] == "pending_switch"
        assert result["router_confidence"] == 1.0

    def test_awaiting_feedback_takes_priority_over_pending_switch(self):
        state = make_state(awaiting_feedback=True, pending_switch_confirmation=True)
        with patch("agent.nodes.router.fast_llm") as mock_llm:
            result = router_node(state)
        mock_llm.assert_not_called()
        assert result["intent"] == "feedback_response"


class TestFallbackBehavior:
    def test_json_parse_failure_defaults_to_issue(self):
        state = make_state(messages=[{"role": "user", "content": "something"}])
        with patch("agent.nodes.router.fast_llm", return_value="not json at all"):
            result = router_node(state)
        assert result["intent"] == "issue"
        assert result["router_confidence"] == 0.5

    def test_invalid_intent_defaults_to_issue(self):
        state = make_state(messages=[{"role": "user", "content": "something"}])
        with patch("agent.nodes.router.fast_llm", return_value=_llm_response("unknown_intent")):
            result = router_node(state)
        assert result["intent"] == "issue"

    def test_empty_llm_response_defaults_to_issue(self):
        state = make_state(messages=[{"role": "user", "content": "something"}])
        with patch("agent.nodes.router.fast_llm", return_value=""):
            result = router_node(state)
        assert result["intent"] == "issue"

    def test_missing_confidence_defaults_to_0_5(self):
        state = make_state(messages=[{"role": "user", "content": "something"}])
        with patch("agent.nodes.router.fast_llm", return_value=json.dumps({"intent": "claim"})):
            result = router_node(state)
        assert result["router_confidence"] == 0.5

    def test_valid_response_passes_through(self):
        state = make_state(messages=[{"role": "user", "content": "something"}])
        with patch("agent.nodes.router.fast_llm", return_value=_llm_response("claim", 0.95)):
            result = router_node(state)
        assert result["intent"] == "claim"
        assert result["router_confidence"] == pytest.approx(0.95)

    def test_return_always_has_required_keys(self):
        state = make_state(messages=[{"role": "user", "content": "hi"}])
        with patch("agent.nodes.router.fast_llm", return_value=_llm_response("greeting")):
            result = router_node(state)
        assert "intent" in result
        assert "router_confidence" in result

    @pytest.mark.parametrize("intent", sorted(_ALL_INTENTS))
    def test_all_valid_intents_pass_through(self, intent):
        state = make_state(messages=[{"role": "user", "content": "test"}])
        with patch("agent.nodes.router.fast_llm", return_value=_llm_response(intent)):
            result = router_node(state)
        assert result["intent"] == intent


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Integration tests — real LLM calls
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIntentClassification:
    """
    Calls the real fast_llm (Groq). Requires GROQ_API_KEY in .env.
    Run with: uv run pytest tests/test_router.py -m integration -v

    A small inter-test delay is applied to stay within Groq's free-tier
    TPM (tokens-per-minute) limit.
    """

    @pytest.fixture(autouse=True)
    def _rate_limit_guard(self):
        yield
        time.sleep(5)  # ~500 tokens/call × 12 calls/min ≈ 6000 TPM limit

    def _classify(self, user_message: str, extra_state: dict | None = None) -> str:
        state = make_state(messages=[{"role": "user", "content": user_message}])
        if extra_state:
            state.update(extra_state)
        result = router_node(state)
        return result["intent"]

    def test_greeting(self):
        assert self._classify("Hi there!") == "greeting"

    def test_greeting_formal(self):
        assert self._classify("Hello, I need some help.") == "greeting"

    def test_out_of_scope(self):
        assert self._classify("What is the capital of France?") == "out_of_scope"

    def test_out_of_scope_weather(self):
        assert self._classify("What's the weather like today?") == "out_of_scope"

    def test_escalation(self):
        assert self._classify("I want to speak to a human agent right now.") == "escalation"

    def test_escalation_explicit(self):
        assert self._classify("Connect me to a supervisor immediately.") == "escalation"

    def test_cancellation(self):
        assert self._classify("I want to cancel my warranty claim.") == "cancellation"

    def test_cancellation_phrased(self):
        assert self._classify("Please drop my claim, I don't want to proceed.") == "cancellation"

    def test_status_query(self):
        assert self._classify("What is the current status of my claim?") == "status_query"

    def test_status_query_phrased(self):
        assert self._classify("Can you tell me where things stand with my claim?") == "status_query"

    def test_frustration_sentiment_only(self):
        # Pure sentiment — no product defect described
        assert self._classify("I am really disappointed with your service.") == "frustration"

    def test_frustration_no_product_issue(self):
        assert self._classify("This is completely unacceptable. I'm very unhappy.") == "frustration"

    def test_claim_filing(self):
        assert self._classify("I want to file a warranty claim for my charger.") == "claim"

    def test_claim_submit(self):
        assert self._classify("I'd like to submit a warranty claim.") == "claim"

    def test_policy_question(self):
        assert self._classify("What does the warranty cover?") == "policy"

    def test_policy_coverage_question(self):
        assert self._classify("Is physical damage covered under the warranty?") == "policy"

    def test_issue_concrete_defect(self):
        assert self._classify("My charger won't turn on at all.") == "issue"

    def test_issue_physical_damage(self):
        assert self._classify("The display on the charging unit is cracked.") == "issue"

    def test_evidence_photo(self):
        assert self._classify("Here is a photo of the damaged charger.") == "evidence"

    def test_evidence_receipt(self):
        assert self._classify("I'm attaching my purchase receipt as proof.") == "evidence"

    def test_clarification_followup(self):
        # Answering a follow-up question asked by the agent
        messages = [
            {"role": "user", "content": "My charger stopped working."},
            {"role": "assistant", "content": "When did this happen?"},
            {"role": "user", "content": "It happened about two weeks ago."},
        ]
        state = make_state(messages=messages)
        result = router_node(state)
        assert result["intent"] == "clarification"

    def test_clarification_yes_no(self):
        messages = [
            {"role": "user", "content": "My charger is broken."},
            {"role": "assistant", "content": "Do you have the original receipt?"},
            {"role": "user", "content": "Yes, I do."},
        ]
        state = make_state(messages=messages)
        result = router_node(state)
        assert result["intent"] == "clarification"

    def test_confidence_is_between_0_and_1(self):
        state = make_state(messages=[{"role": "user", "content": "My charger broke."}])
        result = router_node(state)
        assert 0.0 <= result["router_confidence"] <= 1.0

    def test_classified_intent_is_valid(self):
        state = make_state(messages=[{"role": "user", "content": "Can I file a claim?"}])
        result = router_node(state)
        assert result["intent"] in _ALL_INTENTS
