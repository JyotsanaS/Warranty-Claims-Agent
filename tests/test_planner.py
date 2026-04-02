"""
Unit and integration tests for agent/nodes/planner.py

Layer 1 (hard rules) and Layer 2 (simple intent routes) are fully deterministic.
Layer 3 (LLM planner) uses real LLM calls to verify routing correctness.
_build_state_summary is tested as pure Python with no LLM.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.nodes.planner import (
    _build_state_summary,
    _has_final_verdicts,
    _has_policy_grounding,
    _has_unassessed_claims,
    _llm_planner,
    _VALID_LLM_NODES,
    planner_node,
)


# ── State factory ─────────────────────────────────────────────────────────────

def make_state(**overrides) -> dict:
    base: dict = {
        "session_id": "test-session",
        "trace_id": "test-trace",
        "turn_count": 1,
        "messages": [{"role": "user", "content": "hello"}],
        "empathy_prefix": "",
        "image_local_path": None,
        "policy_context": [],
        "policy_clauses": [],
        "damage_report": {},
        "claim_items": [],
        "active_claim_index": 0,
        "user_claims": [],
        "pending_claim_draft": None,
        "awaiting_post_resolution_followup": False,
        "intent": "greeting",
        "router_confidence": 0.9,
        "context_switch_detected": False,
        "pending_switch_confirmation": False,
        "awaiting_first_user_turn": False,
        "execution_plan": [],
        "next_node": "",
        "planner_reason": "",
        "awaiting_feedback": False,
        "user_feedback": None,
        "token_count": 0,
    }
    base.update(overrides)
    return base


def _covered_claim(**extra) -> dict:
    base = {
        "component": "charger",
        "assertion_type": "malfunction",
        "verbatim_statement": "charger stopped working",
        "requires_visual_validation": True,
        "policy_coverage": {"covered": True, "coverage_months": 24, "policy_clauses": ["clause-1"]},
        "visual_checks": [],
        "claim_verdict": None,
    }
    base.update(extra)
    return base


def _uncovered_claim(**extra) -> dict:
    base = {
        "component": "cable",
        "assertion_type": "cosmetic",
        "verbatim_statement": "cable has a scratch",
        "requires_visual_validation": False,
        "policy_coverage": {"covered": False, "exclusion_reason": "cosmetic damage excluded", "policy_clauses": ["§3.1"]},
        "visual_checks": [],
        "claim_verdict": "rejected",
    }
    base.update(extra)
    return base


def _unassessed_claim(**extra) -> dict:
    base = {
        "component": "adapter",
        "assertion_type": "malfunction",
        "verbatim_statement": "adapter overheats",
        "requires_visual_validation": True,
        "policy_coverage": None,
        "visual_checks": [],
        "claim_verdict": None,
    }
    base.update(extra)
    return base


# ── Helper function tests ─────────────────────────────────────────────────────

class TestHasUnassessedClaims:
    def test_empty_list(self):
        assert _has_unassessed_claims([]) is False

    def test_all_assessed(self):
        claims = [
            {"policy_coverage": {"covered": True}},
            {"policy_coverage": {"covered": False}},
        ]
        assert _has_unassessed_claims(claims) is False

    def test_one_unassessed(self):
        claims = [
            {"policy_coverage": {"covered": True}},
            {"policy_coverage": None},
        ]
        assert _has_unassessed_claims(claims) is True

    def test_missing_key(self):
        assert _has_unassessed_claims([{"component": "charger"}]) is True


class TestHasFinalVerdicts:
    def test_empty_list(self):
        assert _has_final_verdicts([]) is False

    def test_all_approved(self):
        assert _has_final_verdicts([{"claim_verdict": "approved"}, {"claim_verdict": "approved"}]) is True

    def test_all_rejected(self):
        assert _has_final_verdicts([{"claim_verdict": "rejected"}]) is True

    def test_all_escalated(self):
        assert _has_final_verdicts([{"claim_verdict": "escalated"}]) is True

    def test_mixed_final(self):
        assert _has_final_verdicts([{"claim_verdict": "approved"}, {"claim_verdict": "rejected"}]) is True

    def test_one_pending(self):
        assert _has_final_verdicts([{"claim_verdict": "approved"}, {"claim_verdict": "pending"}]) is False

    def test_verdict_none(self):
        assert _has_final_verdicts([{"claim_verdict": None}]) is False


class TestHasPolicyGrounding:
    def test_policy_clauses_present(self):
        assert _has_policy_grounding([], ["clause-1"], []) is True

    def test_policy_context_present(self):
        assert _has_policy_grounding([], [], ["some context"]) is True

    def test_grounding_from_claim_coverage(self):
        claims = [{"policy_coverage": {"policy_clauses": ["section-3"]}}]
        assert _has_policy_grounding(claims, [], []) is True

    def test_no_grounding(self):
        claims = [{"policy_coverage": {"covered": True, "policy_clauses": []}}]
        assert _has_policy_grounding(claims, [], []) is False

    def test_empty_everything(self):
        assert _has_policy_grounding([], [], []) is False

    def test_none_coverage_in_claim(self):
        assert _has_policy_grounding([{"policy_coverage": None}], [], []) is False


# ── _build_state_summary tests ────────────────────────────────────────────────

class TestBuildStateSummary:
    def test_empty_state(self):
        state = make_state(intent="claim")
        summary = _build_state_summary(state)
        assert "intent: claim" in summary
        assert "user_claims_total: 0" in summary
        assert "policy_chunks_available: 0" in summary
        assert "image_available: False" in summary

    def test_unassessed_claim_shown(self):
        state = make_state(intent="claim", user_claims=[_unassessed_claim()])
        summary = _build_state_summary(state)
        assert "user_claims_unassessed: 1" in summary
        assert "not assessed" in summary

    def test_rejected_claim_shown(self):
        state = make_state(intent="claim", user_claims=[_uncovered_claim()])
        summary = _build_state_summary(state)
        assert "verdict=rejected" in summary
        assert "covered=False" in summary
        assert "cosmetic damage excluded" in summary

    def test_policy_chunks_counted(self):
        state = make_state(policy_context=["chunk1", "chunk2", "chunk3"])
        summary = _build_state_summary(state)
        assert "policy_chunks_available: 3" in summary

    def test_image_flag(self):
        state = make_state(image_local_path="/tmp/img.jpg")
        summary = _build_state_summary(state)
        assert "image_available: True" in summary

    def test_visual_checks_counted(self):
        pending_check = {"check_description": "look for cracks", "finding": ""}
        claim = _covered_claim(visual_checks=[pending_check])
        state = make_state(user_claims=[claim])
        summary = _build_state_summary(state)
        assert "visual_checks=1" in summary
        assert "claims_with_pending_visual_checks: 1" in summary

    def test_covered_claims_awaiting_evidence_counted(self):
        state = make_state(user_claims=[_covered_claim(visual_checks=[], claim_verdict=None)])
        summary = _build_state_summary(state)
        assert "covered_claims_awaiting_evidence_plan: 1" in summary

    def test_covered_claim_with_verdict_not_counted_as_awaiting_evidence(self):
        state = make_state(user_claims=[_covered_claim(visual_checks=[], claim_verdict="approved")])
        summary = _build_state_summary(state)
        assert "covered_claims_awaiting_evidence_plan: 0" in summary

    def test_handles_missing_claim_fields_gracefully(self):
        state = make_state(user_claims=[{"component": "device"}])
        summary = _build_state_summary(state)
        assert "device" in summary
        assert "not assessed" in summary


# ── Layer 1: Hard rules ───────────────────────────────────────────────────────

class TestLayer1HardRules:
    def test_awaiting_feedback_routes_to_feedback_node(self):
        result = planner_node(make_state(awaiting_feedback=True))
        assert result["next_node"] == "feedback_node"

    def test_awaiting_feedback_beats_switch_confirmation(self):
        result = planner_node(make_state(awaiting_feedback=True, pending_switch_confirmation=True))
        assert result["next_node"] == "feedback_node"

    def test_awaiting_feedback_beats_post_resolution(self):
        result = planner_node(make_state(awaiting_feedback=True, awaiting_post_resolution_followup=True))
        assert result["next_node"] == "feedback_node"

    def test_pending_switch_confirmation_routes_to_handler(self):
        result = planner_node(make_state(pending_switch_confirmation=True))
        assert result["next_node"] == "confirmation_handler"

    def test_pending_switch_beats_post_resolution(self):
        result = planner_node(make_state(pending_switch_confirmation=True, awaiting_post_resolution_followup=True))
        assert result["next_node"] == "confirmation_handler"

    def test_awaiting_post_resolution_routes_to_close_node(self):
        result = planner_node(make_state(awaiting_post_resolution_followup=True))
        assert result["next_node"] == "post_resolution_close_node"

    def test_post_resolution_beats_simple_intent(self):
        result = planner_node(make_state(awaiting_post_resolution_followup=True, intent="greeting"))
        assert result["next_node"] == "post_resolution_close_node"

    def test_prompt_injection_routes_to_injection_node(self):
        result = planner_node(make_state(intent="prompt_injection"))
        assert result["next_node"] == "prompt_injection_node"

    def test_layer1_does_not_call_llm(self):
        with patch("agent.nodes.planner._llm_planner") as mock_llm:
            planner_node(make_state(awaiting_feedback=True))
            mock_llm.assert_not_called()


# ── Layer 2: Simple intent routes ─────────────────────────────────────────────

class TestLayer2SimpleIntents:
    @pytest.mark.parametrize("intent,expected", [
        ("greeting", "greeting_node"),
        ("out_of_scope", "fallback_node"),
        ("escalation", "escalation_node"),
        ("cancellation", "cancellation_node"),
        ("status_query", "status_node"),
    ])
    def test_simple_intent_routes(self, intent, expected):
        result = planner_node(make_state(intent=intent))
        assert result["next_node"] == expected

    def test_simple_intent_does_not_call_llm(self):
        with patch("agent.nodes.planner._llm_planner") as mock_llm:
            planner_node(make_state(intent="greeting"))
            mock_llm.assert_not_called()


# ── Layer 3: LLM planner — real API calls ─────────────────────────────────────

class TestLayer3LLMPlanner:
    """
    These tests make real LLM calls to verify the planner routes correctly
    for each claim-flow scenario. No mocking of the LLM.
    """

    def test_unassessed_claim_routes_to_policy_checker(self):
        state = make_state(
            intent="claim",
            messages=[{"role": "user", "content": "My charger stopped working suddenly."}],
            user_claims=[_unassessed_claim()],
        )
        node, reason = _llm_planner(state)
        assert node == "policy_checker", f"Expected policy_checker, got {node!r}. Reason: {reason}"

    def test_new_claim_message_routes_to_claim_state_updater(self):
        state = make_state(
            intent="claim",
            messages=[{"role": "user", "content": "I have a cracked screen on my device."}],
            user_claims=[],
        )
        node, reason = _llm_planner(state)
        assert node == "claim_state_updater", f"Expected claim_state_updater, got {node!r}. Reason: {reason}"

    def test_final_verdict_no_explanation_routes_to_agent_respond(self):
        """Verdict set but agent hasn't explained why yet → agent_respond first."""
        state = make_state(
            intent="clarification",
            messages=[
                {"role": "user", "content": "I have scratches on my device."},
            ],
            user_claims=[_uncovered_claim()],
            policy_context=["§3.1 Cosmetic damage is excluded from warranty coverage."],
        )
        node, reason = _llm_planner(state)
        assert node == "agent_respond", f"Expected agent_respond, got {node!r}. Reason: {reason}"

    def test_verdict_explained_in_conversation_routes_to_claim_decision(self):
        """Agent has already explained the rejection → claim_decision next."""
        state = make_state(
            intent="clarification",
            messages=[
                {"role": "user", "content": "I have scratches on my device."},
                {"role": "assistant", "content": "Unfortunately, cosmetic damage such as scratches is excluded from warranty coverage under §3.1. Since the damage does not affect device functionality, this claim is not covered."},
                {"role": "user", "content": "Okay I understand."},
            ],
            user_claims=[_uncovered_claim()],
            policy_context=["§3.1 Cosmetic damage is excluded from warranty coverage."],
        )
        node, reason = _llm_planner(state)
        assert node == "claim_decision", f"Expected claim_decision, got {node!r}. Reason: {reason}"

    def test_covered_claim_with_image_routes_to_evidence_planner(self):
        state = make_state(
            intent="evidence",
            messages=[{"role": "user", "content": "Here is the photo of the damaged charger."}],
            image_local_path="/tmp/damage.jpg",
            user_claims=[_covered_claim(visual_checks=[])],
        )
        node, reason = _llm_planner(state)
        assert node == "evidence_planner", f"Expected evidence_planner, got {node!r}. Reason: {reason}"

    def test_visual_checks_with_image_routes_to_vision_analysis(self):
        pending_check = {
            "check_description": "Look for burn marks on the cable",
            "expected_finding": "visible burn or scorch marks",
            "finding": "",
        }
        state = make_state(
            intent="evidence",
            messages=[{"role": "user", "content": "Here is the photo."}],
            image_local_path="/tmp/damage.jpg",
            user_claims=[_covered_claim(visual_checks=[pending_check])],
        )
        node, reason = _llm_planner(state)
        assert node == "vision_analysis", f"Expected vision_analysis, got {node!r}. Reason: {reason}"

    def test_frustration_with_no_claims_routes_to_agent_respond(self):
        state = make_state(
            intent="frustration",
            messages=[{"role": "user", "content": "This is completely unacceptable!"}],
            user_claims=[],
        )
        node, reason = _llm_planner(state)
        assert node == "agent_respond", f"Expected agent_respond, got {node!r}. Reason: {reason}"

    def test_output_is_always_a_valid_node(self):
        """LLM must always return a node in the allowed set."""
        state = make_state(
            intent="policy",
            messages=[{"role": "user", "content": "What does the warranty cover?"}],
        )
        node, reason = _llm_planner(state)
        assert node in _VALID_LLM_NODES, f"Got invalid node {node!r}"
        assert isinstance(reason, str) and reason


# ── LLM fallback on bad output ────────────────────────────────────────────────

class TestLLMFallback:
    """Error handling when the LLM returns unusable output. fast_llm is mocked
    because we need to simulate garbage responses that a real LLM won't produce."""

    def test_invalid_node_name_falls_back_to_agent_respond(self):
        with patch("agent.nodes.planner.fast_llm", return_value='{"next_node": "nonexistent_node", "reason": "bad"}'):
            node, reason = _llm_planner(make_state(intent="claim"))
            assert node == "agent_respond"
            assert "Fallback" in reason

    def test_malformed_json_falls_back_to_agent_respond(self):
        with patch("agent.nodes.planner.fast_llm", return_value="not valid json at all"):
            node, reason = _llm_planner(make_state(intent="claim"))
            assert node == "agent_respond"
            assert "Fallback" in reason

    def test_empty_response_falls_back_to_agent_respond(self):
        with patch("agent.nodes.planner.fast_llm", return_value=""):
            node, reason = _llm_planner(make_state(intent="claim"))
            assert node == "agent_respond"
            assert "Fallback" in reason


# ── Return shape ──────────────────────────────────────────────────────────────

class TestReturnShape:
    REQUIRED_KEYS = {"execution_plan", "next_node", "planner_reason"}

    @pytest.mark.parametrize("intent", ["greeting", "escalation", "out_of_scope", "prompt_injection"])
    def test_deterministic_intents_have_required_keys(self, intent):
        result = planner_node(make_state(intent=intent))
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_execution_plan_is_list(self):
        result = planner_node(make_state(intent="greeting"))
        assert isinstance(result["execution_plan"], list)

    def test_next_node_is_non_empty_string(self):
        result = planner_node(make_state(intent="greeting"))
        assert isinstance(result["next_node"], str)
        assert result["next_node"] != ""
