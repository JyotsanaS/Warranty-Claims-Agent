"""
Unit tests for agent/nodes/planner.py

planner_node is fully deterministic (no LLM calls), so no mocking is required.
Every branch in the decision tree is covered.
"""
from __future__ import annotations

import pytest

from agent.nodes.planner import (
    _has_final_verdicts,
    _has_policy_grounding,
    _has_unassessed_claims,
    planner_node,
)


# ── State factory ─────────────────────────────────────────────────────────────

def make_state(**overrides) -> dict:
    """Return a minimal valid AgentState dict with sensible defaults."""
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
        "claim_state_updated_this_turn": False,
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
        "policy_coverage": {"covered": False, "exclusion_reason": "cosmetic damage excluded"},
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
        # No policy_coverage key at all — treated same as None
        assert _has_unassessed_claims([{"component": "charger"}]) is True


class TestHasFinalVerdicts:
    def test_empty_list(self):
        assert _has_final_verdicts([]) is False

    def test_all_approved(self):
        claims = [{"claim_verdict": "approved"}, {"claim_verdict": "approved"}]
        assert _has_final_verdicts(claims) is True

    def test_all_rejected(self):
        assert _has_final_verdicts([{"claim_verdict": "rejected"}]) is True

    def test_all_escalated(self):
        assert _has_final_verdicts([{"claim_verdict": "escalated"}]) is True

    def test_mixed_final(self):
        claims = [{"claim_verdict": "approved"}, {"claim_verdict": "rejected"}]
        assert _has_final_verdicts(claims) is True

    def test_one_pending(self):
        claims = [{"claim_verdict": "approved"}, {"claim_verdict": "pending"}]
        assert _has_final_verdicts(claims) is False

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
        claims = [{"policy_coverage": None}]
        assert _has_policy_grounding(claims, [], []) is False


# ── planner_node branch tests ─────────────────────────────────────────────────

class TestAwaiting:
    """Priority-1 and priority-2 short circuits."""

    def test_awaiting_feedback_routes_to_feedback_node(self):
        state = make_state(awaiting_feedback=True)
        result = planner_node(state)
        assert result["next_node"] == "feedback_node"

    def test_awaiting_feedback_takes_priority_over_switch_confirmation(self):
        state = make_state(awaiting_feedback=True, pending_switch_confirmation=True)
        result = planner_node(state)
        assert result["next_node"] == "feedback_node"

    def test_pending_switch_confirmation_routes_to_confirmation_handler(self):
        state = make_state(pending_switch_confirmation=True)
        result = planner_node(state)
        assert result["next_node"] == "confirmation_handler"

    def test_pending_switch_confirmation_takes_priority_over_claim_state_intents(self):
        state = make_state(
            pending_switch_confirmation=True,
            intent="claim",
            claim_state_updated_this_turn=False,
        )
        result = planner_node(state)
        assert result["next_node"] == "confirmation_handler"


class TestClaimStateUpdater:
    """Intent-driven routing to claim_state_updater."""

    @pytest.mark.parametrize("intent", ["claim", "issue", "evidence", "clarification", "policy"])
    def test_claim_intents_route_to_updater(self, intent):
        state = make_state(intent=intent, claim_state_updated_this_turn=False)
        result = planner_node(state)
        assert result["next_node"] == "claim_state_updater"

    def test_frustration_with_text_routes_to_updater(self):
        state = make_state(
            intent="frustration",
            messages=[{"role": "user", "content": "this charger is terrible"}],
            claim_state_updated_this_turn=False,
        )
        result = planner_node(state)
        assert result["next_node"] == "claim_state_updater"

    def test_frustration_with_image_and_no_text_routes_to_updater(self):
        state = make_state(
            intent="frustration",
            messages=[{"role": "user", "content": ""}],
            image_local_path="/tmp/img.jpg",
            claim_state_updated_this_turn=False,
            user_claims=[],   # no claim memory → not image_only_followup
            claim_items=[],
        )
        result = planner_node(state)
        assert result["next_node"] == "claim_state_updater"

    def test_already_updated_this_turn_skips_updater(self):
        # claim intent but claim_state_updated_this_turn=True → falls through
        state = make_state(
            intent="claim",
            claim_state_updated_this_turn=True,
            user_claims=[_unassessed_claim()],
        )
        result = planner_node(state)
        assert result["next_node"] == "policy_checker"

    def test_image_only_followup_skips_updater(self):
        """Image uploaded with no text and existing claim memory → skip to policy/evidence."""
        claim = _covered_claim(visual_checks=[], claim_verdict=None)
        state = make_state(
            intent="claim",
            claim_state_updated_this_turn=False,
            messages=[{"role": "user", "content": ""}],  # no text
            image_local_path="/tmp/damage.jpg",
            user_claims=[claim],
            claim_items=[{"component": "charger"}],
        )
        result = planner_node(state)
        # Should NOT go to claim_state_updater; covered + no checks + image → evidence_planner
        assert result["next_node"] == "evidence_planner"

    def test_non_claim_intent_skips_updater(self):
        state = make_state(intent="greeting", claim_state_updated_this_turn=False)
        result = planner_node(state)
        assert result["next_node"] == "greeting_node"


class TestPostResolutionFollowup:
    """Awaiting post-resolution followup branches."""

    def test_negative_reply_routes_to_close_node(self):
        state = make_state(
            awaiting_post_resolution_followup=True,
            intent="greeting",   # ensure not caught earlier
            messages=[{"role": "user", "content": "no"}],
            claim_state_updated_this_turn=True,  # skip updater branch
        )
        result = planner_node(state)
        assert result["next_node"] == "post_resolution_close_node"

    @pytest.mark.parametrize("text", ["no", "nope", "nah", "nothing", "no thanks", "not now", "nothing else"])
    def test_all_negative_markers_route_to_close(self, text):
        state = make_state(
            awaiting_post_resolution_followup=True,
            messages=[{"role": "user", "content": text}],
            claim_state_updated_this_turn=True,
        )
        result = planner_node(state)
        assert result["next_node"] == "post_resolution_close_node"

    def test_non_negative_reply_also_routes_to_close_node(self):
        # Regardless of reply content, post-resolution always closes with feedback prompt.
        state = make_state(
            awaiting_post_resolution_followup=True,
            messages=[{"role": "user", "content": "yes please help me"}],
            claim_state_updated_this_turn=True,
            intent="greeting",
        )
        result = planner_node(state)
        assert result["next_node"] == "post_resolution_close_node"


class TestSimpleIntentRoutes:
    """Direct terminal intent routing."""

    @pytest.mark.parametrize("intent,expected_node", [
        ("prompt_injection", "prompt_injection_node"),
        ("greeting",     "greeting_node"),
        ("out_of_scope", "fallback_node"),
        ("escalation",   "escalation_node"),
        ("cancellation", "cancellation_node"),
        ("status_query", "status_node"),
    ])
    def test_simple_intent_routes(self, intent, expected_node):
        state = make_state(
            intent=intent,
            claim_state_updated_this_turn=True,  # skip updater
        )
        result = planner_node(state)
        assert result["next_node"] == expected_node


class TestFrustrationWithoutPendingWork:
    def test_frustration_no_unassessed_claims_routes_to_agent_respond(self):
        # All claims are assessed → frustration goes directly to agent_respond
        state = make_state(
            intent="frustration",
            claim_state_updated_this_turn=True,
            user_claims=[_covered_claim(claim_verdict="rejected")],
        )
        result = planner_node(state)
        assert result["next_node"] == "agent_respond"


class TestPolicyChecker:
    def test_unassessed_claim_routes_to_policy_checker(self):
        state = make_state(
            intent="claim",
            claim_state_updated_this_turn=True,
            user_claims=[_unassessed_claim()],
        )
        result = planner_node(state)
        assert result["next_node"] == "policy_checker"

    def test_mixed_assessed_unassessed_routes_to_policy_checker(self):
        state = make_state(
            intent="claim",
            claim_state_updated_this_turn=True,
            user_claims=[_covered_claim(claim_verdict="approved"), _unassessed_claim()],
        )
        result = planner_node(state)
        assert result["next_node"] == "policy_checker"


class TestEvidencePlanner:
    def test_covered_claim_no_checks_with_image_routes_to_evidence_planner(self):
        state = make_state(
            intent="claim",
            claim_state_updated_this_turn=True,
            image_local_path="/tmp/img.jpg",
            user_claims=[_covered_claim(visual_checks=[])],
        )
        result = planner_node(state)
        assert result["next_node"] == "evidence_planner"

    def test_no_image_does_not_route_to_evidence_planner(self):
        state = make_state(
            intent="claim",
            claim_state_updated_this_turn=True,
            image_local_path=None,
            user_claims=[_covered_claim(visual_checks=[])],
        )
        result = planner_node(state)
        assert result["next_node"] != "evidence_planner"


class TestVisionAnalysis:
    def test_covered_claim_with_pending_checks_and_image_routes_to_vision(self):
        pending_check = {
            "check_description": "look for burn marks",
            "expected_finding": "visible burn marks",
            "finding": "",   # unevaluated
        }
        state = make_state(
            intent="claim",
            claim_state_updated_this_turn=True,
            image_local_path="/tmp/damage.jpg",
            user_claims=[_covered_claim(visual_checks=[pending_check])],
        )
        result = planner_node(state)
        assert result["next_node"] == "vision_analysis"

    def test_completed_checks_do_not_route_to_vision(self):
        completed_check = {
            "check_description": "look for burn marks",
            "expected_finding": "visible burn marks",
            "finding": "burn marks found",
        }
        state = make_state(
            intent="claim",
            claim_state_updated_this_turn=True,
            image_local_path="/tmp/damage.jpg",
            user_claims=[_covered_claim(visual_checks=[completed_check], claim_verdict="approved")],
            policy_clauses=["clause-1"],
        )
        result = planner_node(state)
        assert result["next_node"] == "claim_decision"


class TestClaimDecision:
    def test_final_verdicts_with_grounding_routes_to_claim_decision(self):
        state = make_state(
            intent="claim",
            claim_state_updated_this_turn=True,
            user_claims=[_covered_claim(claim_verdict="approved")],
            policy_clauses=["section-4.2"],
        )
        result = planner_node(state)
        assert result["next_node"] == "claim_decision"

    def test_final_verdicts_grounded_via_policy_context(self):
        state = make_state(
            intent="claim",
            claim_state_updated_this_turn=True,
            user_claims=[_uncovered_claim()],
            policy_context=["The warranty excludes cosmetic damage."],
            policy_clauses=[],
        )
        result = planner_node(state)
        assert result["next_node"] == "claim_decision"

    def test_final_verdicts_grounded_via_claim_clauses(self):
        covered = _covered_claim(
            claim_verdict="approved",
            policy_coverage={"covered": True, "policy_clauses": ["clause-7"]},
        )
        state = make_state(
            intent="claim",
            claim_state_updated_this_turn=True,
            user_claims=[covered],
            policy_clauses=[],
            policy_context=[],
        )
        result = planner_node(state)
        assert result["next_node"] == "claim_decision"

    def test_final_verdicts_without_grounding_routes_to_agent_respond(self):
        state = make_state(
            intent="claim",
            claim_state_updated_this_turn=True,
            user_claims=[_covered_claim(
                claim_verdict="approved",
                policy_coverage={"covered": True, "policy_clauses": []},
            )],
            policy_clauses=[],
            policy_context=[],
        )
        result = planner_node(state)
        assert result["next_node"] == "agent_respond"
        assert "no retrieved policy grounding" in result["planner_reason"]


class TestAgentRespond:
    def test_no_claims_routes_to_agent_respond(self):
        state = make_state(
            intent="claim",
            claim_state_updated_this_turn=True,
            user_claims=[],
        )
        result = planner_node(state)
        assert result["next_node"] == "agent_respond"

    def test_all_assessed_no_final_verdicts_routes_to_agent_respond(self):
        # covered but claim_verdict is still None (pending validation)
        state = make_state(
            intent="claim",
            claim_state_updated_this_turn=True,
            user_claims=[_covered_claim(claim_verdict=None, visual_checks=[])],
            image_local_path=None,
        )
        result = planner_node(state)
        assert result["next_node"] == "agent_respond"
        assert "More information" in result["planner_reason"]


class TestReturnShape:
    """Ensure planner_node always returns the required keys."""

    REQUIRED_KEYS = {"execution_plan", "next_node", "planner_reason", "claim_state_updated_this_turn"}

    @pytest.mark.parametrize("intent", [
        "greeting", "claim", "escalation", "frustration", "out_of_scope", "prompt_injection",
    ])
    def test_return_contains_required_keys(self, intent):
        state = make_state(intent=intent)
        result = planner_node(state)
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_execution_plan_is_list(self):
        result = planner_node(make_state())
        assert isinstance(result["execution_plan"], list)

    def test_next_node_is_string(self):
        result = planner_node(make_state())
        assert isinstance(result["next_node"], str)
        assert result["next_node"] != ""
