"""
Tests for the hallucination guardrail, coverage_service.assess_coverage (guarded path),
and claim_decision_node (guarded path).

All external I/O (LLM calls, Guardrails AI) is mocked — no network calls are made.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

class _FakeValidationError(Exception):
    """Stand-in for guardrails.errors.ValidationError."""


class _FakeFailedReaskError(Exception):
    """Stand-in for guardrails FailedReaskError (name contains 'FailedReask')."""
    pass
_FakeFailedReaskError.__name__ = "FailedReaskError"


def _inject_guardrails(guard_validate_side_effect=None):
    """
    Injects fake guardrails modules into sys.modules so that
    `from guardrails import Guard` etc. inside is_grounded work
    without the real package being installed.

    Returns a context manager that cleans up on exit.
    """
    fake_guard_instance = MagicMock()
    fake_guard_instance.use.return_value = fake_guard_instance
    if guard_validate_side_effect is not None:
        fake_guard_instance.validate.side_effect = guard_validate_side_effect

    fake_guardrails_mod = ModuleType("guardrails")
    fake_guardrails_mod.Guard = MagicMock(return_value=fake_guard_instance)  # type: ignore[attr-defined]

    fake_hub_mod = ModuleType("guardrails.hub")
    fake_hub_mod.GroundedAIHallucination = MagicMock()  # type: ignore[attr-defined]

    fake_errors_mod = ModuleType("guardrails.errors")
    fake_errors_mod.ValidationError = _FakeValidationError  # type: ignore[attr-defined]

    mods = {
        "guardrails": fake_guardrails_mod,
        "guardrails.hub": fake_hub_mod,
        "guardrails.errors": fake_errors_mod,
    }

    @contextmanager
    def _ctx():
        orig = {k: sys.modules.get(k) for k in mods}
        sys.modules.update(mods)
        try:
            yield fake_guard_instance
        finally:
            for k, v in orig.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    return _ctx()


@contextmanager
def _fake_span_ctx():
    """Minimal mock for start_span used in coverage_service."""
    span = MagicMock()

    @contextmanager
    def _start_span(name, attrs=None):
        yield span

    yield _start_span, span


# ═══════════════════════════════════════════════════════════════════════════════
# is_grounded
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsGrounded:
    def test_empty_value_skips_check(self):
        """Empty value should return True without calling the validator."""
        from agent.guardrails.hallucination import is_grounded
        assert is_grounded("", "some policy text", "query") is True

    def test_empty_reference_skips_check(self):
        """Empty reference should return True without calling the validator."""
        from agent.guardrails.hallucination import is_grounded
        assert is_grounded("some output", "", "query") is True

    def test_grounded_output_returns_true(self):
        """validate() succeeds (no exception) → grounded → returns True."""
        with _inject_guardrails(guard_validate_side_effect=None) as guard:
            from agent.guardrails import hallucination as mod
            result = mod.is_grounded("correct clause text", "policy: correct clause text", "q")
        assert result is True
        guard.validate.assert_called_once()

    def test_validation_error_returns_false(self):
        """validate() raises ValidationError → hallucination detected → returns False."""
        with _inject_guardrails(guard_validate_side_effect=_FakeValidationError("hallucination")) as guard:
            from agent.guardrails import hallucination as mod
            result = mod.is_grounded("fabricated clause", "policy: unrelated text", "q")
        assert result is False

    def test_failed_reask_error_returns_false(self):
        """validate() raises an error whose class name contains 'FailedReask' → returns False."""
        with _inject_guardrails(guard_validate_side_effect=_FakeFailedReaskError("reask")) as guard:
            from agent.guardrails import hallucination as mod
            result = mod.is_grounded("some text", "some reference", "q")
        assert result is False

    def test_unexpected_error_fails_open(self):
        """Any unexpected validator error must fail open (returns True) — never block the flow."""
        with _inject_guardrails(guard_validate_side_effect=RuntimeError("network down")) as guard:
            from agent.guardrails import hallucination as mod
            result = mod.is_grounded("text", "reference", "query")
        assert result is True

    def test_guardrails_import_error_fails_open(self):
        """If guardrails is not installed at all, is_grounded must fail open."""
        # Remove any injected guardrails so the import fails naturally
        orig = {k: sys.modules.pop(k, None) for k in ["guardrails", "guardrails.hub", "guardrails.errors"]}
        try:
            from agent.guardrails import hallucination as mod
            result = mod.is_grounded("text", "reference", "query")
        finally:
            for k, v in orig.items():
                if v is not None:
                    sys.modules[k] = v
        assert result is True


# ═══════════════════════════════════════════════════════════════════════════════
# _build_templated_decision
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildTemplatedDecision:
    def _fn(self):
        from agent.nodes.claim_decision import _build_templated_decision
        return _build_templated_decision

    def test_approved_claim_with_clauses(self):
        fn = self._fn()
        user_claims = [
            {
                "component": "charger",
                "claim_verdict": "approved",
                "policy_coverage": {"policy_clauses": ["Section 3.1", "Section 3.2"]},
            }
        ]
        text = fn(user_claims, ["Section 3.1", "Section 3.2"])
        assert "charger" in text
        assert "APPROVED" in text
        assert "Section 3.1" in text

    def test_rejected_claim_shows_exclusion_reason(self):
        fn = self._fn()
        user_claims = [
            {
                "component": "cable",
                "claim_verdict": "rejected",
                "policy_coverage": {
                    "policy_clauses": [],
                    "exclusion_reason": "Physical damage not covered",
                },
            }
        ]
        text = fn(user_claims, [])
        assert "cable" in text
        assert "REJECTED" in text
        assert "Physical damage not covered" in text

    def test_multiple_claims(self):
        fn = self._fn()
        user_claims = [
            {"component": "charger", "claim_verdict": "approved", "policy_coverage": {"policy_clauses": ["S1"]}},
            {"component": "cable", "claim_verdict": "rejected", "policy_coverage": {"policy_clauses": []}},
        ]
        text = fn(user_claims, ["S1"])
        assert "charger" in text
        assert "cable" in text
        assert "APPROVED" in text
        assert "REJECTED" in text


# ═══════════════════════════════════════════════════════════════════════════════
# assess_coverage — hallucination guard paths
# ═══════════════════════════════════════════════════════════════════════════════

_COVERED_JSON = json.dumps({
    "covered": True,
    "policy_clauses": ["Section 4"],
    "coverage_months": 24,
})
_NOT_COVERED_JSON = json.dumps({
    "covered": False,
    "exclusion_reason": "Out of warranty",
    "policy_clauses": [],
})
_CHUNKS = [{"text": "Section 4: charger is covered for 24 months", "score": 0.9}]


def _patch_coverage_deps(fast_llm_returns, is_grounded_side_effect):
    """
    Stacks all patches needed to unit-test assess_coverage in isolation.
    fast_llm_returns: list of return values for successive fast_llm calls.
    is_grounded_side_effect: list of booleans returned by is_grounded.
    """
    return (
        patch("agent.services.coverage_service.fast_llm", side_effect=fast_llm_returns),
        patch("agent.services.coverage_service.get_prompt", return_value="system prompt"),
        patch("agent.services.coverage_service.start_span", side_effect=lambda name, attrs=None: _mock_span_cm()),
        patch("agent.guardrails.hallucination.is_grounded", side_effect=is_grounded_side_effect),
    )


@contextmanager
def _mock_span_cm():
    yield MagicMock()


class TestAssessCoverageGuardrail:
    def _run(self, fast_llm_returns, is_grounded_returns):
        p_llm, p_prompt, p_span, p_grounded = _patch_coverage_deps(
            fast_llm_returns, is_grounded_returns
        )
        with p_llm, p_prompt, p_span, p_grounded:
            from agent.services.coverage_service import assess_coverage
            return assess_coverage("charger", "stopped working", _CHUNKS)

    def test_grounded_approval_passes_through(self):
        """covered=True and grounded → result returned unchanged."""
        result = self._run(
            fast_llm_returns=[_COVERED_JSON],
            is_grounded_returns=[True],
        )
        assert result["covered"] is True
        assert result["policy_clauses"] == ["Section 4"]

    def test_hallucination_triggers_retry(self):
        """First check fails → retry is called with stricter prompt → retry grounded → use retry result."""
        result = self._run(
            fast_llm_returns=[_COVERED_JSON, _COVERED_JSON],  # initial + retry
            is_grounded_returns=[False, True],                 # initial fails, retry passes
        )
        assert result["covered"] is True

    def test_hallucination_retry_also_fails_applies_fallback(self):
        """Both checks fail → fallback: covered=False with manual review reason."""
        result = self._run(
            fast_llm_returns=[_COVERED_JSON, _COVERED_JSON],
            is_grounded_returns=[False, False],
        )
        assert result["covered"] is False
        assert "Manual review" in (result.get("exclusion_reason") or "")

    def test_rejection_skips_hallucination_check(self):
        """covered=False → guardrail is never invoked."""
        with (
            patch("agent.services.coverage_service.fast_llm", return_value=_NOT_COVERED_JSON),
            patch("agent.services.coverage_service.get_prompt", return_value=""),
            patch("agent.services.coverage_service.start_span", side_effect=lambda name, attrs=None: _mock_span_cm()),
            patch("agent.guardrails.hallucination.is_grounded") as mock_grounded,
        ):
            from agent.services.coverage_service import assess_coverage
            result = assess_coverage("charger", "dropped", _CHUNKS)

        mock_grounded.assert_not_called()
        assert result["covered"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# claim_decision_node — hallucination guard paths
# ═══════════════════════════════════════════════════════════════════════════════

_CLAIM = {
    "component": "charger",
    "claim_verdict": "approved",
    "policy_coverage": {"policy_clauses": ["Section 4"], "covered": True},
}

_STATE_BASE: dict = {
    "session_id": "abc12345",
    "user_claims": [_CLAIM],
    "policy_clauses": ["Section 4"],
    "policy_context": ["Section 4: charger is covered for 24 months"],
    "claim_items": [],
}


class TestClaimDecisionGuardrail:
    def _run(self, main_llm_returns, is_grounded_returns, state_override=None):
        state = dict(_STATE_BASE, **(state_override or {}))
        with (
            patch("agent.nodes.claim_decision.main_llm", side_effect=main_llm_returns),
            patch("agent.nodes.claim_decision.get_prompt", return_value="system"),
            patch("agent.guardrails.hallucination.is_grounded", side_effect=is_grounded_returns),
        ):
            from agent.nodes.claim_decision import claim_decision_node
            return claim_decision_node(state)

    def test_no_policy_context_skips_guardrail(self):
        """No policy_context → reference is empty → guardrail is never called."""
        with (
            patch("agent.nodes.claim_decision.main_llm", return_value="Your claim is approved."),
            patch("agent.nodes.claim_decision.get_prompt", return_value="system"),
            patch("agent.guardrails.hallucination.is_grounded") as mock_grounded,
        ):
            from agent.nodes.claim_decision import claim_decision_node
            claim_decision_node(dict(_STATE_BASE, policy_context=[]))
        mock_grounded.assert_not_called()

    def test_grounded_text_used_as_is(self):
        """LLM output is grounded → text returned unchanged (no retry)."""
        result = self._run(
            main_llm_returns=["Approved under Section 4."],
            is_grounded_returns=[True],
        )
        messages = result["messages"]
        assert any("Approved under Section 4" in m["content"] for m in messages)

    def test_hallucination_triggers_retry_and_passes(self):
        """First check fails → retry called → retry grounded → retry text used."""
        result = self._run(
            main_llm_returns=["Fabricated clause text.", "Grounded retry text."],
            is_grounded_returns=[False, True],
        )
        messages = result["messages"]
        assert any("Grounded retry text" in m["content"] for m in messages)

    def test_hallucination_persists_uses_templated_fallback(self):
        """Both checks fail → templated fallback used — verdict (approved) must be preserved."""
        result = self._run(
            main_llm_returns=["Bad text.", "Still bad text."],
            is_grounded_returns=[False, False],
        )
        messages = result["messages"]
        combined = " ".join(m["content"] for m in messages)
        # Templated decision includes the component and verdict
        assert "charger" in combined.lower()
        assert "APPROVED" in combined

    def test_verdict_never_changed_by_guardrail(self):
        """Even when the guardrail triggers a fallback, the user_claims verdicts are untouched."""
        result = self._run(
            main_llm_returns=["Bad text.", "Still bad."],
            is_grounded_returns=[False, False],
        )
        for claim in result["user_claims"]:
            assert claim.get("claim_verdict") == "approved"
