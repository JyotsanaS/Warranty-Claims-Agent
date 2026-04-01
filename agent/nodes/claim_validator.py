"""
Claim validator node — deterministic mapping of VisualCheck results → claim_verdict.
No LLM call.
"""
from __future__ import annotations

from models.state import AgentState

_CONFIDENCE_THRESHOLD = 0.5


def claim_validator_node(state: AgentState) -> dict:
    user_claims: list[dict] = list(state.get("user_claims", []))

    for i, claim in enumerate(user_claims):
        # Skip already-decided claims
        if claim.get("claim_verdict") not in (None, "pending"):
            continue

        checks: list[dict] = claim.get("visual_checks", [])
        coverage = claim.get("policy_coverage") or {}

        # Not covered → rejected
        if not coverage.get("covered"):
            user_claims[i] = dict(claim)
            user_claims[i]["claim_verdict"] = "rejected"
            continue

        # No visual checks → pending (awaiting more evidence)
        if not checks:
            user_claims[i] = dict(claim)
            user_claims[i]["claim_verdict"] = "pending"
            continue

        # All checks must have been evaluated (non-empty finding)
        unevaluated = [c for c in checks if not c.get("finding")]
        if unevaluated:
            user_claims[i] = dict(claim)
            user_claims[i]["claim_verdict"] = "pending"
            continue

        # Verdict logic:
        # - approved: all checks above threshold AND majority support claim
        # - rejected: any check strongly contradicts (claim_supported=False, confidence ≥ threshold)
        # - pending: mixed results
        supported = [
            c for c in checks
            if c.get("claim_supported") and float(c.get("confidence", 0)) >= _CONFIDENCE_THRESHOLD
        ]
        contradicted = [
            c for c in checks
            if not c.get("claim_supported") and float(c.get("confidence", 0)) >= _CONFIDENCE_THRESHOLD
        ]

        if contradicted:
            verdict = "rejected"
        elif len(supported) == len(checks):
            verdict = "approved"
        elif supported:
            verdict = "pending"
        else:
            verdict = "pending"

        user_claims[i] = dict(claim)
        user_claims[i]["claim_verdict"] = verdict

    return {"user_claims": user_claims}
