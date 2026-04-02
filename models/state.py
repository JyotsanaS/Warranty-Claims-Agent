"""
Typed data structures for the VoltEdge LangGraph agent.
All structures passed through the graph are defined here.
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ── Per-check structure for visual validation ─────────────────────────────────

class VisualCheck(BaseModel):
    check_description: str
    expected_finding: str
    finding: str = ""
    observation: str = ""
    claim_supported: bool = False
    confidence: float = 0.0


# ── Policy coverage outcome ───────────────────────────────────────────────────

class PolicyCoverage(BaseModel):
    covered: bool = False
    coverage_months: Optional[int] = None
    exclusion_reason: Optional[str] = None
    policy_clauses: list[str] = Field(default_factory=list)


# ── One extracted claim assertion per conversation turn ───────────────────────

class UserClaim(BaseModel):
    component: str
    assertion_type: str = "damage"
    verbatim_statement: str = ""
    requires_visual_validation: bool = False
    policy_coverage: Optional[dict] = None   # PolicyCoverage serialised as dict
    visual_checks: list[dict] = Field(default_factory=list)   # VisualCheck dicts
    claim_verdict: Optional[str] = None      # approved | rejected | pending | escalated


# ── Working memory per claim item ─────────────────────────────────────────────

class ClaimContext(BaseModel):
    component: str = ""
    incident_type: str = ""
    incident_date: Optional[str] = None
    purchase_date: Optional[str] = None
    has_receipt: bool = False
    has_image: bool = False
    claim_status: str = "open"               # open | approved | rejected | escalated | cancelled
    warranty_eligible: Optional[bool] = None
    audit_notes: list[str] = Field(default_factory=list)


# ── Router output ─────────────────────────────────────────────────────────────

class RouterOutput(BaseModel):
    intent: str
    confidence: float
    requires_policy_lookup: bool = False
    requires_vision: bool = False


# ── Agent graph state (LangGraph TypedDict) ───────────────────────────────────

class AgentState(TypedDict):
    # Identity / accounting
    session_id: str
    trace_id: str
    turn_count: int

    # Conversation history — append-only via operator.add reducer
    # Each element: {"role": "user"|"assistant"|"system", "content": str}
    messages: Annotated[list[dict], add]

    # Response assembly
    empathy_prefix: str

    # Image from this turn
    image_local_path: Optional[str]

    # RAG output
    policy_context: list[str]
    policy_clauses: list[str]

    # Vision output
    damage_report: dict

    # Claim tracking
    claim_items: list[dict]        # list of ClaimContext serialised as dicts
    active_claim_index: int
    user_claims: list[dict]        # list of UserClaim serialised as dicts
    pending_claim_draft: Optional[dict]
    awaiting_post_resolution_followup: bool
    claim_state_updated_this_turn: bool

    # Router signals
    intent: str
    router_confidence: float
    context_switch_detected: bool
    pending_switch_confirmation: bool
    awaiting_first_user_turn: bool

    # Planner output
    execution_plan: list[str]
    next_node: str
    planner_reason: str

    # Feedback collection
    awaiting_feedback: bool
    user_feedback: Optional[str]     # "like" | "dislike" | None

    # Session control
    terminate_session: bool

    # Token budget
    token_count: int
