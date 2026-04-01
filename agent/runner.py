"""
Graph runner — bridges the LangGraph agent to the SSE event format expected by the API.

Emits events in order:
  tool_call      — when RAG / vision nodes start
  tool_result    — when they finish
  text_delta     — one per word of the assistant response
  claim_decision — final verdict (when all claims resolved)
  done           — stream end
  error          — on unrecoverable failure
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from models.state import AgentState

logger = logging.getLogger(__name__)

# Nodes that are considered "tool" calls for the UI
_TOOL_NODES = {
    "policy_checker": "rag_retrieval",
    "evidence_planner": "evidence_planner",
    "vision_analysis": "vision_analysis",
}

# Nodes that produce user-visible text responses
_RESPONSE_NODES = {
    "greeting_node",
    "fallback_node",
    "status_node",
    "escalation_node",
    "cancellation_node",
    "confirmation_handler",
    "agent_respond",
    "claim_decision",
}

# session_id → final state dict; populated after each graph run
_final_states: dict[str, dict] = {}


def pop_final_state(session_id: str) -> dict:
    """Return and remove the stored final state for a session."""
    return _final_states.pop(session_id, {})


def _sse(event: str, data: dict) -> str:
    return f"data: {json.dumps({'event': event, 'data': data})}\n\n"


def _words(text: str):
    """Yield tokens suitable for streaming (word + trailing space)."""
    words = text.split()
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")


def stream_agent_response(
    session_id: str,
    messages: list[dict],
    image_path: str | None = None,
    claim_items: list[dict] | None = None,
    user_claims: list[dict] | None = None,
    active_claim_index: int = 0,
):
    """
    Generator yielding SSE-formatted strings.

    Parameters
    ----------
    session_id         : current session identifier
    messages           : full conversation history as {role, content} dicts
    image_path         : absolute path to uploaded image, or None
    claim_items        : persisted ClaimContext list from previous turns
    user_claims        : persisted UserClaim list from previous turns
    active_claim_index : index of the currently active claim
    """
    from agent.graph import get_compiled_graph

    graph = get_compiled_graph()

    initial_state: AgentState = {
        "session_id": session_id,
        "trace_id": str(uuid.uuid4()),
        "turn_count": 1,
        "messages": list(messages),
        "empathy_prefix": "",
        "image_local_path": image_path,
        "policy_context": [],
        "policy_clauses": [],
        "damage_report": {},
        "claim_items": list(claim_items or []),
        "active_claim_index": active_claim_index,
        "user_claims": list(user_claims or []),
        "intent": "",
        "router_confidence": 0.0,
        "context_switch_detected": False,
        "pending_switch_confirmation": False,
        "token_count": 0,
    }

    # ── Langfuse trace setup ──────────────────────────────────────────────────
    from gateway.langfuse_client import get_langfuse, set_trace_context
    lf = get_langfuse()
    lf_trace = None
    if lf is not None:
        try:
            from app.config import settings as _cfg
            lf_trace = lf.trace(
                id=initial_state["trace_id"],
                name=f"{_cfg.langfuse_project_name}.agent_turn",
                session_id=session_id,
                tags=[_cfg.langfuse_project_name],
                metadata={
                    "image_attached": image_path is not None,
                    "prior_claims": len(initial_state.get("user_claims", [])),
                },
            )
            set_trace_context(initial_state["trace_id"])
        except Exception:
            logger.warning("Failed to create Langfuse trace", exc_info=True)

    claim_decision_data: dict | None = None
    final_state: dict = {}
    trace: list[dict] = []
    stream_start = time.perf_counter()
    node_start = stream_start

    # ── Langfuse trace setup (v3+ API) ────────────────────────────────────────
    import contextlib
    from gateway.langfuse_client import get_langfuse, set_trace_context
    lf = get_langfuse()

    _lf_ctx = contextlib.nullcontext(None)
    if lf is not None:
        try:
            from app.config import settings as _cfg
            _lf_ctx = lf.start_as_current_observation(
                as_type="trace",
                name=f"{_cfg.langfuse_project_name}.agent_turn",
                session_id=session_id,
                tags=[_cfg.langfuse_project_name],
                metadata={
                    "image_attached": image_path is not None,
                    "prior_claims": len(initial_state.get("user_claims", [])),
                },
            )
        except Exception:
            logger.warning("Failed to create Langfuse trace context", exc_info=True)

    with _lf_ctx as lf_trace:
        if lf_trace is not None:
            try:
                set_trace_context(lf_trace.trace_id)
            except Exception:
                pass

        try:
            for step in graph.stream(initial_state, stream_mode="updates"):
            for node_name, updates in step.items():
                logger.debug("Node completed: %s", node_name)

                now = time.perf_counter()
                duration_ms = round((now - node_start) * 1000)
                elapsed_ms = round((now - stream_start) * 1000)
                node_start = now

                # ── Build per-node trace metadata ────────────────────────────
                step_info: dict = {
                    "node": node_name,
                    "duration_ms": duration_ms,
                    "elapsed_ms": elapsed_ms,
                }
                if node_name == "router":
                    step_info["intent"] = updates.get("intent", "")
                    step_info["confidence"] = round(updates.get("router_confidence", 0.0), 2)
                elif node_name == "claim_extractor":
                    step_info["claims_found"] = len(updates.get("user_claims", []))
                elif node_name == "policy_checker":
                    step_info["chunks_retrieved"] = len(updates.get("policy_context", []))
                    step_info["clauses"] = updates.get("policy_clauses", [])
                elif node_name == "empathy_node":
                    step_info["prefix"] = updates.get("empathy_prefix", "")
                elif node_name == "vision_analysis":
                    report = updates.get("damage_report", {})
                    step_info["image_quality"] = report.get("image_quality", "")
                    step_info["confidence"] = round(report.get("overall_confidence", 0.0), 2)
                elif node_name == "claim_decision":
                    claims = updates.get("user_claims", [])
                    step_info["verdicts"] = {
                        c.get("component", "?"): c.get("claim_verdict") for c in claims
                    }
                trace.append(step_info)
                yield _sse("node_trace", step_info)

                # ── Langfuse node span ───────────────────────────────────────
                if lf_trace is not None:
                    try:
                        node_end_time = time.time()
                        node_start_time = node_end_time - duration_ms / 1000
                        import datetime
                        span = lf_trace.span(
                            name=node_name,
                            start_time=datetime.datetime.fromtimestamp(node_start_time, tz=datetime.timezone.utc),
                            end_time=datetime.datetime.fromtimestamp(node_end_time, tz=datetime.timezone.utc),
                            metadata={k: v for k, v in step_info.items() if k != "node"},
                        )
                        span.end()
                        set_trace_context(initial_state["trace_id"], span.id)
                    except Exception:
                        logger.debug("Failed to create Langfuse span for %s", node_name, exc_info=True)

                # ── Tool-call events ─────────────────────────────────────────
                if node_name in _TOOL_NODES:
                    tool_label = _TOOL_NODES[node_name]
                    yield _sse("tool_call", {"tool": tool_label, "status": "running"})

                    result_detail: dict = {"tool": tool_label}
                    if node_name == "policy_checker":
                        result_detail["chunks_found"] = len(updates.get("policy_context", []))
                    elif node_name == "vision_analysis":
                        result_detail["result"] = updates.get("damage_report", {})
                    yield _sse("tool_result", result_detail)

                # ── Text-delta events ────────────────────────────────────────
                if node_name in _RESPONSE_NODES:
                    for msg in updates.get("messages", []):
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            for token in _words(msg.get("content", "")):
                                yield _sse("text_delta", {"token": token})

                # ── Collect claim decision data (emitted after text) ──────────
                if node_name == "claim_decision":
                    updated_claims = updates.get("user_claims", [])
                    verdicts = [c.get("claim_verdict") for c in updated_claims]
                    if all(v == "approved" for v in verdicts):
                        status = "approved"
                    elif any(v == "rejected" for v in verdicts):
                        status = "rejected"
                    elif any(v == "escalated" for v in verdicts):
                        status = "escalated"
                    else:
                        status = "pending"

                    clauses: list[str] = []
                    for c in updated_claims:
                        clauses.extend((c.get("policy_coverage") or {}).get("policy_clauses", []))
                    claim_decision_data = {
                        "status": status,
                        "reason": "Final decision based on policy review and evidence assessment.",
                        "cited_clauses": list(dict.fromkeys(clauses)),
                    }

                final_state.update(updates)

        if claim_decision_data:
            yield _sse("claim_decision", claim_decision_data)

        total_ms = round((time.perf_counter() - stream_start) * 1000)
        yield _sse("done", {"trace": trace, "total_ms": total_ms})

    except Exception as exc:
        logger.exception("Agent graph error in session %s: %s", session_id, exc)
        if lf_trace is not None:
            try:
                lf_trace.update(level="ERROR", status_message=str(exc))
            except Exception:
                pass
        yield _sse("error", {"message": str(exc)})

    finally:
        if lf is not None:
            try:
                lf.flush()
            except Exception:
                logger.debug("Langfuse flush failed", exc_info=True)

    _final_states[session_id] = final_state
