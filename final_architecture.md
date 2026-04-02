# Agent Architecture — VoltEdge Warranty Claims

## Overview
A **LangGraph-based stateful agent** for processing EV charger warranty claims. Uses a `StateGraph` with a central planner orchestrating specialized nodes.

---

## Graph Topology

```
START → router → planner → [conditional routing]
                              ├── Simple terminal nodes → END
                              │   (greeting, fallback, escalation, cancellation,
                              │    status, post_resolution_close, feedback,
                              │    confirmation_handler)
                              ├── claim_state_updater → planner (loop back)
                              ├── empathy_node → planner (loop back)
                              ├── policy_checker → planner (loop back)
                              ├── evidence_planner → planner (loop back)
                              ├── vision_analysis → claim_validator → planner (loop back)
                              ├── agent_respond → END
                              └── claim_decision → END
```

---

## Node Responsibilities

| Node | Role |
|---|---|
| **router** | LLM classifies user intent into 11 categories (greeting, claim, evidence, escalation, etc.). Uses `fast_llm`. Bypasses LLM if `pending_switch_confirmation` or `awaiting_feedback`. |
| **planner** | Pure logic (no LLM). Reads state to decide `next_node` and `execution_plan`. Acts as the central orchestrator/dispatcher. |
| **claim_state_updater** | Updates claim record — LLM extracts structured claim context (`ClaimContext`) and normalizes `UserClaim` objects. Handles multi-claim sessions and context switches. |
| **policy_checker** | RAG retrieval from policy doc → LLM assesses coverage per claim. Sets `claim_verdict = "rejected"` if not covered. |
| **evidence_planner** | LLM plans `VisualCheck` objects for covered claims — specifying what to look for in an uploaded image. |
| **vision_analysis** | Multimodal LLM analyzes uploaded image against all planned `VisualCheck`s. |
| **claim_validator** | **Deterministic** (no LLM). Maps visual check results → `claim_verdict` (approved/rejected/pending) using confidence threshold (0.5). |
| **agent_respond** | Generates reply — conversational response with policy + claim context injected. |
| **claim_decision** | Policy-grounded verdict — issues formal final decision with case reference and policy clause citations. |
| **empathy_node** | De-escalates frustrated users by prepending an empathy prefix before the main response. |
| **Simple terminal nodes** | LLM-generated canned responses for greeting/escalation/status/cancellation. Triggered directly by intent. |
| **confirmation_handler / feedback_node / post_resolution_close** | Second-tier terminal nodes triggered by flow-control flags (`pending_switch_confirmation`, `awaiting_feedback`, post-resolution state) rather than intent. |

---

## State (`AgentState`)
Key fields:
- `messages` — full conversation history
- `claim_items` — `ClaimContext` list (structured session data)
- `user_claims` — `UserClaim` list (policy-assessable claims)
- `intent`, `router_confidence` — set by router
- `next_node`, `execution_plan`, `planner_reason` — set by planner
- `policy_context`, `policy_clauses` — RAG results
- `damage_report` — vision analysis output
- `image_local_path` — uploaded image path
- `awaiting_feedback`, `pending_switch_confirmation`, etc. — flow control flags

---

## Router → Planner: Intent-Driven Routing

The agent uses a two-stage dispatch model where **router identifies intent** and **planner acts on it**.

The router makes a single `fast_llm` call to classify the user message into one of 11 intents, writing `intent` and `router_confidence` into state. It always hands off to the planner — it never routes directly to a terminal node.

The planner then uses `intent` as its primary signal, combined with broader state context (existing claims, image presence, policy coverage status), to decide the actual next node:

| Intent | Planner decision |
|---|---|
| `greeting` | `greeting_node` |
| `out_of_scope` | `fallback_node` |
| `escalation` | `escalation_node` |
| `cancellation` | `cancellation_node` |
| `status_query` | `status_node` |
| `claim / issue / evidence / clarification / policy` | `claim_state_updater` → (loops back) → `policy_checker` → `evidence_planner` → `vision_analysis` → `claim_decision` |
| `frustration` | `claim_state_updater` if unassessed claims exist, else `agent_respond` |

This separation is intentional: the router answers **"what does the user want?"** while the planner answers **"what should the agent do next?"** — allowing the planner to factor in session state beyond just the current message intent.

---

## Key Design Patterns
1. **Two-stage dispatch** — router classifies intent via LLM; planner translates intent + full state into an execution decision via pure logic (no LLM)
2. **Planner is the brain** — all routing decisions after the initial router live in `planner_node` (pure logic, no LLM)
3. **Loop-back pattern** — most nodes return to planner after completing, enabling multi-step pipelines in one turn
4. **RAG → LLM pipeline** — `policy_checker` retrieves chunks then calls LLM; `evidence_planner` → `vision_analysis` → `claim_validator` is a 3-step image validation pipeline
5. **Observability** — every node is wrapped with `_instrument_node` for OpenTelemetry tracing; `runner.py` emits SSE events (`node_trace`, `tool_call`, `tool_result`, `text_delta`, `claim_decision`, `done`)
6. **Prompts in TOML** — all system prompts loaded from `agent/prompts.toml` via `get_prompt()`

---

## Guardrails

Framework: **[Guardrails AI](https://guardrailsai.com/)** — `GroundedAIHallucination` validator, backed by `fast_llm`.

### Policy Hallucination Detection

LLM calls that reason over retrieved policy chunks can fabricate clause names or coverage rationale not present in the source text. A hallucinated approval is a direct business and legal risk.

The validator checks whether LLM output (`value`) is grounded in the retrieved policy chunks (`reference`) given the original question (`query`). On failure, **one retry** is attempted with the same context before falling back — giving the LLM a chance to self-correct before the fallback is invoked.

| Node | Why | Retry | Final fallback |
|---|---|---|---|
| `coverage_service.assess_coverage` | Source of all downstream verdict data. LLM outputs `policy_clauses` and `exclusion_reason` — if ungrounded and `covered=True`, every downstream node operates on corrupted data. Check only runs on `covered=True`; rejections are conservative by default. | Same chunks, stricter prompt: "only cite clauses present verbatim in the provided policy sections." | Downgrade: `covered=False`, `exclusion_reason="Coverage not grounded in retrieved policy. Manual review required."` |
| `claim_decision_node` | LLM generates customer-facing prose. Verdict is locked in deterministically by `claim_validator` — but the explanation can cite clauses never retrieved, creating a misleading audit trail. | Same context, explicit instruction to cite only from the `policy_clauses` list already in state. | Discard LLM prose. Replace with templated response built from pre-verified `policy_clauses` in state. Verdict is never changed. |

**Fail-open:** If the validator itself errors, the check is skipped and a warning is emitted to the trace. The claim flow is never blocked by a guardrail failure.

### Image Quality Gate

Applied in `vision_service.analyze_image` before the multimodal LLM call. Checks pixel dimensions (PIL) and blurriness (Laplacian variance). On failure, returns `checks=[]` — `claim_validator` sees unevaluated checks, sets `verdict=pending`, and the agent asks the user to re-upload.
