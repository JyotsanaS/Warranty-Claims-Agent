# Architecture — VoltEdge Warranty Claims

## System Architecture

VoltEdge is implemented as a small, modular application with clear runtime boundaries between the API layer, agent runtime, model gateway, retrieval layer, and persistence layer.

### High-Level Components

| Component | Responsibility |
|---|---|
| **Streamlit UI** | Chat interface for session creation, message submission, image upload, and rendering streamed agent output. |
| **FastAPI app** | Public application boundary. Validates requests, manages sessions, accepts optional image uploads, and streams agent events back over SSE. |
| **LangGraph agent runtime** | Executes the claim workflow as a stateful graph over a typed `AgentState`. Owns orchestration, routing, and decision flow. |
| **LLM gateway** | Centralizes text, vision, and embedding model access in `gateway/llm_gateway.py` so provider-specific code stays out of business logic. |
| **RAG layer** | Retrieves grounded warranty-policy context from Pinecone using locally generated embeddings. |
| **Storage layer** | Persists uploaded images and per-session JSON artifacts under `RESULTS_DIR`; keeps active session state in memory for the current process. |
| **Observability** | Wraps API, LLM, RAG, and graph execution with OpenTelemetry spans and persists serialized traces with session artifacts. |

### Runtime Flow

1. The Streamlit client creates a session through FastAPI and sends each turn as text with an optional image.
2. FastAPI stores uploaded images on the local filesystem, appends the user message to the in-memory session record, and opens an SSE stream.
3. The LangGraph runner executes the agent graph for that turn and emits structured events such as node traces, tool activity, text deltas, and claim decisions.
4. Agent nodes call the LLM gateway for text or vision reasoning and call the RAG layer when policy grounding is required.
5. At the end of the turn, the backend persists the updated conversation, claim state, and trace artifacts under the session directory.

### Backend and Session Management

- The FastAPI service acts as the application boundary for the system. It owns request validation, multipart image intake, SSE response streaming, and session lifecycle endpoints.
- Sessions are explicitly created through `POST /api/v1/sessions` and keyed by `session_id`, which gives the agent a stable unit of state across turns.
- Active session state is maintained in memory and includes message history, claim status, structured claim state, active-claim pointer, and flow-control flags such as first-turn and post-resolution follow-up handling.
- The message endpoint is stateful at the application layer: it merges the current turn with prior session context, invokes the graph, then writes the updated agent state back into the session record after streaming completes.
- Idle session eviction is implemented as a background task. Sessions with no activity for the configured timeout are removed from the in-memory store to bound process memory and keep stale conversational state from accumulating.
- API rate limiting is intentionally simple in the current design: a per-session request cap of **15 turns per 5 minutes** on the message endpoint protects the expensive agent path without adding distributed infrastructure.
- In production, this would be extended into a two-layer control model: **edge rate limiting** for per-IP, per-user, and account-level quotas, and **backend throttling** to cap concurrent agent and vision workloads so burst traffic cannot exhaust the application tier.
- The current limiter is process-local and in-memory; a production deployment would move these controls to a shared store such as **Redis** or **Memcached** so limits remain consistent across multiple application instances.
- Each completed turn is persisted as a session artifact under `RESULTS_DIR/<session_id>/`, including conversation history, structured claim data, latest trace summary, and serialized OpenTelemetry trace output.
- Image uploads are validated before entering the agent flow and are stored under the per-session artifact directory so claim evidence stays tied to the corresponding conversation state.

### Architectural Notes

- The backend is the single integration boundary for the frontend; the UI does not call the agent, vector store, or model providers directly.
- The graph owns business flow, while transport concerns such as HTTP, multipart upload handling, and SSE remain in the API layer.
- Retrieval and model access are isolated behind dedicated modules, which keeps node logic focused on claim handling rather than infrastructure details.
- The current implementation is intentionally simple in its state model: active sessions are process-local, while durable artifacts are written to disk per session.

## Scale to >1M Users

At that scale, the core agent design can remain the same, but the runtime architecture must move from a single-process prototype model to a distributed service model with externalized state and explicit workload control.

### Required Changes

| Area | Production design direction |
|---|---|
| **API tier** | Run multiple stateless FastAPI instances behind a load balancer. Keep the API layer responsible for request validation, auth, and SSE/WebSocket delivery, but remove all process-local assumptions. |
| **Session state** | Move active session state out of in-memory Python dictionaries into a shared session store such as Redis or a database-backed state service so any app instance can resume a conversation safely. |
| **Rate limiting and throttling** | Enforce per-IP, per-user, and per-account quotas at the edge, backed by Redis or Memcached. Apply backend throttling for concurrent agent turns and vision-heavy workloads to protect the expensive path. |
| **Async claim execution** | For long-running turns, decouple claim execution from the request thread using a job queue and worker pool. The API tier becomes the ingress layer; agent workers handle graph execution. |
| **Artifact and image storage** | Replace local filesystem storage with object storage for uploaded images and session artifacts so files remain durable and accessible across instances. |
| **RAG infrastructure** | Keep retrieval as a separate service boundary around the vector index and embedding pipeline, with policy re-indexing handled asynchronously rather than inside the request path. |
| **Observability and operations** | Export traces, logs, and usage metrics to centralized observability systems; operational decisions at this scale require system-wide visibility rather than per-node local logs. |

### Design Principle

The main architectural shift is to make the **API tier stateless**, the **agent runtime horizontally scalable**, and all coordination concerns such as session state, quotas, artifacts, and background work owned by shared infrastructure instead of individual application processes.

### Prototype -> Production Migration

1. **Stateless API tier**: Replace process-local assumptions with multiple FastAPI instances behind a load balancer.
2. **Externalized session state**: Move active session and flow-control state from in-memory Python dictionaries to Redis or another shared session store.
3. **Persistent system of record**: Store conversations, structured claim state, verdicts, citations, feedback, and trace metadata in a database, while keeping large artifacts such as images in object storage.
4. **Edge rate limiting and backend throttling**: Enforce per-IP, per-user, and account-level quotas at the edge, and cap concurrent agent and vision workloads in the backend.
5. **Async execution model**: Move long-running or multimodal claim execution to queues and worker pools so request handling and agent execution can scale independently.
6. **Auditability and versioning**: Version policy corpus, prompts, model configuration, and decision logic so every claim outcome is reproducible and reviewable.
7. **Feedback and drift monitoring**: Capture user feedback and downstream outcomes, and continuously compare production behavior against offline benchmark suites to detect quality regressions.
8. **Graceful degradation**: When retrieval, vision, or model providers are degraded, fall back to pending or manual-review flows instead of failing the claim path outright.

## Agent Design

### Overview
A **LangGraph-based stateful agent** for processing EV charger warranty claims. Uses a `StateGraph` with a central planner orchestrating specialized nodes.

---

### Graph Topology

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

### Node Responsibilities

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

### State (`AgentState`)
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

### Router → Planner: Intent-Driven Routing

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

### Key Design Patterns
1. **Two-stage dispatch** — router classifies intent via LLM; planner translates intent + full state into an execution decision via pure logic (no LLM)
2. **Planner is the brain** — all routing decisions after the initial router live in `planner_node`
3. **Loop-back pattern** — most nodes return to planner after completing, enabling multi-step pipelines in one turn
4. **RAG → LLM pipeline** — `policy_checker` retrieves chunks then calls LLM; `evidence_planner` → `vision_analysis` → `claim_validator` is a 3-step image validation pipeline
5. **Observability** — every node is wrapped with `_instrument_node` for OpenTelemetry tracing; `runner.py` emits SSE events (`node_trace`, `tool_call`, `tool_result`, `text_delta`, `claim_decision`, `done`)
6. **Prompts in TOML** — all system prompts loaded from `agent/prompts.toml` via `get_prompt()`

---

### Guardrails

Framework: **[Guardrails AI](https://guardrailsai.com/)** — `GroundedAIHallucination` validator, backed by `fast_llm`.

#### Prompt Injection Detection

Prompt injection is handled as a separate router-level guardrail, especially on turns that can update claim state or influence policy reasoning. If the detector sees attempts to override policy, manipulate session state, or force an unsupported decision, the agent responds with a safe refusal such as: *"Sorry, we believe this request is attempting to manipulate policy or claim handling. We are closing this session for now."* The session is then terminated instead of continuing through the claim workflow.

#### Policy Hallucination Detection

LLM calls that reason over retrieved policy chunks can fabricate clause names or coverage rationale not present in the source text. A hallucinated approval is a direct business and legal risk.

The validator checks whether LLM output (`value`) is grounded in the retrieved policy chunks (`reference`) given the original question (`query`). On failure, **one retry** is attempted with the same context before falling back — giving the LLM a chance to self-correct before the fallback is invoked.

| Node | Why | Retry | Final fallback |
|---|---|---|---|
| `coverage_service.assess_coverage` | Source of all downstream verdict data. LLM outputs `policy_clauses` and `exclusion_reason` — if ungrounded and `covered=True`, every downstream node operates on corrupted data. Check only runs on `covered=True`; rejections are conservative by default. | Same chunks, stricter prompt: "only cite clauses present verbatim in the provided policy sections." | Downgrade: `covered=False`, `exclusion_reason="Coverage not grounded in retrieved policy. Manual review required."` |
| `claim_decision_node` | LLM generates customer-facing prose. Verdict is locked in deterministically by `claim_validator` — but the explanation can cite clauses never retrieved, creating a misleading audit trail. | Same context, explicit instruction to cite only from the `policy_clauses` list already in state. | Discard LLM prose. Replace with templated response built from pre-verified `policy_clauses` in state. Verdict is never changed. |

**Fail-open:** If the validator itself errors, the check is skipped and a warning is emitted to the trace. The claim flow is never blocked by a guardrail failure.

#### Image Quality Gate

Applied in `vision_service.analyze_image` before the multimodal LLM call. Checks pixel dimensions (PIL) and blurriness (Laplacian variance). On failure, returns `checks=[]` — `claim_validator` sees unevaluated checks, sets `verdict=pending`, and the agent asks the user to re-upload.
