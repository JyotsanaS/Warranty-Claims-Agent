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

---

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

---

### Architectural Notes

- The backend is the single integration boundary for the frontend; the UI does not call the agent, vector store, or model providers directly.
- The graph owns business flow, while transport concerns such as HTTP, multipart upload handling, and SSE remain in the API layer.
- Retrieval and model access are isolated behind dedicated modules, which keeps node logic focused on claim handling rather than infrastructure details.
- The current implementation is intentionally simple in its state model: active sessions are process-local, while durable artifacts are written to disk per session.

## Scale to >1M Users

### Capacity Framing

Before designing for scale, the load needs to be concrete.

```
1M registered users
× 20% DAU                     = 200,000 sessions/day
× 7 turns/session             = 1,400,000 agent invocations/day

Spread over 10 active hours:
1.4M / (10 × 3,600)           ≈ 39 turns/second steady state
2–3× peak factor              ≈ 80–120 turns/second at peak

Average session active time   ≈ 10 minutes (7 turns + think time)
Peak concurrent sessions      ≈ 3,300
```

Each turn triggers an average of 3.5 LLM calls (router, claim reasoning, response generation), with vision turns reaching 5. That puts the steady-state LLM call rate at **~137 calls/second**, peaking at **~410 calls/second** during bursts. These numbers set the sizing target for every architectural decision below.

---

### Concurrency

The current implementation is single-process and in-memory. At this load, the agent runtime must be horizontally scalable with no process-local state assumptions.

**API tier:** Run multiple stateless FastAPI instances behind a load balancer. The API layer handles request validation, image intake, and SSE delivery. It does not execute the agent graph.

**Agent workers:** Decouple agent execution from the request thread. The API tier enqueues a job per turn; a pool of agent workers picks up jobs and streams results back. Workers scale independently of the API tier based on queue depth and LLM throughput.

**Session state:** Move active session state — message history, claim status, flow-control flags — out of in-memory Python dictionaries into Redis. Any worker can resume any session safely. Redis also handles SSE result routing back to the originating API instance holding the open connection.

**Storage:** Replace local filesystem storage with object storage for uploaded images and session artifacts. Per-session JSON artifacts move to a database-backed store so claim state is durable across restarts and accessible for audit.

**SSE at scale:** SSE connections are long-lived. The load balancer must support sticky routing or SSE responses must be relayed through Redis pub/sub to reach the correct API instance holding the open connection.

---

### Cost

For an LLM system, cost is the primary scaling constraint — not raw concurrency. At 1.4M turns/day, token spend compounds quickly.

**Per-turn cost breakdown (blended):**

```
Input tokens per turn:
  System prompt + 10-turn history window (400 chars/msg) + policy context
  ≈ 1,500 tokens × 3.5 calls = ~5,250 input tokens/turn

Output tokens per turn:
  ≈ 300 tokens × 3.5 calls   = ~1,050 output tokens/turn

At 1.4M turns/day (Groq Llama 3.3 70B: ~$0.59/M input, ~$0.79/M output):
  Input:  7.35B tokens → ~$4,340/day
  Output: 1.47B tokens → ~$1,160/day
  Total:                  ≈ $5,500/day → ~$165K/month
```

This is the cost floor before optimizations. Several design decisions in this system directly control it:

- **Sliding window (last 10 turns):** Full conversation history is kept in state but each node receives only the last 10 turns. Without this, input tokens at turn 7 would be 2–3× higher.
- **Planner-level content truncation (400 chars/message):** The planner receives compressed history to make routing decisions. Full content is passed only to nodes that need it.
- **Three-layer planner:** Hard state-flag rules and a direct intent map handle mechanical transitions without an LLM call. LLM is only invoked for genuine claim-reasoning decisions, cutting calls from 4 to 2 on simple turns.
- **fast_llm for routing, main_llm for reasoning:** The router runs on every single turn and uses the cheapest available model. Expensive model calls are reserved for policy coverage assessment and final decision generation.

**Cost-tiered model strategy at scale:**

| Call type | Model tier | Rationale |
|---|---|---|
| Intent routing | Small / fast model | Highest frequency, classification task |
| Claim state extraction | Mid-tier | Structured extraction, moderate complexity |
| Policy coverage reasoning | Main model | Legal-quality reasoning, low frequency |
| Vision analysis | Vision model | Only on turns with uploaded evidence |
| Final decision prose | Main model | Customer-facing, low frequency |

**Provider strategy:** Groq is the primary inference provider for its low-latency throughput. At 137 LLM calls/second steady state, a single provider outage stops all active claims. A fallback provider must be configured with automatic cutover so a provider incident degrades to higher latency, not a full outage.

---

### Rate Limiting

The current implementation applies a per-session cap of 15 turns per 5 minutes as a simple process-local guard. At scale this becomes a two-layer control.

**Edge layer:** Enforce per-IP, per-user, and per-account quotas at the load balancer or API gateway before requests reach application instances. This protects against abuse without consuming application resources.

**Backend throttling:** Cap concurrent agent turns and vision workloads at the worker pool level. Vision turns (evidence_planner → vision_analysis → claim_decision) are 40–50% more expensive than text-only turns. Separate queue depth limits for vision-heavy jobs prevent a burst of image uploads from saturating the model provider.

**Shared state:** Both layers require Redis or Memcached so limits are consistent across all API instances and not reset on instance restart.

---

### Claims Database

Every resolved claim must land in a structured system of record — separate from session artifacts — to support audit, appeals, analytics, and model retraining.

**Schema (core fields):**

| Field | Type | Purpose |
|---|---|---|
| `claim_id` | UUID | Stable claim identifier |
| `session_id` | UUID | Links to full conversation |
| `component` | string | e.g. `LED Display`, `Main Logic Board` |
| `incident_type` | string | e.g. `dead_pixels`, `logic_board_failure` |
| `purchase_date` | date | Warranty eligibility anchor |
| `verdict` | enum | `approved` / `rejected` / `pending` / `escalated` |
| `policy_clauses` | string[] | Clauses cited in the decision |
| `vision_used` | bool | Whether image evidence was evaluated |
| `vision_verdict` | enum | `approved` / `rejected` / `none` |
| `confidence` | float | Validator confidence at decision time |
| `turns_to_resolve` | int | Efficiency signal |
| `escalated` | bool | Whether the session hit escalation |
| `user_feedback` | enum | `like` / `dislike` / `null` |
| `policy_version` | string | Policy corpus version active at decision time |
| `model_version` | string | Model version that issued the verdict |
| `created_at` | timestamp | Decision timestamp |

`policy_version` and `model_version` are critical for auditability. If a policy clause changes or a model is updated, every prior claim must remain traceable to the exact version that produced it. Without this, auditing a disputed claim six months later is not possible.

---

### Feedback Loop

The system already collects `user_feedback` (like/dislike) per session. At scale this becomes a structured pipeline that closes the loop between what the agent decides and what users actually experience.

```
User feedback (like/dislike)
  + Claim outcome (approved/rejected/pending)
  + Turns to resolution
  + Vision verdict vs final verdict
  + Guardrail triggers
        ↓
Feedback store (DB)
        ↓
Offline eval pipeline (sample → LLM-as-judge → score)
        ↓
Drift alerts → prompt/model/retrieval updates
        ↓
CI/CD regression gate before any change ships
```

**Key signals to monitor:**

- **Dislike on an approved claim** — agent approved something the user still contested
- **Escalation after claim_decision** — verdict was correct but communicated poorly, or user disagreed with the outcome
- **Re-upload rate** — image quality gate firing too aggressively or instructions unclear
- **Pending rate trending up** — model struggling to reach decisions, possibly due to retrieval degradation or prompt drift
- **Vision-vs-final verdict disagreement** — claim_validator overriding vision at an increasing rate

These signals feed directly into the drift detection layer described in EVALS.md.

---

### Prototype → Production Migration

1. **Stateless API tier** — Multiple FastAPI instances behind a load balancer; remove all process-local session assumptions.
2. **Externalized session state** — Active session and flow-control state moves to Redis; any worker can resume any session.
3. **Async agent execution** — API tier enqueues turns; dedicated worker pool executes the graph; results streamed back via Redis pub/sub.
4. **Claims system of record** — Every resolved claim written to a structured DB with verdict, policy clauses cited, policy version, and model version. Supports audit, appeals, and analytics.
5. **Feedback pipeline** — User feedback events (like/dislike, escalation, re-upload) written to a feedback store and fed into the offline eval pipeline for continuous quality monitoring.
6. **Edge rate limiting and backend throttling** — Per-IP/user/account quotas at the edge; concurrent agent and vision job caps at the worker layer, backed by Redis.
7. **Provider failover** — Primary and fallback LLM providers configured; automatic cutover on latency SLO breach so a provider incident degrades to higher latency rather than a full outage.
8. **Auditability and versioning** — Policy corpus, prompts, model configuration, and decision logic versioned so every claim outcome is reproducible and reviewable.
9. **Graceful degradation** — If retrieval, vision, or model providers are degraded, fall back to `pending` or manual-review flows rather than failing the claim path outright.

## Model Selection

### Current Choices

The system uses three model tiers, each assigned to a specific class of task, all served through Groq's inference API.

| Role | Model | Rationale |
|---|---|---|
| **Fast / routing** | `llama-3.1-8b-instant` | Runs on every turn. Intent classification across 11 intents is a structured output task — it does not require deep reasoning. 8B is fast and cheap enough that routing cost is negligible even at 1.4M turns/day. |
| **Main / reasoning** | `llama-3.3-70b-versatile` | Policy coverage assessment, claim state extraction, final decision generation — tasks requiring multi-step reasoning over retrieved policy text. 70B provides the reasoning depth needed for legally-grounded decisions without the cost of a frontier closed model. |
| **Vision** | `llama-4-scout-17b-16e-instruct` | Multimodal damage image analysis. Scout handles structured visual reasoning (per-check observations, confidence scoring) at lower cost than GPT-4V, with no dependency on a closed provider. |
| **Embeddings** | `BAAI/bge-m3` (local) | Runs locally via SentenceTransformer — no embedding API cost, no retrieval latency, and BGE-M3's dense retrieval quality brought context recall to 1.00 on the eval dataset. |

Groq was chosen as the inference provider over OpenAI or Anthropic hosted APIs for three reasons: lower latency per call (sub-second p50 on 8B, ~1–2s on 70B), open-weight models that keep decision logic reproducible and auditable, and pricing that makes the cost arithmetic below workable at scale.

---

### Cost Comparison at Scale

Using the capacity numbers from above: **7.35B input tokens/day, 1.47B output tokens/day** across all LLM calls at 1.4M turns/day.

| Option | Input $/M | Output $/M | Est. daily cost | Est. monthly |
|---|---|---|---|---|
| Groq Llama 3.3 70B + 8B (current) | $0.59 / $0.05 | $0.79 / $0.08 | ~$5,500 | ~$165K |
| Gemini 2.5 Flash (text + vision) | $0.30 | $2.50 | ~$5,900 | ~$177K |
| Gemini 2.5 Flash-Lite (routing) | $0.10 | $0.40 | ~$1,300 | ~$39K |
| Gemini 2.5 Pro | $1.25 | $10.00 | ~$23,900 | ~$717K |
| Gemini 3.1 Pro Preview | $2.00 | $12.00 | ~$32,400 | ~$972K |
| Self-hosted Qwen3-32B on H100 | — | — | ~$2,000 | ~$60K |

**Key observations:**

**Gemini 2.5 Pro and 3.1 Pro are prohibitively expensive at this volume.** Output pricing at $10–12/M is 12–15× Groq's $0.79/M. At 1.47B output tokens/day, that difference is $14,700/day in output cost alone. Hard to justify unless the quality gap is decisive, which it is not for structured policy reasoning over retrieved chunks.

**Gemini 2.5 Flash is worth considering.** Similar monthly cost to the current Groq setup (~$177K vs ~$165K) but natively multimodal — one model handles text reasoning and vision analysis, removing the need for a separate vision model tier. Gemini also offers **context caching at $0.20/M cached tokens**. Since the warranty policy corpus is a static document, caching policy chunks means every RAG call pays cache price instead of full input price, potentially cutting retrieval input costs by 60–70%.

**Gemini 2.5 Flash-Lite** at $0.10 input / $0.40 output is a strong candidate to replace the 8B routing model — cheaper on the highest-frequency call, multimodal, same API surface.

**Self-hosted Qwen3-32B is the cheapest option at scale** once infrastructure ops are absorbed, discussed below.

---

### Self-Hosted Inference on H100

At sufficient volume, self-hosted inference becomes cheaper than API pricing. The key is to size correctly by model tier — not all LLM calls hit the 70B model.

**Call breakdown at steady state (39 turns/sec):**

```
8B routing (router + simple terminal nodes): ~51 calls/sec
Vision model (image turns, ~35% of turns):  ~14 calls/sec
70B reasoning (claim turns, ~70% of turns,
  ~2 calls each):                           ~54 calls/sec
```

**Hardware sizing per tier:**

```
70B reasoning tier (Llama 3.3 70B in BF16):
  Requires 2× H100 80GB per serving pair
  Cloud H100: ~$3.50/hr per GPU → $7/hr per pair
  vLLM continuous batching: ~30–40 concurrent requests per pair
  Avg call duration: ~2.5s → ~12–16 calls/sec per pair
  Pairs needed for 54 calls/sec: ~5 pairs = 10 H100s

Vision tier (Scout / Qwen2.5-VL):
  Smaller model, fits on 1× H100
  14 calls/sec comfortably handled by 2 H100s

8B routing tier:
  Very fast, minimal VRAM — 1 H100 handles all routing
  1 H100

Total: ~13 H100s

Daily cost: 13 × $3.50/hr × 24hr ≈ $1,092/day → ~$33K/month
vs Groq API (full stack):              $5,500/day → ~$165K/month
```

Self-hosted is **~5× cheaper** at this volume. That gap is large enough to absorb significant infra ops overhead — GPU cluster management, vLLM configuration and tuning, autoscaling, observability, and on-call engineering time.

**Qwen3-32B reduces the GPU count further.** Qwen3-32B fits on a single H100 80GB without tensor parallelism, is benchmark-competitive with Llama 3.3 70B on structured reasoning, and has a thinking mode that can improve policy coverage reasoning on ambiguous claims. At ~8–10 calls/sec per single H100, the reasoning tier needs 6 H100s instead of 10.

```
Qwen3-32B self-hosted total: ~9 H100s
Daily cost: 9 × $3.50/hr × 24hr ≈ $756/day → ~$23K/month
```

**For vision, Qwen2.5-VL-72B** is meaningfully stronger than LLaVA and competitive with GPT-4V on fine-grained physical damage assessment, spatial reasoning, and document understanding — all directly relevant to warranty evidence evaluation (scratches, dead pixels, burn marks, corrosion).

**Latency engineering overhead.** Hosted APIs like Groq are latency-optimised out of the box. Self-hosted open models require dedicated engineering effort to reach comparable latency:

- **Speculative decoding** — a smaller draft model generates token candidates that the main model verifies in parallel, significantly reducing generation latency at the cost of additional GPU memory.
- **KV cache tuning** — prefix caching for system prompts and static policy context reduces TTFT on repeated call patterns, but requires careful configuration per model and workload.
- **Continuous batching and scheduler tuning** — vLLM's default scheduler settings are not optimal for every workload. Latency vs throughput tradeoffs must be profiled and tuned for the specific call distribution (short routing calls vs long reasoning calls).
- **Quantization tradeoffs** — INT8 or INT4 quantization reduces memory pressure and increases throughput but can degrade output quality on structured reasoning tasks. Each quantization level needs evaluation against the claim decision quality bar before deployment.
- **Autoscaling lag** — unlike API providers that absorb burst traffic instantly, self-hosted GPU clusters have cold-start times. Pre-warming strategies and minimum replica floors are needed to prevent latency spikes under sudden load.

This is a real engineering investment — typically weeks of profiling and tuning to reach production-grade latency — and should be factored into any self-hosting decision alongside the hardware cost savings.

**Bottom line:** Self-hosting ~13 H100s cuts inference cost from ~$165K/month to ~$33K/month at this scale — a 5× saving — but requires upfront GPU infrastructure and dedicated latency engineering that hosted APIs like Groq provide out of the box. The right trigger to self-host is when API spend consistently exceeds the combined cost of hardware and that engineering overhead, typically around the $150–200K/month API spend mark.

---

### Recommended Migration Path

Rather than choosing one provider upfront, the right approach is a cost-tiered progression tied to volume:

```
Stage 1 — Prototype / early scale (today):
  Groq API — fast to ship, zero infra, ~$165K/month floor
  Fast model (8B) for routing, main model (70B) for reasoning,
  Scout for vision.

Stage 2 — Cost optimisation without additional infra:
  Gemini 2.5 Flash for reasoning + vision (single multimodal model)
  Gemini 2.5 Flash-Lite for routing
  Gemini context caching for static policy corpus
  Estimated saving: 20–30% vs all-Groq setup

Stage 3 — Large scale (API spend exceeds ~$200K/month):
  Self-hosted Qwen3-32B (reasoning) + Qwen2.5-VL (vision) on H100
  Groq or Gemini Flash-Lite retained for routing (still cheaper than self-hosting 8B)
  Crossover point where infra ops cost is worth absorbing
```

The LLM gateway (`gateway/llm_gateway.py`) is already abstracted so that model and provider changes are configuration updates, not code changes. This makes the progression above operationally low-risk.

---

### Open vs Closed Models

The choice between open-weight and closed API models is not only a cost question for a claims system — it has direct implications for auditability and data handling.

| Dimension | Open-weight (Llama / Qwen) | Closed API (GPT-4o / Gemini Pro / Claude) |
|---|---|---|
| **Auditability** | Pinned model weights → identical output for identical input, forever | Provider can update model behavior silently between versions |
| **Data privacy** | Claims and PII stay within your infrastructure | Sent to a third-party API endpoint |
| **Cost at scale** | Self-hosting becomes cheaper past ~$150–200K/month API spend | Per-token pricing with no escape valve |
| **Fine-tuning** | Train on your own claims DB once labelled data accumulates | Unavailable or highly restricted |
| **Vendor lock-in** | None — swap providers or hosts freely | Provider deprecates model, migration is forced |
| **Quality ceiling** | Closing fast — 70B competitive on structured reasoning | Still ahead on open-ended generation edge cases |
| **Ops burden** | GPU cluster, vLLM, autoscaling, on-call | Zero — API key and prompt |
| **Time to production** | Weeks (infra setup and tuning) | Days |

**For a claims system, two constraints make open-weight the stronger long-term choice:**

**Reproducibility.** A claim approved in January must be fully explainable in December if disputed. If the underlying model has been silently updated, the reasoning cannot be reproduced. Open-weight models with pinned versions eliminate this risk entirely.

**Data privacy.** Warranty claims contain PII — purchase dates, serial numbers, device identifiers, and user descriptions of incidents. Routing that data through a third-party closed API creates compliance exposure at scale. Self-hosted open models keep all claim data within the operator's own infrastructure.

The quality argument for closed models is the strongest counterpoint, and it is shrinking. Qwen3-32B and Llama 3.3 70B are already sufficient for policy-grounded structured reasoning, which is a narrower task than open-ended generation where frontier models still have a meaningful edge.

**Pragmatic stance:** Start on Groq API (fastest path to production, minimal ops), architect the gateway so provider and model changes are config-level decisions, and migrate to self-hosted open-weight models once volume justifies the infrastructure investment.

