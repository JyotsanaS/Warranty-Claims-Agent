# Evals

This prototype does not need a large benchmark report to be credible. It does need a defensible evaluation plan.

For a warranty agent, the central question is not "does the model sound good?" It is "does the system make the right decision, for the right reason, with evidence I can audit?" That changes what should be measured.

The evaluation strategy below is built around the actual failure modes of this system:

- routing the user into the wrong workflow
- retrieving the wrong policy context
- approving or rejecting a claim with weak grounding
- over-trusting low-quality images
- producing a response that sounds reasonable but is not operationally safe

## What I Would Optimize For

I would optimize this system against four outcomes:

1. policy-grounded correctness
2. low false-approval rate
3. predictable latency and cost
4. traceability when something goes wrong

Those priorities are deliberate. In this domain, a confident but incorrect approval is materially worse than asking for clarification or escalating to review.

## Evaluation Layers

I would evaluate the agent at three layers, not one.

### 1. Component-level checks

These are fast regressions that isolate a single subsystem.

- router intent classification
- planner next-step selection
- policy retrieval relevance
- image-quality gate behavior
- hallucination guardrail behavior

This repo already has the start of that approach in [`tests/`](/home/antpc/Desktop/chargepoint/tests): planner tests, router tests, retrieval tests, hallucination guardrail tests, and image-quality tests. That is the right foundation because it catches logic regressions before they become "LLM quality" discussions.

### 2. Scenario-level end-to-end evals

This is the most important layer.

I would maintain a small golden set of complete warranty scenarios, each with:

- user turns
- optional image attachment
- expected retrieval target sections
- expected final claim outcome
- expected explanation constraints

The point is not to prove the system is perfect. The point is to make sure a model swap, prompt edit, retrieval change, or guardrail tweak does not silently change business behavior.

### 3. Production monitoring

Offline evals catch regressions in a controlled environment. They do not tell me whether the system is drifting in the field.

In production I would monitor:

- error and latency
- retrieval hit quality
- guardrail trigger rates
- image re-upload rates
- escalation rates
- approval / rejection distribution shifts

That gives a practical signal that the agent is encountering traffic it was not tuned for.

## What I Would Measure

### Routing and planning

The router and planner are cheap to test and disproportionately important.

Metrics:

- intent accuracy
- planner next-node accuracy
- context-switch detection precision/recall
- incorrect terminal routing rate

Failure to watch:

- a user asking a policy question and getting routed into a generic conversational reply
- a new claim being merged into the wrong existing claim context

### Retrieval

For this system, retrieval quality is upstream of almost every important decision. If retrieval degrades, everything after it becomes harder to trust.

Metrics:

- hit@k for expected sections
- recall@k for required clauses
- empty-retrieval rate
- duplicate chunk rate after deduping

Failure to watch:

- correct section exists in the corpus but never appears in the retrieved top-k
- final answer cites a clause that was never actually retrieved

### Coverage reasoning

This is where the business risk sits.

Metrics:

- final verdict accuracy
- precision / recall by verdict class
- false approval rate
- unsupported citation rate
- groundedness / faithfulness score

Of these, false approval rate is the most important. If I had to choose one hard release gate for this project, it would be "do not regress false approvals."

### Vision and evidence handling

The system should not behave as though an unreadable image is valid evidence.

Metrics:

- blurry / undersized image rejection accuracy
- invalid image fail-safe rate
- visual-check completion rate
- pending-for-insufficient-evidence rate

Failure to watch:

- low-quality images being treated as valid evidence
- image-analysis uncertainty being converted into unjustified confidence

### End-to-end claim handling

The user experiences the agent as one system, not five subsystems.

Metrics:

- end-to-end claim outcome accuracy
- average turns to resolution
- unnecessary escalation rate
- user follow-up rate after decision
- p50 / p95 end-to-end latency
- token cost per resolved claim

## LLM-as-a-Judge

I would use LLM-as-a-judge, but only as a regression tool, not as the source of truth.

Its role here is to score qualities that are expensive to check manually on every run:

- whether the final explanation is grounded in retrieved policy
- whether the response handles uncertainty appropriately
- whether the agent's explanation is internally consistent with the verdict
- whether the response is clear enough for a customer-facing workflow

I would not let the judge decide whether the business outcome is correct in isolation. For high-risk cases, especially approvals, I would anchor evaluation to expected labels and human review.

### Judge input

Each scenario in the regression set should include:

- the user conversation
- retrieved policy chunks
- any image-analysis output
- final agent response
- expected verdict

### Judge output

The judge should produce:

- a pass/fail recommendation
- rubric scores for grounding, correctness, clarity, and uncertainty handling
- a short written rationale

### Where I would not trust the judge

- approval scenarios
- ambiguous policy language
- mixed evidence cases
- cases where the model must say "I do not have enough evidence"

Those are exactly the cases where human review still matters.

## Production Observability

A production-grade agent needs two kinds of observability: system telemetry and decision telemetry.

### System telemetry

- request rate
- error rate
- p50 / p95 / p99 latency
- stream completion rate
- Pinecone latency
- model latency by route
- token usage by model

### Decision telemetry

- retrieved chunk count
- empty retrieval rate
- guardrail failure / fallback rate
- image-quality rejection rate
- claim verdict distribution
- escalation rate
- manual-review rate

This repo has the beginning of that foundation through [`observability/tracing.py`](/home/antpc/Desktop/chargepoint/observability/tracing.py) and streamed node traces in the agent flow, but the instrumentation is not complete yet. In particular, I would still want richer capture of retrieval details, selected chunks, intermediate planner decisions, guardrail outcomes, and the exact evidence used to reach a final claim decision.

## Drift Strategy

I would expect drift in three places.

### Retrieval drift

Causes:

- policy corpus changes
- embedding-model changes
- Pinecone index or namespace issues

Signals:

- declining hit@k on the golden set
- rising empty-retrieval rate
- more cases where cited clauses do not appear in retrieved chunks

### Image drift

Causes:

- different camera quality
- worse lighting
- new device or damage patterns
- users uploading the wrong evidence artifact

Signals:

- rising re-upload rate
- higher pending rate due to insufficient evidence
- lower confidence in visual checks

### Conversation drift

Causes:

- new user phrasing patterns
- unsupported issue types
- product-policy changes not reflected in prompts or retrieval data

Signals:

- increased fallback rate
- more clarification turns
- higher escalation rate after policy explanation

## Release Gates

Before I would trust a prompt or model change in this repo, I would require:

- unit tests passing
- retrieval regression set passing
- end-to-end golden scenarios passing
- no increase in false approval rate
- no material regression in grounding scores
- no unacceptable latency or cost jump

That is intentionally stricter than "the outputs still look reasonable."

## What I Would Build Next

The fastest path from prototype to credible evaluation would be:

1. create 25-50 golden end-to-end claim scenarios
2. label each with expected sections, verdict, and explanation constraints
3. run them in CI on every meaningful agent change
4. score them for verdict correctness, grounding, latency, and cost
5. require human review for approvals and ambiguous failures

Those checks should be added to the CI/CD pipeline so prompt, model, retrieval, and workflow changes are evaluated before release rather than after production drift is observed.
