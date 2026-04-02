# Evals

## Purpose of Evals

The goal of the eval strategy is to measure whether the system can resolve warranty and claims conversations correctly, with the right evidence, and without unnecessary back-and-forth.

This is not a generic chatbot evaluation problem. The system is expected to understand what the user is asking, choose the right next step, retrieve the correct policy context, reason over that policy without hallucinating, interpret visual evidence correctly, and clearly communicate the final outcome.

Because of that, the eval plan should cover both end-to-end behavior and component-level behavior. End-to-end evals tell us whether the product works as a complete system. Component-level evals help isolate where failures are coming from when end-to-end performance drops.

## System Components Being Evaluated

The system should be evaluated across the main components involved in claim handling:

- Intent classification: whether the router correctly identifies what the user is trying to do
- Planning: whether the planner chooses the correct next node or next action
- Claim state extraction: whether the system captures the right structured claim details from the conversation
- RAG / policy retrieval: whether the system retrieves the right policy sections needed to make a decision
- Coverage reasoning: whether the system applies the retrieved policy correctly to the user's claim
- Vision and evidence analysis: whether the vision model correctly interprets uploaded images without inventing damage or evidence
- Image quality check: whether low-quality or unusable images are rejected correctly
- Final decision generation: whether the final user-facing response clearly communicates the correct verdict with the right grounding

## End-to-End Evals

End-to-end evals should be the primary way we judge the system. The user experiences one product, not separate router, planner, retrieval, vision, and guardrail components. This layer tells us whether the full claim-handling flow works from start to finish.

Each end-to-end eval should represent a realistic claim scenario. That can include a multi-turn conversation, an optional image, the expected claim outcome, and the policy basis for that outcome.

The main end-to-end metrics should be:

- Answer correctness: whether the final response communicates the correct decision with the right policy grounding
- Turn efficiency: how many turns it takes to resolve the claim
- End-to-end hallucination rate: whether the final flow included fabricated policy details, unsupported reasoning, or invented visual findings

We have implemented an LLM-as-judge metric for Answer Correctness.

This metric evaluates whether the final assistant response communicated the correct claim decision and whether that decision was properly grounded in policy. The judge reads only the user-facing conversation, not the internal agent state, because the purpose of the metric is to evaluate what the system actually communicated to the user.

The scoring rubric is:

- `1.0`: correct decision, correctly grounded in policy
- `0.5`: correct decision, but weak or missing policy grounding
- `0.0`: incorrect decision, or no clear decision communicated

On the current sample dataset, the baseline Answer Correctness score is `0.88` across 4 evaluated scenarios. The main gap in the current results is not incorrect decisions, but incomplete policy grounding in one partially correct case.

## Component-Level Evals

Component-level evals are meant to isolate failures in individual parts of the system. End-to-end evals tell us whether the system worked. Component-level evals tell us where it failed.

The main component-level evals for this system should be:

- Intent classification accuracy: whether the router correctly identifies the user's intent
- Planning accuracy: whether the planner chooses the correct next step from the current state
- RAG evals: whether retrieval returns the right policy context needed to make the decision
- Vision correctness: whether the vision model correctly interprets the uploaded image without inventing evidence

For RAG, the primary metrics should be:

- Context recall
- Context precision

These are especially important because retrieval quality directly affects policy grounding and final decision quality.

## Guardrails

Some protections in the system are enforced while the agent is running, rather than evaluated only through offline component evals. These guardrails are part of the runtime safety layer and help prevent bad outputs from reaching the user or affecting the claim flow.

The main guardrails currently implemented are:

- Policy hallucination detection: checks whether policy reasoning is grounded and helps prevent unsupported or fabricated policy claims
- Image quality check: blocks low-quality, blurry, or unusable images from being treated as valid evidence
- Prompt injection detection: detects malicious or instruction-overriding inputs and prevents them from affecting the workflow

## Production Setup

A usable eval framework needs to be part of the production setup, not something run occasionally in isolation.

### Instrumentation and Trace Collection

The first requirement is consistent trace capture from live runs. We have already added tracing through OpenInference, and sample traces are available in `results`. This gives us the foundation needed to inspect how the system behaved across routing, planning, retrieval, vision, guardrails, and final decision generation.

For each production session, we should capture enough information to reconstruct and evaluate the full claim flow, including:

- Conversation history
- Final claim outcome
- Retrieved policy context
- Planner decisions
- Vision outputs
- Guardrail triggers and outcomes

Without this level of trace data, it is difficult to diagnose whether a failure came from intent classification, planning, retrieval, evidence handling, or response generation.

### Continuous Evaluation

Collected traces should feed directly into the eval framework. For smaller volumes, results can be evaluated continuously as new traces come in. For larger volumes, we should not try to evaluate everything blindly. Instead, we should sample traces intelligently and prioritize the ones most likely to surface failures.

This is where active learning becomes useful. Rather than reviewing only random traffic, we should bias evaluation toward traces that show uncertainty, unusual behavior, guardrail activity, long conversations, or disagreement with expected patterns. That allows the eval process to stay efficient while still catching important regressions.

### Evaluation Cost and Model Strategy

The cost of evaluation also needs to be treated as part of the production setup. Even when we use an open-source judge model such as Llama 70B, evaluation is not free. The cost simply shifts from API pricing to infrastructure, including GPU capacity, hosting, batching, scheduling, storage, and maintenance.

There is also a practical tradeoff between closed and open-source models for evaluation. Closed models are usually easier to plug in and operate, but their cost and behavior remain dependent on the provider. Open-source models give us more control over deployment, reproducibility, tuning, and long-term behavior, but they come with additional infrastructure and operational complexity. The advantage of open-source models is that their evaluation behavior is ultimately in our control, which matters if we want to tune them for the domain over time.

For that reason, the eval stack should not depend on one expensive judge for every case. It should be designed as a cost-tiered system.

The first layer should be deterministic checks and standard tests for anything that does not require an LLM judge. Unit tests, scenario tests, guardrail checks, and rule-based validations should remain the cheapest layer and should run most often.

The second layer can use smaller or cheaper models for narrow evaluation tasks. In many places, a full large-model judge is unnecessary. Smaller classifiers, lightweight local models, or narrow deterministic evaluators can be used to validate specific behaviors such as intent quality, citation presence, trace anomalies, or unusually long claim flows.

The most expensive judge models should be reserved for the cases where they add the most value. That includes benchmark datasets, sampled production traces, ambiguous cases, and flows that look risky based on uncertainty, long turn count, unusual retrieval patterns, or guardrail activity.

This is also where sampling becomes important. In production, we should not evaluate every trace with the heaviest judge. We should continuously evaluate where possible, but once traffic grows, the system should move toward active sampling and prioritization. That allows us to spend evaluation cost on the traces most likely to reveal regressions or drift.

A practical long-term setup is therefore a hybrid one:

- Deterministic and unit-level checks for fast and cheap regression coverage
- Smaller or cheaper models for filtering, triage, and narrow evaluation tasks
- Larger judge models only for benchmark runs, sampled production traces, and hard cases

That kind of layered design keeps evaluation operationally affordable while still preserving enough depth to catch failures that simple tests will miss.

### CI/CD Integration

The eval framework should also be part of the release pipeline. Changes to prompts, models, retrieval settings, or workflow logic should trigger evaluation before they ship.

At a minimum, the CI/CD setup should include:

- Scenario testing for regression coverage
- Unit tests for component-level behavior
- Offline benchmarking for both end-to-end and component-level evals

We have already added a few `pytest` tests in the `tests` folder, which is a good start. That coverage should be expanded further so that routing, planning, retrieval, guardrails, vision, and decision logic are all tested more thoroughly over time.

### Drift Detection

Production setup should also support drift detection. Even if the system remains operational, its behavior can still shift in ways that reduce quality.

We should monitor for changes in:

- Intent distribution
- Approval, rejection, and pending rates
- Average turns to resolution
- Escalation rate
- Retrieval quality
- Policy hallucination guardrail triggers
- Image re-upload and insufficient-evidence patterns

This is what makes the eval framework practical. It allows us to benchmark the system offline, validate changes before release, and detect when production behavior starts moving away from expected performance.
