# [Chargepoint] - Senior Staff AI Engineer (Agentic AI)

## Take‑Home Technical Exercise - Candidate Instructions

## 📌 Overview

This exercise simulates a core challenge in our ecosystem: **building a multimodal autonomous agent** that can handle customer support, policy reasoning, and visual verification.

You are tasked with building a prototype for an **Automated Warranty & Claims Agent**. The system must converse with a user to understand a hardware issue, retrieve relevant company policies (RAG), and validate "proof of damage" via image analysis to decide if a claim should be approved.

This assignment is designed to be completed in **2 working days** of active work.

---

## 🧩 Requirements

You will submit your work as a **public GitHub repository** containing:

### **1. Functional Prototype**

A Python-based implementation (FastAPI, Flask, Django) that demonstrates:

* **Reasoning Engine:** An LLM agent that can maintain state and drive a conversation.
* **RAG Integration:** The agent must query a provided (or synthetic) policy document to justify its decisions.
* **Visual Validator:** A module that analyzes uploaded images (e.g., a cracked screen or a receipt) to extract structured "damage reports."

### **2. System Architecture Document (`ARCHITECTURE.md`)**

As a Senior Staff Engineer, your technical strategy is as important as your code. Provide:

* **Production Design:** A diagram or detailed description of how this scales to 1M+ users (handling concurrency, rate limiting, and cost).
* **Model Selection:** Rationale for choosing specific models (e.g., GPT-4o vs. fine-tuned Llama 3 / LLaVA).
* **Guardrails:** How you implement safety layers to prevent prompt injection or "policy hallucinations."

### **3. Evaluation & Observability Framework (`EVALS.md`)**

Describe how you would move this from "prototype" to "production grade":

* Define the metrics you would use to measure agent success (e.g., faithfulness, relevancy).
* Propose an automated "LLM-as-a-Judge" pipeline for regression testing.
* Detail how you would monitor image classifier "drift" in the wild.

### **4. AI Tool Usage (`AI_USAGE.md`)**

Document your collaboration with AI (Copilot, Claude, etc.):

* Where did AI accelerate your development?
* Where did you have to override AI suggestions to ensure architectural integrity?

---

## 📊 The Data & Scope

For this exercise, you should use/create:

1. **Policy Base:** A small text corpus (Markdown or PDF) defining warranty rules (e.g., "Screens are covered for 12 months; water damage is not").
2. **Test Images:** 3–5 sample images (real or web-sourced) representing "Damaged," "Normal," and "Invalid/Random" states.

---

## 🧪 Technical Expectations

### **1. Multimodal Logic**

The system shouldn't just "see" an image; it should **reason** about it.

* *Example:* If the user says "My phone fell in the pool," the agent should look for corrosion in the image and cross-reference the "Water Damage" clause in the policy.

### **2. Structured Outputs**

The image validator must return structured data (JSON), not just prose, so the agent can programmatically decide the next workflow step.

### **3. Reliability**

Handle edge cases: What happens if the RAG system returns no relevant context? What if the image is too blurry?

---

## 📝 Evaluation Criteria

**Submissions will be evaluated on the candidate's ability to transform a non-deterministic multimodal prototype into a production-grade, observable, and architecturally resilient system that balances cost, latency, and reasoning accuracy.**

---

### **Communication**

* Example: Can you explain complex trade-offs (e.g., Latency vs. Accuracy) to non-technical stakeholders in your report?

---

## 📦 Submission Instructions

1. Create a **public GitHub repository**.
2. Include a `README.md` with clear setup instructions (e.g., `pip install -r requirements.txt`).
3. Use Docker and Docker Compose to package your entire system (including the agent, any local models, and vector databases) to ensure a "one-command" setup experience for the reviewers.
4. Ensure all environment variables (API keys) are handled via a `.env.example` file.
4. Share the link with the hiring team.

---

## 🧡 A Note to Candidates

We respect your time. We are looking for **depth of thought**, **technical judgment**, and **code-quality** over sheer volume of code. This is your chance to show us how you think about AI at scale.

---