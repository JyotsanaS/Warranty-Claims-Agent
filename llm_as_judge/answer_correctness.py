"""
LLM-as-Judge: Answer Correctness Evaluator

Loads all session trace folders under results/, joins each to the ground-truth
dataset via session_id, then uses an LLM judge to evaluate answer_correctness
based solely on what the agent communicated to the user in the conversation.

Metric: answer_correctness
  1.0 — agent clearly communicated the correct verdict AND grounded it in policy
  0.5 — agent communicated the correct verdict but reasoning was vague/uncited
  0.0 — agent communicated the wrong verdict, or failed to reach one

Usage:
    uv run python llm_as_judge/answer_correctness.py
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import litellm
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

JUDGE_MODEL = "groq/llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DATASET_PATH = ROOT / "data" / "scenarios" / "dataset.csv"
RESULTS_DIR = ROOT / "results"

# ── Judge prompt ──────────────────────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a warranty claims AI agent called VoltEdge.

Your task is to assess **answer_correctness**: did the agent clearly communicate the
correct verdict to the user, and was that verdict grounded in policy?

You will be given:
- gt_verdict       : the ground-truth expected verdict (approved or rejected)
- gt_reason        : the expected policy-based justification
- conversation     : the full conversation between user and agent (USER / ASSISTANT turns)

Your job is to read the ASSISTANT turns and determine:
1. What verdict did the agent actually communicate to the user?
   (approved = replacement/repair offered; rejected = not covered / excluded)
2. Did it match the gt_verdict?
3. Was the reasoning grounded in relevant policy (citations, clauses, exclusions)?

Do NOT look at internal state or structured data — only at what the agent said.

Scoring:
  1.0 — verdict communicated matches gt AND policy reasoning is clear and cited
  0.5 — verdict communicated matches gt BUT reasoning is vague or policy not cited
  0.0 — verdict communicated does NOT match gt, or agent never communicated a verdict

Return JSON only:
{
  "agent_communicated_verdict": "<approved | rejected | unclear>",
  "verdict_match": <true | false>,
  "answer_correctness": <0.0 | 0.5 | 1.0>,
  "judgment": "<correct | partial | incorrect>",
  "explanation": "<1-2 sentence explanation>"
}"""


def load_gt_by_session() -> dict[str, dict]:
    """Return {session_id: {scenario, verdict, reason, user_query}} from dataset CSV."""
    gt: dict[str, dict] = {}
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row["session_id"].strip()
            gt[sid] = {
                "scenario": row["scenario"].strip(),
                "verdict": row["verdict"].strip(),
                "reason": row["reason"].strip(),
                "user_query": row["user_query"].strip(),
            }
    return gt


def load_trace(session_id: str) -> dict:
    path = RESULTS_DIR / session_id / "conversation.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_conversation(trace: dict) -> str:
    """Return only USER / ASSISTANT turns as a plain string."""
    lines = []
    for msg in trace.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        if role == "user":
            lines.append(f"USER: {content}")
        elif role == "assistant":
            lines.append(f"ASSISTANT: {content}")
    return "\n".join(lines)


def call_judge(gt_verdict: str, gt_reason: str, conversation: str) -> dict:
    user_content = (
        f"gt_verdict : {gt_verdict}\n"
        f"gt_reason  : {gt_reason}\n\n"
        f"conversation:\n{conversation}"
    )
    response = litellm.completion(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        api_key=GROQ_API_KEY,
    )
    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


def run() -> None:
    gt_map = load_gt_by_session()

    # Discover all session folders that have a conversation.json
    session_dirs = sorted(
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and (d / "conversation.json").exists()
    )

    # Filter to sessions we have ground truth for
    sessions_to_eval = [d for d in session_dirs if d.name in gt_map]
    sessions_skipped = [d.name for d in session_dirs if d.name not in gt_map]

    print(f"\n{'='*70}")
    print("  LLM-as-Judge: Answer Correctness Evaluation")
    print(f"  Judge model : {JUDGE_MODEL}")
    print(f"  Sessions    : {len(sessions_to_eval)} evaluated, {len(sessions_skipped)} skipped (no gt)")
    print(f"{'='*70}\n")

    results = []
    for i, session_dir in enumerate(sessions_to_eval, start=1):
        session_id = session_dir.name
        gt = gt_map[session_id]

        trace = load_trace(session_id)
        conversation = format_conversation(trace)

        print(f"[{i}/{len(sessions_to_eval)}] {gt['scenario']}")
        print(f"      session_id : {session_id}")
        print(f"      gt_verdict : {gt['verdict']}")

        judgment = call_judge(
            gt_verdict=gt["verdict"],
            gt_reason=gt["reason"],
            conversation=conversation,
        )

        score = judgment.get("answer_correctness", 0.0)
        label = judgment.get("judgment", "unknown")
        communicated = judgment.get("agent_communicated_verdict", "unclear")
        explanation = judgment.get("explanation", "")

        print(f"      agent said : {communicated}")
        print(f"      score      : {score}  [{label.upper()}]")
        print(f"      explanation: {explanation}")
        print()

        results.append({
            "session_id": session_id,
            "scenario": gt["scenario"],
            "gt_verdict": gt["verdict"],
            "agent_communicated_verdict": communicated,
            "answer_correctness": score,
            "judgment": label,
            "explanation": explanation,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    avg = sum(r["answer_correctness"] for r in results) / len(results) if results else 0.0
    correct = sum(1 for r in results if r["judgment"] == "correct")
    partial = sum(1 for r in results if r["judgment"] == "partial")
    incorrect = sum(1 for r in results if r["judgment"] == "incorrect")

    print(f"{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Sessions evaluated  : {len(results)}")
    print(f"  Correct             : {correct}")
    print(f"  Partial             : {partial}")
    print(f"  Incorrect           : {incorrect}")
    print(f"  Avg answer_correctness: {avg:.2f}")
    print(f"{'='*70}\n")

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "judge_model": JUDGE_MODEL,
                "avg_answer_correctness": round(avg, 4),
                "sessions_evaluated": len(results),
                "sessions_skipped": sessions_skipped,
                "scenarios": results,
            },
            f,
            indent=2,
        )
    print(f"  Results saved to: {out_path}\n")


if __name__ == "__main__":
    run()
