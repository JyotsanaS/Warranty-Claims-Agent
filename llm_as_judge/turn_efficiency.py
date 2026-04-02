"""
LLM-as-Judge: Turn Efficiency Evaluator

This evaluator scores whether each conversation resolved the claim efficiently.
It combines deterministic turn counts from the saved conversation with an LLM
judge that evaluates whether the number of turns was reasonable for the task.

Usage:
    uv run python llm_as_judge/turn_efficiency.py
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import litellm
from dotenv import load_dotenv


ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

JUDGE_MODEL = "groq/llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DATASET_PATH = ROOT / "data" / "scenarios" / "dataset.csv"
RESULTS_DIR = ROOT / "results"


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a warranty claims AI agent called VoltEdge.

Your task is to assess **turn_efficiency**: did the agent resolve the claim in a
reasonable number of turns, without unnecessary back-and-forth?

You will be given:
- scenario           : the scenario name
- user_query         : the original user request
- gt_verdict         : the expected claim outcome
- conversation       : the full conversation between user and assistant
- user_turns         : number of USER messages
- assistant_turns    : number of ASSISTANT messages
- total_turns        : total USER + ASSISTANT messages

Focus on whether the assistant asked only for information that was necessary,
avoided repetitive clarification, and moved the claim toward resolution
efficiently.

Scoring:
  1.0 — resolved efficiently with no clearly unnecessary turns
  0.5 — mostly resolved, but with some avoidable back-and-forth or minor delay
  0.0 — inefficient flow with clear unnecessary turns, repetition, or failure to
        move the claim forward in a reasonable way

Return JSON only:
{
  "claim_resolved": <true | false>,
  "turn_efficiency": <0.0 | 0.5 | 1.0>,
  "judgment": "<efficient | partial | inefficient>",
  "explanation": "<1-2 sentence explanation>"
}"""


def load_gt_by_session() -> dict[str, dict]:
    gt: dict[str, dict] = {}
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            session_id = row["session_id"].strip()
            gt[session_id] = {
                "scenario": row["scenario"].strip(),
                "user_query": row["user_query"].strip(),
                "verdict": row["verdict"].strip(),
                "reason": row["reason"].strip(),
            }
    return gt


def load_trace(session_id: str) -> dict:
    path = RESULTS_DIR / session_id / "conversation.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_conversation(trace: dict) -> str:
    lines: list[str] = []
    for msg in trace.get("messages", []):
        role = msg.get("role", "")
        content = str(msg.get("content", "")).strip()
        if role == "user":
            lines.append(f"USER: {content}")
        elif role == "assistant":
            lines.append(f"ASSISTANT: {content}")
    return "\n".join(lines)


def count_turns(trace: dict) -> dict[str, int]:
    user_turns = 0
    assistant_turns = 0

    for msg in trace.get("messages", []):
        role = msg.get("role", "")
        if role == "user":
            user_turns += 1
        elif role == "assistant":
            assistant_turns += 1

    return {
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "total_turns": user_turns + assistant_turns,
    }


def call_judge(
    scenario: str,
    user_query: str,
    gt_verdict: str,
    conversation: str,
    user_turns: int,
    assistant_turns: int,
    total_turns: int,
) -> dict:
    user_content = (
        f"scenario        : {scenario}\n"
        f"user_query      : {user_query}\n"
        f"gt_verdict      : {gt_verdict}\n"
        f"user_turns      : {user_turns}\n"
        f"assistant_turns : {assistant_turns}\n"
        f"total_turns     : {total_turns}\n\n"
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

    session_dirs = sorted(
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and (d / "conversation.json").exists()
    )

    sessions_to_eval = [d for d in session_dirs if d.name in gt_map]
    sessions_skipped = [d.name for d in session_dirs if d.name not in gt_map]

    print(f"\n{'=' * 70}")
    print("  LLM-as-Judge: Turn Efficiency Evaluation")
    print(f"  Judge model : {JUDGE_MODEL}")
    print(f"  Sessions    : {len(sessions_to_eval)} evaluated, {len(sessions_skipped)} skipped (no gt)")
    print(f"{'=' * 70}\n")

    results: list[dict] = []
    for i, session_dir in enumerate(sessions_to_eval, start=1):
        session_id = session_dir.name
        gt = gt_map[session_id]
        trace = load_trace(session_id)
        conversation = format_conversation(trace)
        turn_counts = count_turns(trace)

        print(f"[{i}/{len(sessions_to_eval)}] {gt['scenario']}")
        print(f"      session_id      : {session_id}")
        print(f"      user_turns      : {turn_counts['user_turns']}")
        print(f"      assistant_turns : {turn_counts['assistant_turns']}")
        print(f"      total_turns     : {turn_counts['total_turns']}")

        judgment = call_judge(
            scenario=gt["scenario"],
            user_query=gt["user_query"],
            gt_verdict=gt["verdict"],
            conversation=conversation,
            user_turns=turn_counts["user_turns"],
            assistant_turns=turn_counts["assistant_turns"],
            total_turns=turn_counts["total_turns"],
        )

        score = judgment.get("turn_efficiency", 0.0)
        label = judgment.get("judgment", "unknown")
        claim_resolved = judgment.get("claim_resolved", False)
        explanation = judgment.get("explanation", "")

        print(f"      resolved        : {claim_resolved}")
        print(f"      score           : {score}  [{label.upper()}]")
        print(f"      explanation     : {explanation}")
        print()

        results.append(
            {
                "session_id": session_id,
                "scenario": gt["scenario"],
                "gt_verdict": gt["verdict"],
                "user_turns": turn_counts["user_turns"],
                "assistant_turns": turn_counts["assistant_turns"],
                "total_turns": turn_counts["total_turns"],
                "claim_resolved": claim_resolved,
                "turn_efficiency": score,
                "judgment": label,
                "explanation": explanation,
            }
        )

    avg = sum(r["turn_efficiency"] for r in results) / len(results) if results else 0.0
    efficient = sum(1 for r in results if r["judgment"] == "efficient")
    partial = sum(1 for r in results if r["judgment"] == "partial")
    inefficient = sum(1 for r in results if r["judgment"] == "inefficient")
    avg_total_turns = sum(r["total_turns"] for r in results) / len(results) if results else 0.0

    print(f"{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Sessions evaluated   : {len(results)}")
    print(f"  Efficient            : {efficient}")
    print(f"  Partial              : {partial}")
    print(f"  Inefficient          : {inefficient}")
    print(f"  Avg turn_efficiency  : {avg:.2f}")
    print(f"  Avg total_turns      : {avg_total_turns:.2f}")
    print(f"{'=' * 70}\n")

    out_path = Path(__file__).parent / "turn_efficiency_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "judge_model": JUDGE_MODEL,
                "avg_turn_efficiency": round(avg, 4),
                "avg_total_turns": round(avg_total_turns, 4),
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
