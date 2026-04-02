"""
LLM-as-Judge: Vision Correctness Evaluator

This evaluator runs in two stages for each session:
1. Extract what the trace says about the image as a normalized `vision_claim`
2. Compare the extracted `vision_claim` against dataset `vision_gt`

Usage:
    uv run python llm_as_judge/vision_correctness.py
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

UNKNOWN_VISION_CLAIM = "unclear_or_not_assessed"
NO_VISION_CLAIM = "none"


EXTRACTION_SYSTEM_PROMPT = """You are an expert evaluator for a warranty claims vision pipeline.

Your task is to extract the normalized image-related claim from the trace.

You will be given:
- allowed_labels : the valid normalized vision labels from the dataset
- trace_view     : a trace summary built from the session conversation and saved state

Rules:
- Use only what the trace says about the image or visible evidence.
- Do not infer from the expected warranty outcome alone.
- Prefer the assistant's stated visual conclusion over the user's raw description.
- If the trace shows no image evaluation, no uploaded image, or no vision assessment was needed,
  return "none".
- Use "unclear_or_not_assessed" only when image-based assessment seems relevant but the trace does
  not preserve a clear enough visual conclusion.
- Return exactly one normalized label.

Return JSON only:
{
  "vision_claim": "<one label from allowed_labels or unclear_or_not_assessed>",
  "confidence": <0.0-1.0>,
  "explanation": "<1-2 sentence explanation based only on the trace>"
}"""


COMPARISON_SYSTEM_PROMPT = """You are an expert evaluator for a warranty claims vision pipeline.

Your task is to compare an extracted vision claim against the dataset ground truth.

You will be given:
- vision_gt
- extracted_vision_claim
- extraction_explanation

Scoring:
- 1.0 = extracted claim matches the ground truth or is clearly semantically equivalent
- 0.5 = extracted claim is partially aligned but loses important detail or specificity
- 0.0 = extracted claim does not match, or no clear vision claim was extracted where one was needed

Return JSON only:
{
  "vision_match": <true | false>,
  "vision_correctness": <0.0 | 0.5 | 1.0>,
  "judgment": "<correct | partial | incorrect>",
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
                "image_uploaded": row["image_uploaded"].strip(),
                "vision_gt": row.get("vision_gt", "").strip(),
                "verdict": row["verdict"].strip(),
                "reason": row["reason"].strip(),
            }
    return gt


def load_trace(session_id: str) -> dict:
    path = RESULTS_DIR / session_id / "conversation.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_trace_for_vision(trace: dict) -> str:
    lines: list[str] = []

    lines.append("CONVERSATION:")
    for msg in trace.get("messages", []):
        role = str(msg.get("role", "")).strip().upper()
        content = str(msg.get("content", "")).strip()
        if role and content:
            lines.append(f"{role}: {content}")

    claim_items = trace.get("claim_items", [])
    if claim_items:
        lines.append("")
        lines.append("CLAIM ITEMS:")
        for item in claim_items:
            lines.append(
                "- component={component}; incident_type={incident_type}; has_image={has_image}; "
                "claim_status={claim_status}; warranty_eligible={warranty_eligible}".format(
                    component=item.get("component", "unknown"),
                    incident_type=item.get("incident_type", "unknown"),
                    has_image=item.get("has_image"),
                    claim_status=item.get("claim_status"),
                    warranty_eligible=item.get("warranty_eligible"),
                )
            )

    user_claims = trace.get("user_claims", [])
    if user_claims:
        lines.append("")
        lines.append("USER CLAIMS:")
        for claim in user_claims:
            lines.append(
                "- component={component}; assertion_type={assertion_type}; "
                "requires_visual_validation={requires_visual_validation}; "
                "claim_verdict={claim_verdict}".format(
                    component=claim.get("component", "unknown"),
                    assertion_type=claim.get("assertion_type", "unknown"),
                    requires_visual_validation=claim.get("requires_visual_validation"),
                    claim_verdict=claim.get("claim_verdict"),
                )
            )
            for check in claim.get("visual_checks", []):
                lines.append(
                    "  visual_check: description={description}; expected={expected}; "
                    "finding={finding}; observation={observation}; "
                    "claim_supported={claim_supported}; confidence={confidence}".format(
                        description=check.get("check_description", ""),
                        expected=check.get("expected_finding", ""),
                        finding=check.get("finding", ""),
                        observation=check.get("observation", ""),
                        claim_supported=check.get("claim_supported"),
                        confidence=check.get("confidence"),
                    )
                )

    latest_image_path = trace.get("latest_image_path")
    if latest_image_path:
        lines.append("")
        lines.append(f"LATEST IMAGE PATH: {latest_image_path}")

    return "\n".join(lines)


def call_llm_json(system_prompt: str, user_content: str) -> dict:
    response = litellm.completion(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        api_key=GROQ_API_KEY,
    )
    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


def extract_vision_claim(trace_view: str, allowed_labels: list[str]) -> dict:
    user_content = (
        f"allowed_labels: {json.dumps(allowed_labels)}\n\n"
        f"trace_view:\n{trace_view}"
    )
    return call_llm_json(EXTRACTION_SYSTEM_PROMPT, user_content)


def compare_vision_claim(vision_gt: str, extracted_vision_claim: str, extraction_explanation: str) -> dict:
    user_content = (
        f"vision_gt: {vision_gt}\n"
        f"extracted_vision_claim: {extracted_vision_claim}\n"
        f"extraction_explanation: {extraction_explanation}"
    )
    return call_llm_json(COMPARISON_SYSTEM_PROMPT, user_content)


def run() -> None:
    gt_map = load_gt_by_session()
    valid_labels = sorted(
        {
            row["vision_gt"]
            for row in gt_map.values()
            if row.get("vision_gt")
        }
    )

    session_dirs = sorted(
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and (d / "conversation.json").exists()
    )

    sessions_to_eval = [
        d for d in session_dirs
        if d.name in gt_map and gt_map[d.name].get("vision_gt")
    ]
    sessions_skipped = [
        d.name for d in session_dirs
        if d.name not in gt_map or not gt_map.get(d.name, {}).get("vision_gt")
    ]

    print(f"\n{'=' * 70}")
    print("  LLM-as-Judge: Vision Correctness Evaluation")
    print(f"  Judge model : {JUDGE_MODEL}")
    print(f"  Sessions    : {len(sessions_to_eval)} evaluated, {len(sessions_skipped)} skipped")
    print(f"{'=' * 70}\n")

    results: list[dict] = []
    for i, session_dir in enumerate(sessions_to_eval, start=1):
        session_id = session_dir.name
        gt = gt_map[session_id]
        trace = load_trace(session_id)
        trace_view = format_trace_for_vision(trace)

        print(f"[{i}/{len(sessions_to_eval)}] {gt['scenario']}")
        print(f"      session_id : {session_id}")
        print(f"      vision_gt  : {gt['vision_gt']}")

        extraction = extract_vision_claim(
            trace_view=trace_view,
            allowed_labels=valid_labels,
        )
        vision_claim = extraction.get("vision_claim", UNKNOWN_VISION_CLAIM)
        extraction_confidence = extraction.get("confidence", 0.0)
        extraction_explanation = extraction.get("explanation", "")

        comparison = compare_vision_claim(
            vision_gt=gt["vision_gt"],
            extracted_vision_claim=vision_claim,
            extraction_explanation=extraction_explanation,
        )

        score = comparison.get("vision_correctness", 0.0)
        label = comparison.get("judgment", "unknown")
        match = comparison.get("vision_match", False)
        explanation = comparison.get("explanation", "")

        print(f"      vision_claim : {vision_claim}")
        print(f"      match        : {match}")
        print(f"      score        : {score}  [{label.upper()}]")
        print(f"      explanation  : {explanation}")
        print()

        results.append(
            {
                "session_id": session_id,
                "scenario": gt["scenario"],
                "vision_gt": gt["vision_gt"],
                "vision_claim": vision_claim,
                "extraction_confidence": extraction_confidence,
                "extraction_explanation": extraction_explanation,
                "vision_match": match,
                "vision_correctness": score,
                "judgment": label,
                "explanation": explanation,
            }
        )

    avg = sum(r["vision_correctness"] for r in results) / len(results) if results else 0.0
    correct = sum(1 for r in results if r["judgment"] == "correct")
    partial = sum(1 for r in results if r["judgment"] == "partial")
    incorrect = sum(1 for r in results if r["judgment"] == "incorrect")
    missing_trace_signal = sum(
        1 for r in results
        if r["vision_claim"] == UNKNOWN_VISION_CLAIM
    )

    print(f"{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Sessions evaluated     : {len(results)}")
    print(f"  Correct                : {correct}")
    print(f"  Partial                : {partial}")
    print(f"  Incorrect              : {incorrect}")
    print(f"  Missing trace signal   : {missing_trace_signal}")
    print(f"  Avg vision_correctness : {avg:.2f}")
    print(f"{'=' * 70}\n")

    out_path = Path(__file__).parent / "vision_correctness_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "judge_model": JUDGE_MODEL,
                "avg_vision_correctness": round(avg, 4),
                "sessions_evaluated": len(results),
                "sessions_skipped": sessions_skipped,
                "missing_trace_signal": missing_trace_signal,
                "allowed_labels": valid_labels,
                "scenarios": results,
            },
            f,
            indent=2,
        )
    print(f"  Results saved to: {out_path}\n")


if __name__ == "__main__":
    run()
