"""
Evidence planning service for visual check generation.
"""
from __future__ import annotations

import json
import logging

from gateway.llm_gateway import fast_llm
from agent.prompt_store import get_prompt
from models.state import VisualCheck
from observability.tracing import start_span

logger = logging.getLogger(__name__)


def plan_visual_checks(component: str, issue_summary: str) -> list[dict]:
    with start_span(
        "evidence.plan_visual_checks",
        {"claim.component": component, "claim.issue_summary": issue_summary},
    ) as span:
        prompt = f"Component: {component}\nCustomer issue: {issue_summary}"
        messages = [
            {"role": "system", "content": get_prompt("evidence_service")},
            {"role": "user", "content": prompt},
        ]

        raw = fast_llm(messages, json_mode=True)
        checks: list[dict] = []
        try:
            data = json.loads(raw)
            raw_checks = data.get("checks", [])
            for raw_check in raw_checks:
                check = VisualCheck(
                    check_description=raw_check.get("check_description", ""),
                    expected_finding=raw_check.get("expected_finding", ""),
                )
                checks.append(check.model_dump())
        except Exception as exc:
            span.record_exception(exc)
            logger.warning("Evidence planner parse failed for %s", component)
            checks = [
                VisualCheck(
                    check_description="Overall damage assessment",
                    expected_finding="Visible physical damage consistent with reported issue",
                ).model_dump()
            ]

        span.set_attribute("evidence.check_count", len(checks))
        return checks
