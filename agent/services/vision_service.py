"""
Vision processing service for claim image analysis.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
from pathlib import Path

import numpy as np
from PIL import Image

from agent.prompt_store import get_prompt
from gateway.llm_gateway import vision_llm
from models.state import VisualCheck
from observability.tracing import start_span
from storage.image_store import read_image

logger = logging.getLogger(__name__)


def _detect_mime_type(image_path: str, raw_bytes: bytes) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type and mime_type.startswith("image/"):
        return mime_type

    if raw_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if raw_bytes.startswith(b"RIFF") and raw_bytes[8:12] == b"WEBP":
        return "image/webp"

    suffix = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


_MIN_DIMENSION_PX = 100       # both width and height must meet this
_BLUR_VARIANCE_THRESHOLD = 80  # Laplacian variance below this → too blurry


def _check_image_quality(raw_bytes: bytes) -> str | None:
    """
    Returns a rejection_reason string if the image fails quality checks,
    or None if the image is acceptable.
    Checks: minimum pixel dimensions and blurriness (Laplacian variance).
    """
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        w, h = img.size
        if w < _MIN_DIMENSION_PX or h < _MIN_DIMENSION_PX:
            return f"image_too_small (got {w}x{h}, minimum {_MIN_DIMENSION_PX}x{_MIN_DIMENSION_PX})"

        arr = np.array(img.convert("L"), dtype=np.float32)
        laplacian = (
            np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0) +
            np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1) - 4 * arr
        )
        variance = float(laplacian.var())
        if variance < _BLUR_VARIANCE_THRESHOLD:
            return f"image_too_blurry (variance={variance:.1f}, threshold={_BLUR_VARIANCE_THRESHOLD})"
    except Exception as exc:
        logger.warning("Image quality check failed to run: %s", exc)

    return None


def analyze_image(image_path: str, checks: list[dict]) -> dict:
    with start_span(
        "vision.analyze_image",
        {"vision.image_path": image_path, "vision.check_count": len(checks)},
    ) as span:
        all_checks = list(checks)
        if not all_checks:
            all_checks = [
                VisualCheck(
                    check_description="General product condition",
                    expected_finding="Visible damage, defect, or malfunction indicator",
                ).model_dump()
            ]

        try:
            raw_bytes = read_image(image_path)
            if not raw_bytes:
                raise ValueError("Image file is empty or unreadable")
            b64 = base64.b64encode(raw_bytes).decode()
            mime_type = _detect_mime_type(image_path, raw_bytes)
        except Exception as exc:
            span.record_exception(exc)
            logger.error("Vision: failed to read image %s: %s", image_path, exc)
            return {
                "damage_report": {"error": str(exc)},
                "checks": [],
            }

        rejection_reason = _check_image_quality(raw_bytes)
        if rejection_reason:
            logger.warning("Vision: image quality gate rejected %s — %s", image_path, rejection_reason)
            span.set_attribute("vision.quality_rejection", rejection_reason)
            return {
                "damage_report": {"image_quality": "poor", "rejection_reason": rejection_reason},
                "checks": [],
            }

        if not mime_type.startswith("image/"):
            logger.error("Vision: unsupported image type for %s (%s)", image_path, mime_type)
            return {
                "damage_report": {"error": f"Unsupported image type: {mime_type}"},
                "checks": [],
            }

        user_content = [
            {
                "type": "text",
                "text": f"Please perform the following visual checks on this image:\n\n{json.dumps(all_checks, indent=2)}",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            },
        ]
        messages = [
            {"role": "system", "content": get_prompt("vision_service")},
            {"role": "user", "content": user_content},
        ]

        logger.info(
            "Vision: sending multimodal request for %s with mime_type=%s and %d checks",
            image_path,
            mime_type,
            len(all_checks),
        )
        raw = vision_llm(messages)
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            span.record_exception(exc)
            logger.warning(
                "Vision analysis JSON parse failed: %s | raw response (first 500 chars): %s",
                exc,
                raw[:500] if isinstance(raw, str) else repr(raw)[:500],
            )
            data = {
                "image_quality": "poor",
                "image_type": "other",
                "overall_confidence": 0.0,
                "checks": [],
            }

        damage_report = {
            "image_quality": data.get("image_quality", ""),
            "image_type": data.get("image_type", ""),
            "overall_confidence": data.get("overall_confidence", 0.0),
            "extracted_purchase_date": data.get("extracted_purchase_date"),
            "extracted_serial_number": data.get("extracted_serial_number"),
        }
        span.set_attribute("vision.image_quality", damage_report.get("image_quality", ""))
        span.set_attribute("vision.overall_confidence", damage_report.get("overall_confidence", 0.0))
        return {
            "damage_report": damage_report,
            "checks": data.get("checks", []),
        }
