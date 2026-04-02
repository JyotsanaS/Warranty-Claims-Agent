"""
Tests for the image quality gate in vision_service._check_image_quality.
All test images are generated in-memory — no external files needed.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from agent.services.vision_service import _check_image_quality


def _make_image_bytes(width: int, height: int, noise: float = 0.5) -> bytes:
    """
    Create a JPEG image in memory.
    noise controls how much random variation is added — high noise = sharp,
    low noise (uniform colour) = blurry when passed through the Laplacian.
    """
    rng = np.random.default_rng(42)
    if noise > 0:
        data = (rng.random((height, width, 3)) * 255 * noise).astype(np.uint8)
    else:
        data = np.full((height, width, 3), 128, dtype=np.uint8)
    img = Image.fromarray(data, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_good_image_passes():
    """A sharp, adequately sized image should pass (return None)."""
    raw = _make_image_bytes(width=300, height=300, noise=1.0)
    assert _check_image_quality(raw) is None


def test_too_small_width_rejected():
    """Image narrower than the minimum dimension should be rejected."""
    raw = _make_image_bytes(width=50, height=300, noise=1.0)
    reason = _check_image_quality(raw)
    assert reason is not None
    assert "image_too_small" in reason


def test_too_small_height_rejected():
    """Image shorter than the minimum dimension should be rejected."""
    raw = _make_image_bytes(width=300, height=50, noise=1.0)
    reason = _check_image_quality(raw)
    assert reason is not None
    assert "image_too_small" in reason


def test_too_small_both_dimensions_rejected():
    """Tiny image (both dims below threshold) should be rejected."""
    raw = _make_image_bytes(width=32, height=32, noise=1.0)
    reason = _check_image_quality(raw)
    assert reason is not None
    assert "image_too_small" in reason


def test_blurry_image_rejected():
    """A uniform (zero-noise) image has near-zero Laplacian variance — too blurry."""
    raw = _make_image_bytes(width=300, height=300, noise=0.0)
    reason = _check_image_quality(raw)
    assert reason is not None
    assert "image_too_blurry" in reason


def test_invalid_bytes_does_not_raise():
    """Garbage bytes should not raise — quality check must fail open."""
    reason = _check_image_quality(b"not an image at all")
    assert reason is None  # fails open, warns in log


def test_empty_bytes_does_not_raise():
    """Empty bytes should not raise — fails open."""
    reason = _check_image_quality(b"")
    assert reason is None
