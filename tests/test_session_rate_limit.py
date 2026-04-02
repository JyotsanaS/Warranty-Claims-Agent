from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.routers.sessions import (
    SESSION_TURN_LIMIT,
    SESSION_TURN_WINDOW_SECONDS,
    _enforce_message_rate_limit,
    _release_active_request,
)


def make_session(**overrides) -> dict:
    base = {
        "request_timestamps": [],
        "active_request": False,
    }
    base.update(overrides)
    return base


def test_allows_requests_up_to_limit():
    session = make_session()
    now = 1000.0

    for i in range(SESSION_TURN_LIMIT):
        _enforce_message_rate_limit(session, now=now + i)
        assert session["active_request"] is True
        _release_active_request(session)

    assert len(session["request_timestamps"]) == SESSION_TURN_LIMIT


def test_rejects_request_when_limit_is_exceeded():
    session = make_session(
        request_timestamps=[1000.0 + i for i in range(SESSION_TURN_LIMIT)],
        active_request=False,
    )

    with pytest.raises(HTTPException) as exc:
        _enforce_message_rate_limit(session, now=1100.0)

    assert exc.value.status_code == 429
    assert "15 turns per 5 minutes" in exc.value.detail


def test_prunes_old_timestamps_outside_window():
    session = make_session(
        request_timestamps=[1000.0, 1001.0, 1002.0],
        active_request=False,
    )

    _enforce_message_rate_limit(session, now=1000.0 + SESSION_TURN_WINDOW_SECONDS + 1)

    assert session["request_timestamps"] == [
        1002.0,
        1000.0 + SESSION_TURN_WINDOW_SECONDS + 1,
    ]
    assert session["active_request"] is True


def test_rejects_concurrent_request_for_same_session():
    session = make_session(request_timestamps=[1000.0], active_request=True)

    with pytest.raises(HTTPException) as exc:
        _enforce_message_rate_limit(session, now=1001.0)

    assert exc.value.status_code == 429
    assert "already in progress" in exc.value.detail


def test_release_active_request_clears_inflight_flag():
    session = make_session(active_request=True)
    _release_active_request(session)
    assert session["active_request"] is False
