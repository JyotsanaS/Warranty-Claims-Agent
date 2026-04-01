from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings


def _session_dir(session_id: str) -> Path:
    path = Path(settings.results_dir) / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_session_json(session_id: str, filename: str, payload: Any) -> str:
    """Persist a JSON artifact under results/<session_id>/ and return its path."""
    filepath = _session_dir(session_id) / filename
    filepath.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(filepath.resolve())
