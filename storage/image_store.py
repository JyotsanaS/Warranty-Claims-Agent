from datetime import datetime
from pathlib import Path

from app.config import settings


def save_image(session_id: str, image_type: str, file_bytes: bytes, extension: str) -> str:
    """Save image bytes under results/<session_id>/images/. Returns absolute path."""
    upload_dir = Path(settings.results_dir) / session_id / "images"
    upload_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{image_type}_{timestamp}.{extension}"
    filepath = upload_dir / filename
    filepath.write_bytes(file_bytes)
    return str(filepath.resolve())


def read_image(path: str) -> bytes:
    """Read image bytes from local filesystem."""
    return Path(path).read_bytes()
