"""
Shared FastAPI dependencies: subject-id validation, Idempotency-Key handling,
and streamed, size-capped upload persistence.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import Header, Path as PathParam, UploadFile
from typing import Optional

from .config import settings
from .errors import PayloadTooLargeError, UnsupportedMediaError, ValidationError

# Opaque subject ids are Django-generated. Accept UUIDs and other opaque,
# URL-safe tokens, but bound the length and character set so a bad path can
# never reach SQL as something unexpected.
_SUBJECT_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

_ACCEPTED_AUDIO_HINTS = ("audio", "video/webm", "video/mp4", "application/octet-stream")


def valid_subject_id(subject_id: str = PathParam(..., description="Opaque Django subject UUID.")) -> str:
    subject_id = subject_id.strip()
    if not _SUBJECT_ID_RE.match(subject_id):
        raise ValidationError(
            "subject_id must be an opaque URL-safe token (<=128 chars).",
            details={"field": "subject_id"},
        )
    return subject_id


def require_idempotency_key(
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
) -> str:
    """Stateful writes require an Idempotency-Key so a Django retry cannot
    duplicate mastery evidence or an assignment."""
    if not idempotency_key or not idempotency_key.strip():
        raise ValidationError(
            "This operation requires an 'Idempotency-Key' header.",
            details={"header": "Idempotency-Key"},
        )
    key = idempotency_key.strip()
    if len(key) > 200:
        raise ValidationError("Idempotency-Key is too long.", details={"header": "Idempotency-Key"})
    return key


def save_upload(upload: UploadFile, dest: Path) -> None:
    """Stream an UploadFile to ``dest`` in chunks, enforcing the configured
    byte cap without ever loading the whole recording into memory. FastAPI's
    UploadFile is backed by a spooled temp file, so this stays memory-bounded.
    """
    mimetype = (upload.content_type or "").lower()
    if mimetype and not any(hint in mimetype for hint in _ACCEPTED_AUDIO_HINTS):
        raise UnsupportedMediaError(
            f"Unsupported audio content type: {mimetype!r}.",
            details={"content_type": mimetype},
        )
    max_bytes = settings.max_audio_bytes
    written = 0
    upload.file.seek(0)
    with dest.open("wb") as out:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                out.close()
                try:
                    dest.unlink()
                except Exception:
                    pass
                raise PayloadTooLargeError(
                    f"Recording exceeds the {max_bytes} byte limit.",
                    details={"max_bytes": max_bytes},
                )
            out.write(chunk)
    if written == 0:
        try:
            dest.unlink()
        except Exception:
            pass
        raise ValidationError("The uploaded audio file is empty.", details={"field": "audio"})
