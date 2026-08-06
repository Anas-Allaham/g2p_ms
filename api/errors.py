"""
Typed application errors and their mapping to the service's error envelope.

Every failure that reaches the client is one of these (or is normalized into
one by the handlers registered in ``main.py``), so the response body always has
the same shape: ``{"error": {"code", "message", "details"}, "meta": {...}}``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ServiceError(Exception):
    """Base class for errors that map cleanly onto the error envelope.

    ``code`` is a stable, machine-readable string the Django core can branch
    on; ``message`` is human-readable; ``details`` is optional structured
    context (never audio, never secrets)."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}


class AuthError(ServiceError):
    status_code = 401
    code = "unauthorized"


class AuthNotConfiguredError(ServiceError):
    status_code = 503
    code = "auth_not_configured"


class ValidationError(ServiceError):
    status_code = 422
    code = "validation_error"


class NotFoundError(ServiceError):
    status_code = 404
    code = "not_found"


class PayloadTooLargeError(ServiceError):
    status_code = 413
    code = "payload_too_large"


class UnsupportedMediaError(ServiceError):
    status_code = 415
    code = "unsupported_media_type"


class AudioDecodeError(ServiceError):
    status_code = 400
    code = "audio_decode_unavailable"


class ConflictError(ServiceError):
    status_code = 409
    code = "conflict"


class ModelUnavailableError(ServiceError):
    status_code = 503
    code = "model_unavailable"
