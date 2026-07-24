"""
Response envelope + request-id helpers.

Every success is ``{"data": <payload>, "meta": {...}}`` and every error is
``{"error": {...}, "meta": {...}}``. ``meta`` always carries the API version
and the request id so the Django core can correlate logs across services.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import Request

from .config import API_VERSION, SERVICE_NAME

REQUEST_ID_HEADER = "X-Request-ID"


def new_request_id() -> str:
    return uuid.uuid4().hex


def request_id_of(request: Request) -> str:
    """The request id assigned by the middleware (falls back to a fresh one)."""
    rid = getattr(request.state, "request_id", None)
    return rid or new_request_id()


def build_meta(request: Optional[Request] = None, request_id: Optional[str] = None) -> Dict[str, Any]:
    rid = request_id or (request_id_of(request) if request is not None else new_request_id())
    return {
        "service": SERVICE_NAME,
        "api_version": API_VERSION,
        "request_id": rid,
    }


def success(data: Any, request: Optional[Request] = None, request_id: Optional[str] = None) -> Dict[str, Any]:
    return {"data": data, "meta": build_meta(request, request_id)}


def error_body(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request: Optional[Request] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "error": {"code": code, "message": message, "details": details or {}},
        "meta": build_meta(request, request_id),
    }
