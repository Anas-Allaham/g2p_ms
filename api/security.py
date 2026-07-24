"""
Bearer-token authentication for /api/v1.

The service trusts a single shared secret (``SERVICE_API_KEY``) presented by
the Django core as ``Authorization: Bearer <key>``. The comparison is
constant-time so a timing side-channel cannot leak the key. There is no user
auth here — that is the Django core's responsibility; this service only sees
opaque subject UUIDs.
"""

from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings
from .errors import AuthError, AuthNotConfiguredError

# Declaring the scheme (rather than parsing the header by hand) registers a
# ``bearerAuth`` security scheme in the OpenAPI document, which is what makes
# the "Authorize" button appear in Swagger UI / ReDoc. ``auto_error=False`` so
# a missing/malformed header flows into our own checks below and returns the
# service's error envelope (503 when unconfigured, 401 otherwise) instead of
# HTTPBearer's bare 403.
_bearer_scheme = HTTPBearer(
    scheme_name="ServiceApiKey",
    description="Shared service secret presented as `Authorization: Bearer <SERVICE_API_KEY>`.",
    auto_error=False,
)


def require_service_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> None:
    """FastAPI dependency: reject unless a valid service bearer token is
    presented. Raises ``AuthNotConfiguredError`` (503) when the deployment has
    no key set, so a misconfiguration fails closed rather than open."""
    if not settings.auth_configured:
        raise AuthNotConfiguredError(
            "SERVICE_API_KEY is not configured on this deployment."
        )
    token = credentials.credentials.strip() if credentials else None
    if not token:
        raise AuthError("Missing or malformed Authorization header. Expected a Bearer token.")
    # Constant-time comparison; compare_digest handles unequal lengths safely.
    if not hmac.compare_digest(token, settings.service_api_key):
        raise AuthError("Invalid service credentials.")
