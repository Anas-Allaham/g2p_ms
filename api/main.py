"""
FastAPI application factory for the pronunciation AI microservice.

Wires: request-id middleware, a single ``{data, meta}`` / ``{error, meta}``
envelope enforced through consistent exception handlers, optional CORS (off by
default), the /api/v1 routers, and a startup bootstrap that seeds the exercise
bank on first run. OpenAPI is generated automatically at /openapi.json.
"""

from __future__ import annotations

import contextlib

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import API_VERSION, SERVICE_NAME, settings
from .envelopes import REQUEST_ID_HEADER, error_body, new_request_id
from .errors import ServiceError
from .routers import analyses, audio, capabilities, exercises, g2p, health, subjects

from src.core.persistence.db import IdempotencyConflict as _IdempotencyConflict


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    from .bootstrap import bootstrap

    bootstrap()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Pronunciation AI Service",
        version=API_VERSION,
        description=(
            "Anonymous, subject-scoped pronunciation analysis, mastery, "
            "assessment, and adaptive exercises. Called server-to-server by the "
            "Django core; no personal data is stored here."
        ),
        lifespan=lifespan,
    )

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        rid = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        request.state.request_id = rid
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        return response

    _register_error_handlers(app)

    for module in (health, capabilities, g2p, subjects, analyses, audio, exercises):
        app.include_router(module.router)

    @app.get("/", tags=["meta"])
    def root(request: Request):
        return {
            "data": {"service": SERVICE_NAME, "api_version": API_VERSION, "docs": "/docs"},
            "meta": {"service": SERVICE_NAME, "api_version": API_VERSION, "request_id": request.state.request_id},
        }

    return app


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def handle_service_error(request: Request, exc: ServiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message, exc.details, request=request),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=error_body(
                "validation_error",
                "Request validation failed.",
                details={"errors": _safe_errors(exc.errors())},
                request=request,
            ),
        )

    @app.exception_handler(_IdempotencyConflict)
    async def handle_idempotency_conflict(request: Request, exc: _IdempotencyConflict):
        return JSONResponse(
            status_code=409,
            content=error_body(
                "idempotency_in_flight",
                "A request with this Idempotency-Key is still being processed.",
                request=request,
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(request: Request, exc: StarletteHTTPException):
        code = {
            400: "bad_request", 401: "unauthorized", 403: "forbidden",
            404: "not_found", 405: "method_not_allowed", 409: "conflict",
            413: "payload_too_large", 415: "unsupported_media_type",
        }.get(exc.status_code, "http_error")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(code, message, request=request),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        # Never leak internals (tracebacks, storage URLs, audio paths).
        import traceback

        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content=error_body("internal_error", "An unexpected error occurred.", request=request),
        )


def _safe_errors(errors):
    """Strip non-JSON-serializable context (e.g. exception objects) from
    pydantic validation errors before returning them."""
    cleaned = []
    for err in errors:
        cleaned.append({
            "loc": [str(part) for part in err.get("loc", [])],
            "msg": err.get("msg", ""),
            "type": err.get("type", ""),
        })
    return cleaned


app = create_app()
