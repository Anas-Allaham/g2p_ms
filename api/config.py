"""
Runtime configuration for the pronunciation AI microservice.

Read once from the environment. Kept dependency-light (no pydantic-settings)
so importing config never drags in the web stack — scripts and tests can read
it too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Load a local .env if python-dotenv is available. In Modal the environment is
# supplied by Secrets, so this is a no-op there.
try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

API_VERSION = "v1"
SERVICE_NAME = "pronunciation-ai-service"


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _optional_float(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class Settings:
    # Auth. Every /api/v1 route requires Bearer <service_api_key>. If unset the
    # service still boots (health/readiness work) but authenticated routes
    # return 503 so a misconfigured deployment fails loudly rather than open.
    service_api_key: str = field(default_factory=lambda: os.environ.get("SERVICE_API_KEY", "").strip())

    # Synchronous analysis limits (English sentence recordings).
    max_audio_bytes: int = field(default_factory=lambda: _int("MAX_AUDIO_BYTES", 20 * 1024 * 1024))
    max_audio_seconds: float = field(default_factory=lambda: float(_int("MAX_AUDIO_SECONDS", 60)))

    # Browser CORS is OFF by default: this is a server-to-server service called
    # by the Django core, not by browsers. Set CORS_ALLOW_ORIGINS to opt in.
    cors_allow_origins: List[str] = field(
        default_factory=lambda: [
            o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()
        ]
    )

    # Recordings are private + temporary. Retention is never enabled for a
    # shared deployment; exposed only for local debugging.
    retain_audio: bool = field(default_factory=lambda: _flag("RETAIN_AUDIO", False))

    # Offline exercise-audio cleaning. Models remain lazy even when enabled,
    # so health checks, migrations/bootstrap, and unrelated requests stay light.
    audio_cleaning_enabled: bool = field(
        default_factory=lambda: _flag("AUDIO_CLEANING_ENABLED", False)
    )
    audio_cleaning_use_gpu: bool = field(
        default_factory=lambda: _flag("AUDIO_CLEANING_USE_GPU", False)
    )
    audio_cleaning_min_duration_seconds: float = field(
        default_factory=lambda: _float("AUDIO_CLEANING_MIN_DURATION_SECONDS", 0.5)
    )
    audio_cleaning_min_speech_seconds: float = field(
        default_factory=lambda: _float("AUDIO_CLEANING_MIN_SPEECH_SECONDS", 0.3)
    )
    audio_cleaning_max_clipping_ratio: float = field(
        default_factory=lambda: _float("AUDIO_CLEANING_MAX_CLIPPING_RATIO", 0.01)
    )
    audio_cleaning_min_speech_ratio: float | None = field(
        default_factory=lambda: _optional_float("AUDIO_CLEANING_MIN_SPEECH_RATIO")
    )
    audio_cleaning_keep_intermediate_files: bool = field(
        default_factory=lambda: _flag("AUDIO_CLEANING_KEEP_INTERMEDIATE_FILES", False)
    )
    audio_cleaning_timeout_seconds: float = field(
        default_factory=lambda: _float("AUDIO_CLEANING_TIMEOUT_SECONDS", 120.0)
    )
    audio_cleaning_output_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "AUDIO_CLEANING_OUTPUT_DIR",
                str(Path(__file__).resolve().parent.parent / "uploads"),
            )
        )
    )

    @property
    def auth_configured(self) -> bool:
        return bool(self.service_api_key)


settings = Settings()
