"""Optional external fallback when the offline audio cleaner fails.

Cleanvoice is never the primary path. The caller invokes this module only for
eligible local model/processing failures. Pronunciation timing and content are
preserved by disabling every cutting/editing feature.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

try:
    from cleanvoice import Cleanvoice
except Exception:  # pragma: no cover - optional dependency
    Cleanvoice = None


class CleanvoiceProcessingError(RuntimeError):
    """A user-safe external-fallback failure."""


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def cleanvoice_fallback_enabled() -> bool:
    """Enable with a key unless explicitly disabled.

    ``CLEANVOICE_ENABLED`` remains supported for existing deployments, while
    the clearer ``CLEANVOICE_FALLBACK_ENABLED`` takes precedence.
    """
    has_key = bool(os.environ.get("CLEANVOICE_API_KEY", "").strip())
    legacy_default = _env_flag("CLEANVOICE_ENABLED", has_key)
    return _env_flag("CLEANVOICE_FALLBACK_ENABLED", legacy_default)


def cleanvoice_fallback_configured() -> bool:
    return cleanvoice_fallback_enabled() and bool(
        os.environ.get("CLEANVOICE_API_KEY", "").strip()
    )


def _cleanvoice_class():
    global Cleanvoice
    if Cleanvoice is None:
        try:
            from cleanvoice import Cleanvoice as cleanvoice_class
        except Exception:
            return None
        Cleanvoice = cleanvoice_class
    return Cleanvoice


def cleanvoice_sdk_available() -> bool:
    return _cleanvoice_class() is not None


def enhance_recording(input_path: Path, output_path: Path) -> Path:
    """Upload one normalized WAV and download pronunciation-safe enhancement."""
    api_key = os.environ.get("CLEANVOICE_API_KEY", "").strip()
    if not cleanvoice_fallback_enabled():
        raise CleanvoiceProcessingError("Cleanvoice fallback is disabled.")
    if not api_key:
        raise CleanvoiceProcessingError("CLEANVOICE_API_KEY is missing.")
    cleanvoice_class = _cleanvoice_class()
    if cleanvoice_class is None:
        raise CleanvoiceProcessingError("The Cleanvoice SDK is not installed.")

    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        timeout = max(10, int(float(os.environ.get("CLEANVOICE_HTTP_TIMEOUT", "120"))))
    except ValueError:
        timeout = 120

    client: Optional[Any] = None
    try:
        client = cleanvoice_class(api_key=api_key, timeout=timeout)
        client.process(
            str(source),
            remove_noise=True,
            normalize=True,
            studio_sound=_env_flag("CLEANVOICE_STUDIO_SOUND", False),
            fillers=False,
            long_silences=False,
            mouth_sounds=False,
            breath=False,
            stutters=False,
            hesitations=False,
            transcription=False,
            summarize=False,
            social_content=False,
            export_format="wav",
            output_path=str(output),
        )
    except Exception as exc:
        # SDK errors may contain signed URLs or provider details.
        raise CleanvoiceProcessingError(
            "Cleanvoice could not enhance this recording."
        ) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    if not output.is_file() or output.stat().st_size <= 44:
        raise CleanvoiceProcessingError("Cleanvoice returned no valid audio.")
    return output
