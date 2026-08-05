"""Validate reference G2P results before they enter public API workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import ValidationError

if TYPE_CHECKING:
    from src.core.g2p.g2p_service import ReferenceG2PResult


def resolve_supported_reference(text: str) -> "ReferenceG2PResult":
    """Resolve text and reject raw-spelling OOV fallbacks explicitly.

    Dictionary OOV fallback text is not IPA and therefore cannot be converted
    to ARPAbet or scored as a pronunciation reference. Returning a structured
    422 is safer than guessing phones or allowing a boundary conversion 500.
    """
    from src.core.g2p.g2p_service import g2p_convert_with_metadata

    resolution = g2p_convert_with_metadata(text)
    if resolution.oov_words:
        raise ValidationError(
            "No trusted pronunciation is available for one or more words.",
            code="g2p_oov",
            details={
                "oov_words": list(resolution.oov_words),
                "g2p_mode": resolution.g2p_mode,
            },
        )
    return resolution
