"""Validate reference G2P results before they enter public API workflows."""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from .errors import ValidationError

if TYPE_CHECKING:
    from src.core.g2p.g2p_service import ReferenceG2PResult


_REFERENCE_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
_SMART_APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})


def normalize_reference_text(text: str) -> str:
    """Return the exact punctuation-free English words used for scoring.

    Internal apostrophes are retained for contractions. All other punctuation
    becomes a word boundary, matching the G2P's existing word extraction while
    giving clients a clean word-by-word display string.
    """
    normalized = unicodedata.normalize("NFKC", str(text or "")).translate(
        _SMART_APOSTROPHES
    )
    return " ".join(_REFERENCE_WORD_RE.findall(normalized))


def resolve_supported_reference(text: str) -> "ReferenceG2PResult":
    """Resolve text and reject raw-spelling OOV fallbacks explicitly.

    Dictionary OOV fallback text is not IPA and therefore cannot be converted
    to ARPAbet or scored as a pronunciation reference. Returning a structured
    422 is safer than guessing phones or allowing a boundary conversion 500.
    """
    from src.core.g2p.g2p_service import g2p_convert_with_metadata

    normalized_text = normalize_reference_text(text)
    if not normalized_text:
        raise ValidationError(
            "The 'text' field must contain at least one English word.",
            details={"field": "text"},
        )

    resolution = g2p_convert_with_metadata(normalized_text)
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
