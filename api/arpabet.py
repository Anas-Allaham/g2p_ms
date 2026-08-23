"""Translate internal IPA domain payloads at the public API boundary.

NeMo, the POS-aware reference pipeline, Wav2Vec2, PanPhon, alignment,
mastery, and persistence all use IPA.  Django and frontend clients see
stress-free ARPAbet.  Keeping this translation in one module prevents the
public alphabet from leaking into the phonetic/scoring domain.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from src.core.g2p.phoneme_alphabet import (
    ARPABET_TO_IPA,
    IPAInputFormat,
    canonicalize_arpabet_phoneme,
    ipa_to_arpabet,
    ipa_word_to_arpabet_tokens,
)
from src.core.g2p.phoneme_vectors_professional import (
    acoustic_model_equivalent,
    canonicalize_phoneme,
)


_SEQUENCE_FORMATS = {
    "ipa": IPAInputFormat.FORMATTED_REFERENCE,
    "reference_ipa": IPAInputFormat.FORMATTED_REFERENCE,
    "predicted_ipa": IPAInputFormat.RAW_CTC,
}
_RENAMED_KEYS = {
    "ipa": "arpabet",
    "reference_ipa": "reference_arpabet",
    "predicted_ipa": "predicted_arpabet",
}
_SINGLE_PHONEME_KEYS = frozenset({
    "phoneme",
    "symbol",
    "expected",
    "spoken",
    "observed_by_model",
})
_PHONEME_LIST_KEYS = frozenset({
    "target_phonemes",
    "unknown_phonemes",
    "strong_phonemes",
    "uncovered",
    "panphon_inventory_failures",
    "canonical_inventory_unsupported",
})


def ipa_phoneme_to_arpabet(
    phoneme: Any,
    *,
    strict: bool = True,
    allow_ctc_fallbacks: bool = True,
) -> Any:
    """Convert one internal IPA phoneme while preserving gaps and nulls."""
    if phoneme is None or phoneme == "-":
        return phoneme
    tokens = ipa_word_to_arpabet_tokens(
        str(phoneme),
        strict=strict,
        allow_ctc_fallbacks=allow_ctc_fallbacks,
    )
    return " ".join(tokens)


def _convert_confusion_hint(value: str) -> str:
    if " vs " not in value:
        return value
    expected, spoken = value.split(" vs ", 1)
    return (
        f"{ipa_phoneme_to_arpabet(expected)} vs "
        f"{ipa_phoneme_to_arpabet(spoken)}"
    )


def _display_model_equivalent_match(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Make an accepted collapsed-label match intuitive at the public boundary.

    The internal alignment keeps the checkpoint's real observation for audit
    and persistence.  Public clients see the accepted reference phone in
    ``spoken`` and can inspect the unchanged acoustic label separately in
    ``observed_by_model``.
    """
    displayed = dict(value)
    expected = displayed.get("expected")
    spoken = displayed.get("spoken")
    if (
        displayed.get("result") == "correct"
        and isinstance(expected, str)
        and isinstance(spoken, str)
        and expected not in {"", "-"}
        and spoken not in {"", "-"}
        and canonicalize_phoneme(expected) != canonicalize_phoneme(spoken)
        and acoustic_model_equivalent(expected, spoken)
    ):
        displayed["observed_by_model"] = spoken
        displayed["spoken"] = expected
    return displayed


def to_public_arpabet(value: Any, parent_key: str | None = None) -> Any:
    """Return a recursively converted copy suitable for an API response."""
    if isinstance(value, Mapping):
        value = _display_model_equivalent_match(value)
        converted: Dict[str, Any] = {}
        for key, child in value.items():
            converted[_RENAMED_KEYS.get(key, key)] = to_public_arpabet(child, key)
        return converted

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        if parent_key in _PHONEME_LIST_KEYS:
            return [ipa_phoneme_to_arpabet(item) for item in value]
        return [to_public_arpabet(item, parent_key) for item in value]

    if isinstance(value, str):
        if parent_key in _SEQUENCE_FORMATS:
            input_format = _SEQUENCE_FORMATS[parent_key]
            return ipa_to_arpabet(
                value,
                input_format=input_format,
                strict=input_format is IPAInputFormat.FORMATTED_REFERENCE,
            )
        if parent_key in _SINGLE_PHONEME_KEYS:
            return ipa_phoneme_to_arpabet(
                value,
                strict=parent_key != "spoken",
            )
        if parent_key == "confusion_hint":
            return _convert_confusion_hint(value)

    return value


def metrics_to_internal_ipa(metrics: Mapping[str, float] | None) -> Dict[str, float]:
    """Convert a validated public ARPAbet metric map back to internal IPA."""
    converted: Dict[str, float] = {}
    for raw_phoneme, score in (metrics or {}).items():
        arpabet = canonicalize_arpabet_phoneme(str(raw_phoneme))
        ipa = ARPABET_TO_IPA.get(arpabet)
        if ipa is not None:
            converted[ipa] = float(score)
    return converted
