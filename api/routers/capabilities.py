"""
GET /api/v1/capabilities — authenticated, detailed model/feature information.

This is the richer counterpart to /health/ready: it exposes the same trust and
inventory diagnostics the Flask /health route did, plus the service's
processing limits and scientific-limitation labels, so the Django core can
present accurate capability/limitation information to end users.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..config import API_VERSION, settings
from ..envelopes import success
from ..security import require_service_auth

router = APIRouter(prefix=f"/api/{API_VERSION}", tags=["capabilities"], dependencies=[Depends(require_service_auth)])


@router.get("/capabilities")
def capabilities(request: Request):
    from src.core.persistence import db
    from src.core.audio.cleanvoice_service import (
        cleanvoice_configured,
        cleanvoice_enabled,
        cleanvoice_sdk_available,
        cleanvoice_strict,
    )
    from src.core.g2p.content import llm_available
    from src.core.g2p.g2p_service import (
        HETERONYMS_PATH,
        IPA_DICT_PATH,
        get_g2p_mode,
        heteronym_resolution_active,
        validate_heteronym_lexicon,
    )
    from src.core.g2p.phoneme_vectors_professional import (
        canonical_inventory,
        panphon_available,
        scoring_engine,
        scoring_trusted,
        validate_g2p_inventory,
        validate_panphon_inventory,
    )

    from .. import acoustic
    from ..arpabet import to_public_arpabet

    inventory_report = validate_g2p_inventory(db.get_all_bank_phonemes())
    panphon_report = validate_panphon_inventory()
    heteronym_report = validate_heteronym_lexicon()

    data = {
        "device": acoustic.device(),
        "model": {
            "config_present": acoustic.model_config_present(),
            "weight_present": acoustic.model_weight_present(),
            "loaded": acoustic.model_loaded(),
            "architecture": "wav2vec2-ctc-phoneme",
        },
        "scoring": {
            "engine": scoring_engine(),
            "trusted": scoring_trusted(),
            "panphon_available": panphon_available(),
            "panphon_inventory_ok": panphon_report["ok"],
            "panphon_inventory_failures": panphon_report["failures"],
            "canonical_inventory_size": len(canonical_inventory()),
            "canonical_inventory_ok": inventory_report["ok"],
            "canonical_inventory_unsupported": inventory_report["unsupported"],
        },
        "g2p": {
            "available": HETERONYMS_PATH.exists() and IPA_DICT_PATH.exists(),
            "alphabet": "arpabet",
            "internal_alphabet": "ipa",
            "stress": "not_assessed",
            "mode": get_g2p_mode(),
            "heteronym_resolution_active": heteronym_resolution_active(),
            "heteronym_entries_checked": heteronym_report["checked"],
            "heteronym_unsupported_contrasts": heteronym_report["unsupported_contrasts"],
        },
        "exercise_bank": {
            "count": db.count_exercise_bank(),
            "populated": db.count_exercise_bank() > 0,
            "llm_generation_available": llm_available(),
        },
        "preprocessing": {
            "cleanvoice_enabled": cleanvoice_enabled(),
            "cleanvoice_configured": cleanvoice_configured(),
            "cleanvoice_sdk_available": cleanvoice_sdk_available(),
            "cleanvoice_strict": cleanvoice_strict(),
            "audio_retention_enabled": settings.retain_audio,
        },
        "limits": {
            "max_audio_bytes": settings.max_audio_bytes,
            "max_audio_seconds": settings.max_audio_seconds,
            "language": "en",
            "processing": "synchronous",
        },
        "scientific_limitations": {
            "provisional": True,
            "note": (
                "Scores derive from PanPhon articulatory-distance soft evidence, "
                "not calibrated GOP probabilities, and are NOT a CEFR level. "
                "Mastery is a per-phoneme Beta posterior with half-life decay; "
                "levels are reported with an evidence-aware credible interval."
            ),
        },
    }
    return success(to_public_arpabet(data), request)
