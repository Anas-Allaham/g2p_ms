"""
POST /api/v1/g2p — English text to ARPAbet with trust/heteronym/OOV metadata.

Stateless. Same authoritative G2P path the analysis endpoints score against,
so the reading guide and trust flags match what a learner will later be graded
on.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..config import API_VERSION
from ..envelopes import success
from ..schemas import G2PEnvelope, G2PRequest
from ..security import require_service_auth

router = APIRouter(prefix=f"/api/{API_VERSION}", tags=["g2p"], dependencies=[Depends(require_service_auth)])


@router.post("/g2p", response_model=G2PEnvelope)
def convert(payload: G2PRequest, request: Request):
    from src.core.g2p.tokenization import ipa_reading_guide
    from ..arpabet import to_public_arpabet
    from ..reference_validation import resolve_supported_reference

    text = payload.text.strip()
    resolution = resolve_supported_reference(text)
    internal_data = {
        "text": text,
        "ipa": resolution.ipa,
        "guide": ipa_reading_guide(resolution.ipa),
        **resolution.to_dict(),
    }
    return success(to_public_arpabet(internal_data), request)
