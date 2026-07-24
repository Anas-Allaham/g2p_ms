"""
POST /api/v1/g2p — English text to IPA with trust/heteronym/OOV metadata.

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
    from src.core.g2p.g2p_service import g2p_convert_with_metadata
    from src.core.g2p.tokenization import ipa_reading_guide

    text = payload.text.strip()
    resolution = g2p_convert_with_metadata(text)
    data = {
        "text": text,
        "ipa": resolution.ipa,
        "guide": ipa_reading_guide(resolution.ipa),
        **resolution.to_dict(),
    }
    return success(data, request)
