"""
Anonymous subject lifecycle + read models.

  * PUT    /api/v1/subjects/{subject_id}              — idempotent create
  * DELETE /api/v1/subjects/{subject_id}              — cascade delete
  * GET    /api/v1/subjects/{subject_id}/assessment   — evidence-aware level
  * GET    /api/v1/subjects/{subject_id}/gaps         — ranked weak phonemes
  * GET    /api/v1/subjects/{subject_id}/attempts     — cursor-paginated history

A subject is an opaque Django UUID with no personal data. Reads on an unknown
subject return empty, evidence-honest payloads rather than 404s, so the Django
core does not need a create round-trip before its first read.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, Response, status

from ..config import API_VERSION
from ..dependencies import valid_subject_id
from ..envelopes import success
from ..schemas import (
    AssessmentEnvelope,
    AttemptsPageEnvelope,
    DeletedSubjectEnvelope,
    GapsEnvelope,
    SubjectEnvelope,
)
from ..security import require_service_auth

router = APIRouter(
    prefix=f"/api/{API_VERSION}/subjects",
    tags=["subjects"],
    dependencies=[Depends(require_service_auth)],
)


@router.put("/{subject_id}", response_model=SubjectEnvelope)
def put_subject(request: Request, response: Response, subject_id: str = Depends(valid_subject_id)):
    from src.core.persistence import db

    existed = db.get_subject(subject_id) is not None
    row = db.get_or_create_subject(subject_id)
    response.status_code = status.HTTP_200_OK if existed else status.HTTP_201_CREATED
    return success(
        {"subject_id": row["subject_id"], "created_at": row["created_at"], "created": not existed},
        request,
    )


@router.delete("/{subject_id}", response_model=DeletedSubjectEnvelope)
def delete_subject(request: Request, subject_id: str = Depends(valid_subject_id)):
    from src.core.persistence import db

    deleted = db.delete_subject(subject_id)
    return success({"subject_id": subject_id, "deleted": deleted}, request)


@router.get("/{subject_id}/assessment", response_model=AssessmentEnvelope)
def get_assessment(request: Request, subject_id: str = Depends(valid_subject_id)):
    from src.core.persistence import db
    from src.core.exercises import services
    from ..arpabet import to_public_arpabet

    row = db.get_subject(subject_id)
    if row is None:
        return success({"assessment": None}, request)
    return success(to_public_arpabet({"assessment": services.assess_profile(row["id"])}), request)


@router.get("/{subject_id}/gaps", response_model=GapsEnvelope)
def get_gaps(request: Request, subject_id: str = Depends(valid_subject_id)):
    from src.core.persistence import db
    from src.core.scoring import mastery
    from src.core.exercises import services
    from src.core.g2p.tokenization import PHONEME_GUIDE
    from ..arpabet import to_public_arpabet

    row = db.get_subject(subject_id)
    if row is None:
        return success({"phonemes": []}, request)

    now = datetime.now(timezone.utc)
    stats = services.load_profile_stats(row["id"])
    ranked = sorted(stats.items(), key=lambda pair: mastery.lower_confidence_bound(pair[1], now=now))

    phonemes = []
    for phoneme, stat in ranked:
        guide = PHONEME_GUIDE.get(phoneme, {})
        phonemes.append({
            "phoneme": phoneme,
            "mastery": round(mastery.posterior_mean(stat, now=now), 3),
            "lower_confidence_bound": round(mastery.lower_confidence_bound(stat, now=now), 3),
            "independent_attempts": stat.independent_attempts,
            "occurrence_count": stat.occurrence_count,
            "last_practiced_at": stat.last_practiced_at.isoformat() if stat.last_practiced_at else None,
            "example": guide.get("example", ""),
        })
    return success(to_public_arpabet({"phonemes": phonemes}), request)


@router.get("/{subject_id}/attempts", response_model=AttemptsPageEnvelope)
def get_attempts(
    request: Request,
    subject_id: str = Depends(valid_subject_id),
    cursor: Optional[str] = Query(None, description="Opaque cursor from a prior page's next_cursor."),
    limit: int = Query(20, ge=1, le=100),
):
    from src.core.persistence import db

    row = db.get_subject(subject_id)
    if row is None:
        return success({"attempts": [], "next_cursor": None, "has_more": False}, request)

    before_id: Optional[int] = None
    if cursor:
        try:
            before_id = int(cursor)
        except ValueError:
            before_id = None

    rows = db.get_subject_attempts_page(row["id"], limit=limit, before_id=before_id)
    has_more = len(rows) > limit
    page = rows[:limit]
    attempts = [
        {
            "id": r["id"],
            "text": r["text"],
            "phoneme_error_rate": r["phoneme_error_rate"],
            "scorable": bool(r["scorable"]),
            "mastery_updated": bool(r["mastery_updated"]),
            "scoring_engine": r["scoring_engine"],
            "created_at": r["created_at"],
        }
        for r in page
    ]
    next_cursor = str(page[-1]["id"]) if (has_more and page) else None
    return success({"attempts": attempts, "next_cursor": next_cursor, "has_more": has_more}, request)
