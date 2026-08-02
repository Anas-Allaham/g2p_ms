"""
Pronunciation audio analysis.

  * POST /api/v1/pronunciation/analyses            — stateless (no subject state)
  * POST /api/v1/subjects/{subject_id}/analyses    — analyze, persist the
        attempt, and atomically update trusted mastery (idempotent)

Both accept multipart fields ``text``, ``audio``, and optional ``exercise_id``.
The stateful variant requires an ``Idempotency-Key`` header. Processed audio is
never returned or retained: every artifact is deleted in an outer ``finally``
block on success or failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from fastapi.responses import JSONResponse

from .. import analysis as analysis_mod
from ..config import API_VERSION
from ..dependencies import require_idempotency_key, save_upload, valid_subject_id
from ..envelopes import build_meta, success
from ..errors import ValidationError
from ..recording import persist_recording
from ..schemas import AnalysisEnvelope
from ..security import require_service_auth
from ..pronunciation_feedback import with_pronunciation_errors

router = APIRouter(prefix=f"/api/{API_VERSION}", tags=["analyses"], dependencies=[Depends(require_service_auth)])


def _require_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValidationError("The 'text' field is required.", details={"field": "text"})
    return text


def _parse_exercise_id(exercise_id: Optional[str]) -> Optional[int]:
    if exercise_id is None or not str(exercise_id).strip():
        return None
    raw = str(exercise_id).strip()
    if not raw.isdigit():
        raise ValidationError("exercise_id must be an integer.", details={"field": "exercise_id"})
    return int(raw)


@router.post("/pronunciation/analyses", response_model=AnalysisEnvelope)
def analyze_stateless(
    request: Request,
    text: str = Form(...),
    audio: UploadFile = File(...),
):
    user_text = _require_text(text)
    cleanup_paths: List[Optional[Path]] = []
    audio_path = analysis_mod.new_upload_path(audio.content_type or "")
    cleanup_paths.extend(analysis_mod.candidate_cleanup_paths(audio_path))
    try:
        save_upload(audio, audio_path)
        result = analysis_mod.analyze_recording(user_text, audio_path, cleanup_paths)
        payload = analysis_mod.build_analysis_payload(result, user_text, mastery_updated=False, persisted=False)
        return success(payload, request)
    finally:
        analysis_mod.cleanup_audio_files(cleanup_paths)


@router.post("/subjects/{subject_id}/analyses", response_model=AnalysisEnvelope)
def analyze_for_subject(
    request: Request,
    subject_id: str = Depends(valid_subject_id),
    text: str = Form(...),
    audio: UploadFile = File(...),
    exercise_id: Optional[str] = Form(None),
    idempotency_key: str = Depends(require_idempotency_key),
):
    from src.core.persistence import db

    user_text = _require_text(text)
    parsed_exercise_id = _parse_exercise_id(exercise_id)

    scope = f"analysis:{subject_id}"
    # Replay a completed identical retry verbatim.
    replay = db.get_idempotent_response(scope, idempotency_key)
    if replay is not None:
        return {"data": with_pronunciation_errors(replay), "meta": build_meta(request)}

    subject = db.get_or_create_subject(subject_id)
    user_id = subject["id"]

    valid_exercise_id = (
        parsed_exercise_id
        if parsed_exercise_id is not None and db.get_sentence_by_id(parsed_exercise_id) is not None
        else None
    )

    cleanup_paths: List[Optional[Path]] = []
    audio_path = analysis_mod.new_upload_path(audio.content_type or "")
    cleanup_paths.extend(analysis_mod.candidate_cleanup_paths(audio_path))
    try:
        save_upload(audio, audio_path)
        result = analysis_mod.analyze_recording(user_text, audio_path, cleanup_paths)
        record = persist_recording(
            user_id=user_id,
            user_text=user_text,
            analysis=result,
            exercise_id=valid_exercise_id,
            idempotency=(scope, idempotency_key),
        )
        payload = analysis_mod.build_analysis_payload(
            result, user_text, mastery_updated=record["mastery_updated"], persisted=True
        )
        payload["subject_id"] = subject_id
        payload["attempt_id"] = record["attempt_id"]
        db.save_idempotent_response(scope, idempotency_key, payload)
        db.checkpoint()
        return success(payload, request)
    finally:
        analysis_mod.cleanup_audio_files(cleanup_paths)
