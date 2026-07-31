"""
Adaptive exercise selection.

  * POST /api/v1/exercises/generate                    — stateless, from
        supplied per-phoneme metrics (no subject state read or written)
  * POST /api/v1/subjects/{subject_id}/exercises/next  — select/generate the
        next adaptive exercise using the subject's evidence, and record the
        assignment (idempotent)

The ``next`` selection logic is the ported ``/practice/next`` flow: cold-start
diagnostic phase, then weak-phoneme + confusion-aware targeting, with the LLM
generation fallback verified by the same G2P path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from ..config import API_VERSION
from ..dependencies import require_idempotency_key, valid_subject_id
from ..envelopes import build_meta, error_body, success
from ..errors import ValidationError
from ..schemas import ExerciseEnvelope, ExerciseGenerateRequest, NextExerciseEnvelope
from ..security import require_service_auth

router = APIRouter(prefix=f"/api/{API_VERSION}", tags=["exercises"], dependencies=[Depends(require_service_auth)])


@router.post("/exercises/generate", response_model=ExerciseEnvelope)
def generate_stateless(payload: ExerciseGenerateRequest, request: Request):
    from src.core.exercises import services
    from src.core.g2p.g2p_service import g2p_convert_with_metadata
    from src.core.g2p.tokenization import tokenize_reference_ipa
    from ..arpabet import metrics_to_internal_ipa, to_public_arpabet

    try:
        metrics = metrics_to_internal_ipa(payload.metrics)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "The metrics object contains an unsupported ARPAbet phoneme.",
            details={"reason": str(exc)},
        ) from exc
    result = services.generate_exercise(metrics, g2p_convert_with_metadata, tokenize_reference_ipa, set())
    if result.get("exercise") is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_body(
                "exercise_bank_empty",
                result.get("error", "No exercises available."),
                details=to_public_arpabet({"assessment": result.get("assessment")}),
                request=request,
            ),
        )
    return success(to_public_arpabet(result), request)


def _select_next(user_id: int) -> Dict[str, Any]:
    """Ported /practice/next selection. Returns the response payload (without
    recording the assignment)."""
    from src.core.scoring import assessment as assessment_mod
    from src.core.persistence import db
    from src.core.scoring import mastery
    from src.core.exercises import services
    from src.core.g2p.g2p_service import g2p_convert_with_metadata
    from src.core.g2p.tokenization import ipa_reading_guide, tokenize_reference_ipa

    now = datetime.now(timezone.utc)
    stats = services.load_profile_stats(user_id)
    context_stats = db.get_phoneme_context_stats(user_id)
    assessment = services.assess_profile(user_id, now=now)
    diag = assessment_mod.diagnostic_status(context_stats)
    recently_served = db.get_recently_served_sentence_ids(user_id)
    confusion_pairs = db.get_confusion_pairs(user_id)

    overmastered = sorted(mastery.get_overmastered_phonemes(stats, now=now))
    under_observed = diag["uncovered"]

    confusion_hint = None
    confusion_phoneme = None
    exercise_type = "diagnostic"
    targets: List[str] = []

    if diag["in_diagnostic"] or not stats:
        mode = "diagnostic"
        selection = services.choose_exercise(
            targets=[], overmastered=overmastered, under_observed=under_observed,
            overall_level=assessment.get("exercise_level", assessment["overall_level"]),
            recently_served_ids=recently_served, g2p_convert=g2p_convert_with_metadata,
            ipa_to_tokens=tokenize_reference_ipa, exercise_type="diagnostic", diagnostic=True,
        )
    else:
        targets = mastery.rank_weak_phonemes(stats, now=now)
        top_target = targets[0] if targets else None
        confusion = assessment_mod.main_confusion_for(top_target, confusion_pairs) if top_target else None
        if confusion is not None:
            confusion_phoneme = confusion["spoken"]
            confusion_hint = f"{confusion['expected']} vs {confusion['spoken']}"
        if top_target is not None:
            exercise_type = assessment_mod.exercise_type_for_mastery(
                mastery.posterior_mean(stats[top_target], now=now) if top_target in stats else None
            )
        selection = services.choose_exercise(
            targets=targets, overmastered=overmastered, under_observed=under_observed,
            overall_level=assessment.get("exercise_level", assessment["overall_level"]),
            recently_served_ids=recently_served, g2p_convert=g2p_convert_with_metadata,
            ipa_to_tokens=tokenize_reference_ipa, exercise_type=exercise_type,
            top_target=top_target, confusion_phoneme=confusion_phoneme, diagnostic=False,
        )
        mode = selection["source_mode"]

    chosen = selection["exercise"]
    if selection["generated"] is not None:
        gen = selection["generated"]
        new_id = db.insert_sentence(
            text=gen["text"], reference_ipa=gen["reference_ipa"], word_count=gen["word_count"],
            level_proxy=gen["level_proxy"], phoneme_counts=gen["phoneme_counts"],
            source=gen.get("source", "llm_generated"),
        )
        if new_id is not None:
            gen["id"] = new_id
            chosen = gen
            mode = selection["source_mode"]
        else:
            existing = db.get_sentence_by_text(gen["text"])
            if existing is not None:
                gen["id"] = existing["id"]
                chosen = gen
                mode = selection["source_mode"]

    if chosen is None:
        return {"_unavailable": True}

    db.record_practice_assignment(user_id, chosen["id"], targets)

    return {
        "sentence_id": chosen["id"],
        "text": chosen["text"],
        "reference_ipa": chosen["reference_ipa"],
        "reference_guide": ipa_reading_guide(chosen["reference_ipa"]),
        "target_phonemes": targets,
        "mode": mode,
        "exercise_type": exercise_type,
        "confusion_hint": confusion_hint,
        "assessment": assessment,
        "diagnostic": {
            "in_diagnostic": diag["in_diagnostic"],
            "covered_count": diag["covered_count"],
            "coverage_target": diag["coverage_target"],
        },
    }


@router.post("/subjects/{subject_id}/exercises/next", response_model=NextExerciseEnvelope)
def next_for_subject(
    request: Request,
    subject_id: str = Depends(valid_subject_id),
    idempotency_key: str = Depends(require_idempotency_key),
):
    from src.core.persistence import db
    from ..arpabet import to_public_arpabet

    scope = f"exercise_next:{subject_id}"
    replay = db.get_idempotent_response(scope, idempotency_key)
    if replay is not None:
        return {"data": replay, "meta": build_meta(request)}

    subject = db.get_or_create_subject(subject_id)
    user_id = subject["id"]

    if not db.reserve_idempotency_key(scope, idempotency_key, user_id):
        # A concurrent duplicate already reserved it; replay or report in-flight.
        replay = db.get_idempotent_response(scope, idempotency_key)
        if replay is not None:
            return {"data": replay, "meta": build_meta(request)}

    try:
        payload = _select_next(user_id)
        if payload.get("_unavailable"):
            db.release_idempotency_key(scope, idempotency_key)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=error_body(
                    "exercise_bank_empty",
                    "No practice sentences are available. Seed the exercise bank first.",
                    request=request,
                ),
            )
        payload["subject_id"] = subject_id
        public_payload = to_public_arpabet(payload)
        db.save_idempotent_response(scope, idempotency_key, public_payload)
        db.checkpoint()
        return success(public_payload, request)
    except Exception:
        db.release_idempotency_key(scope, idempotency_key)
        raise
