"""
Persist one analyzed recording for a subject.

This is the ported ``_record_recording`` from the Flask app: a single atomic
transaction writing the attempt, its phoneme events, the trusted mastery
update, and assignment completion. Mastery is updated ONLY when the audio was
scorable AND the scoring engine AND the reference G2P are all trusted — the
exact gate the Flask app used.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.core.persistence import db
from src.core.scoring import mastery
from src.core.exercises import services
from src.core.shared.app_datetime import format_db_datetime
from src.core.audio.audio_quality import should_update_mastery
from src.core.g2p.phoneme_vectors_professional import canonicalize_phoneme


def _stat_to_row(stat: "mastery.PhonemeStat") -> Dict[str, Any]:
    return {
        "alpha": stat.alpha,
        "beta": stat.beta,
        "attempts_count": stat.independent_attempts,
        "occurrence_count": stat.occurrence_count,
        "last_practiced_at": format_db_datetime(stat.last_practiced_at or datetime.now(timezone.utc)),
    }


def persist_recording(
    user_id: int,
    user_text: str,
    analysis: Dict[str, Any],
    exercise_id: Optional[int],
    idempotency: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Persist an analyzed recording atomically. Returns
    {attempt_id, mastery_updated}. ``analysis`` is the bundle from
    ``analysis.analyze_recording``."""
    decision = analysis["decision"]
    reference = analysis["reference"]
    metrics = analysis["metrics"]
    alignment: List[Dict[str, Any]] = analysis["rows"]
    engine_name = analysis["engine_name"]
    engine_trusted = analysis["engine_trusted"]

    mastery_updated = should_update_mastery(
        scorable=decision.scorable,
        scoring_trusted=engine_trusted and reference.reference_g2p_trusted,
    )

    phoneme_states: Optional[Dict[str, Dict[str, Any]]] = None
    complete_exercise_id: Optional[int] = exercise_id
    if mastery_updated:
        now = datetime.now(timezone.utc)
        stats = services.load_profile_stats(user_id)
        updated = mastery.update_mastery_for_recording(
            stats, alignment, now, quality_weight=decision.quality_weight
        )
        touched = {
            canonicalize_phoneme(r["expected"])
            for r in alignment if r.get("expected") not in (None, "-")
        }
        phoneme_states = {ph: _stat_to_row(updated[ph]) for ph in touched if ph in updated}

    attempt_kwargs = {
        "text": user_text,
        "reference_ipa": analysis["reference_ipa"],
        "predicted_ipa": analysis["predicted_ipa"],
        "phoneme_error_rate": metrics["phoneme_error_rate"],
        "weighted_error": metrics["weighted_error"],
        "exercise_id": exercise_id,
        "raw_weighted_per": metrics["raw_weighted_per"],
        "quality_weight": decision.quality_weight,
        "scorable": decision.scorable,
        "scoring_engine": engine_name,
        "scoring_trusted": engine_trusted,
        "mastery_updated": mastery_updated,
        "insertion_count": metrics.get("insertion_count", metrics.get("insertions", 0)),
        "reference_unit_count": metrics.get("reference_unit_count", 0),
        "g2p_mode": reference.g2p_mode,
        "reference_g2p_trusted": reference.reference_g2p_trusted,
        "reference_g2p_reason": analysis["reference_reason"],
        "audio_processing_status": analysis["recording"]
        .get("audio_processing", {})
        .get("processing_status", "disabled"),
        "audio_quality_metadata": analysis["recording"].get(
            "audio_processing", {}
        ),
        "audio_processing_error": None,
    }
    attempt_id = db.record_recording_atomic(
        user_id=user_id,
        attempt_kwargs=attempt_kwargs,
        alignment=alignment,
        scoring_engine=engine_name,
        phoneme_states=phoneme_states,
        complete_exercise_id=complete_exercise_id,
        idempotency=idempotency,
    )
    return {"attempt_id": attempt_id, "mastery_updated": mastery_updated}
