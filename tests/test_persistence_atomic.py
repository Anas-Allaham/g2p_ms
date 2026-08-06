"""Atomic recording writes + subject isolation/cascade + idempotency ledger."""

import json

import pytest


def test_attempt_persists_json_safe_audio_processing_metadata(temp_db):
    subject = temp_db.get_or_create_subject("audio-metadata")
    temp_db.record_attempt(
        user_id=subject["id"],
        text="school",
        reference_ipa="skuːl",
        predicted_ipa="skuːl",
        phoneme_error_rate=0.0,
        weighted_error=0.0,
        audio_processing_status="completed",
        audio_quality_metadata={
            "duration_seconds": 1.2,
            "speech_seconds": 0.9,
            "scoring_allowed": True,
            "rejection_reasons": [],
        },
    )
    row = temp_db.get_connection().execute(
        "SELECT audio_processing_status, audio_quality_metadata FROM attempts"
    ).fetchone()
    assert row["audio_processing_status"] == "completed"
    assert json.loads(row["audio_quality_metadata"])["speech_seconds"] == 0.9


def test_atomic_recording_rolls_back_on_failure(temp_db, monkeypatch):
    db = temp_db
    subject = db.get_or_create_subject("atomic")
    uid = subject["id"]

    alignment = [{"expected": "s", "spoken": "s", "result": "correct",
                  "articulatory_distance": 0.0, "alignment_cost": 0.0}]

    def boom(*a, **k):
        raise RuntimeError("simulated mid-transaction failure")

    monkeypatch.setattr(db, "_complete_latest_assignment", boom)

    with pytest.raises(RuntimeError):
        db.record_recording_atomic(
            user_id=uid,
            attempt_kwargs={"text": "t", "reference_ipa": "s", "predicted_ipa": "s",
                            "phoneme_error_rate": 0.0, "weighted_error": 0.0, "scorable": True,
                            "scoring_engine": "panphon", "scoring_trusted": True, "mastery_updated": True},
            alignment=alignment, scoring_engine="panphon",
            phoneme_states={"s": {"alpha": 2.0, "beta": 1.0, "attempts_count": 1,
                                  "occurrence_count": 1, "last_practiced_at": "2026-01-01 00:00:00"}},
            complete_exercise_id=1,
        )

    conn = db.get_connection()
    assert conn.execute("SELECT COUNT(*) AS n FROM attempts").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM attempt_phoneme_events").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM phoneme_skill_state").fetchone()["n"] == 0


def test_atomic_recording_commits_all_on_success(temp_db):
    db = temp_db
    subject = db.get_or_create_subject("ok")
    db.record_recording_atomic(
        user_id=subject["id"],
        attempt_kwargs={"text": "t", "reference_ipa": "s", "predicted_ipa": "s",
                        "phoneme_error_rate": 0.0, "weighted_error": 0.0, "scorable": True,
                        "scoring_engine": "panphon", "scoring_trusted": True, "mastery_updated": True},
        alignment=[{"expected": "s", "spoken": "s", "result": "correct",
                    "articulatory_distance": 0.0, "alignment_cost": 0.0}],
        scoring_engine="panphon",
        phoneme_states={"s": {"alpha": 2.0, "beta": 1.0, "attempts_count": 1,
                              "occurrence_count": 1, "last_practiced_at": "2026-01-01 00:00:00"}},
    )
    conn = db.get_connection()
    assert conn.execute("SELECT COUNT(*) AS n FROM attempts").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM attempt_phoneme_events").fetchone()["n"] == 1
    assert conn.execute("SELECT alpha FROM phoneme_skill_state WHERE phoneme='s'").fetchone()["alpha"] == 2.0


def test_idempotent_atomic_write_is_rejected_on_duplicate_key(temp_db):
    import sqlite3

    db = temp_db
    subject = db.get_or_create_subject("idem")
    kwargs = {"text": "t", "reference_ipa": "s", "predicted_ipa": "s",
              "phoneme_error_rate": 0.0, "weighted_error": 0.0, "scorable": True,
              "scoring_engine": "panphon", "scoring_trusted": True, "mastery_updated": True}
    alignment = [{"expected": "s", "spoken": "s", "result": "correct",
                  "articulatory_distance": 0.0, "alignment_cost": 0.0}]

    db.record_recording_atomic(user_id=subject["id"], attempt_kwargs=kwargs, alignment=alignment,
                               idempotency=("analysis:idem", "key-1"))
    # A second write with the same key must fail and roll back (no 2nd attempt).
    with pytest.raises(sqlite3.IntegrityError):
        db.record_recording_atomic(user_id=subject["id"], attempt_kwargs=kwargs, alignment=alignment,
                                   idempotency=("analysis:idem", "key-1"))
    conn = db.get_connection()
    assert conn.execute("SELECT COUNT(*) AS n FROM attempts").fetchone()["n"] == 1


def test_delete_subject_cascades_and_isolates(temp_db):
    db = temp_db
    a = db.get_or_create_subject("subject-a")
    b = db.get_or_create_subject("subject-b")
    kwargs = {"text": "t", "reference_ipa": "s", "predicted_ipa": "s",
              "phoneme_error_rate": 0.0, "weighted_error": 0.0, "scorable": True,
              "scoring_engine": "panphon", "scoring_trusted": True, "mastery_updated": True}
    alignment = [{"expected": "s", "spoken": "s", "result": "correct",
                  "articulatory_distance": 0.0, "alignment_cost": 0.0}]
    for subject in (a, b):
        db.record_recording_atomic(
            user_id=subject["id"], attempt_kwargs=kwargs, alignment=alignment,
            scoring_engine="panphon",
            phoneme_states={"s": {"alpha": 2.0, "beta": 1.0, "attempts_count": 1,
                                  "occurrence_count": 1, "last_practiced_at": "2026-01-01 00:00:00"}},
        )

    assert db.delete_subject("subject-a") is True
    conn = db.get_connection()
    # A's rows are gone; B's survive (isolation).
    assert conn.execute("SELECT COUNT(*) AS n FROM attempts WHERE subject_pk=?", (a["id"],)).fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM phoneme_skill_state WHERE subject_pk=?", (a["id"],)).fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM attempts WHERE subject_pk=?", (b["id"],)).fetchone()["n"] == 1
    assert db.delete_subject("subject-a") is False  # already gone
