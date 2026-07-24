"""Audio analysis endpoints: stateless vs stateful, trust gate, idempotency,
and temporary-audio cleanup on success and failure."""

import pytest

from api import analysis as analysis_mod


def _uploads_snapshot():
    return set(analysis_mod.UPLOAD_DIR.glob("*"))


def test_stateless_analysis_does_not_persist(client, sample_wav):
    before = _uploads_snapshot()
    r = client.post(
        "/api/v1/pronunciation/analyses",
        data={"text": "school"},
        files={"audio": ("rec.wav", sample_wav, "audio/wav")},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["persisted"] is False
    assert data["reference_ipa"] and data["predicted_ipa"]
    assert "utterance_score" in data["metrics"]
    assert "processed_audio_data_url" not in data  # audio never returned
    # No leftover audio artifacts.
    assert _uploads_snapshot() == before


def test_stateful_analysis_persists_and_updates_trusted_mastery(client, sample_wav):
    subject = "an-1"
    client.put(f"/api/v1/subjects/{subject}")
    before = _uploads_snapshot()
    r = client.post(
        f"/api/v1/subjects/{subject}/analyses",
        data={"text": "school"},
        files={"audio": ("rec.wav", sample_wav, "audio/wav")},
        headers={"Idempotency-Key": "an-1-key"},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["persisted"] is True
    assert data["scoring_trusted"] is True
    assert data["mastery_updated"] is True
    assert data["attempt_id"]
    assert _uploads_snapshot() == before

    gaps = client.get(f"/api/v1/subjects/{subject}/gaps").json()["data"]["phonemes"]
    assert gaps, "trusted recording should have created phoneme mastery state"


def test_quality_warning_is_scored_but_updates_no_mastery(client, sample_wav, monkeypatch):
    from src.core.audio import audio_quality
    from src.core.audio.audio_quality import AudioQualityDecision

    monkeypatch.setattr(
        audio_quality, "analyze_audio_quality",
        lambda *_a, **_k: AudioQualityDecision(False, 0.0, ["no_speech_modulation"], {}),
    )
    subject = "an-warn"
    client.put(f"/api/v1/subjects/{subject}")
    r = client.post(
        f"/api/v1/subjects/{subject}/analyses",
        data={"text": "school"},
        files={"audio": ("rec.wav", sample_wav, "audio/wav")},
        headers={"Idempotency-Key": "warn-key"},
    )
    data = r.json()["data"]
    assert data["scorable"] is True
    assert data["quality_warning"] is True
    assert data["mastery_updated"] is False
    assert "still processed" in data["mastery_note"].lower()
    # Persisted attempt exists, but no mastery evidence.
    assert client.get(f"/api/v1/subjects/{subject}/gaps").json()["data"]["phonemes"] == []


def test_untrusted_reference_never_updates_mastery(client, sample_wav):
    subject = "an-permit"
    client.put(f"/api/v1/subjects/{subject}")
    r = client.post(
        f"/api/v1/subjects/{subject}/analyses",
        data={"text": "They permit entry"},
        files={"audio": ("rec.wav", sample_wav, "audio/wav")},
        headers={"Idempotency-Key": "permit-key"},
    )
    data = r.json()["data"]
    assert data["scoring_trusted"] is True
    assert data["reference_g2p_trusted"] is False
    assert data["unsupported_heteronyms"] == ["permit"]
    assert data["mastery_updated"] is False
    assert client.get(f"/api/v1/subjects/{subject}/gaps").json()["data"]["phonemes"] == []


def test_idempotent_retry_replays_and_does_not_duplicate(client, sample_wav):
    subject = "an-idem"
    client.put(f"/api/v1/subjects/{subject}")
    args = dict(
        data={"text": "school"},
        files={"audio": ("rec.wav", sample_wav, "audio/wav")},
        headers={"Idempotency-Key": "same-key"},
    )
    first = client.post(f"/api/v1/subjects/{subject}/analyses", **args).json()["data"]
    second = client.post(f"/api/v1/subjects/{subject}/analyses", **args).json()["data"]
    assert first["attempt_id"] == second["attempt_id"]
    # Exactly one attempt was recorded.
    hist = client.get(f"/api/v1/subjects/{subject}/attempts").json()["data"]
    assert len(hist["attempts"]) == 1

    # A different key creates a new attempt.
    client.post(
        f"/api/v1/subjects/{subject}/analyses",
        data={"text": "school"},
        files={"audio": ("rec.wav", sample_wav, "audio/wav")},
        headers={"Idempotency-Key": "other-key"},
    )
    hist = client.get(f"/api/v1/subjects/{subject}/attempts").json()["data"]
    assert len(hist["attempts"]) == 2


def test_missing_idempotency_key_is_rejected(client, sample_wav):
    client.put("/api/v1/subjects/an-nokey")
    r = client.post(
        "/api/v1/subjects/an-nokey/analyses",
        data={"text": "school"},
        files={"audio": ("rec.wav", sample_wav, "audio/wav")},
    )
    assert r.status_code == 422
    assert r.json()["error"]["details"]["header"] == "Idempotency-Key"


def test_failure_after_upload_leaves_no_audio_and_returns_500(client, sample_wav, monkeypatch):
    from api import acoustic

    monkeypatch.setattr(acoustic, "transcribe", lambda _p: (_ for _ in ()).throw(RuntimeError("boom")))
    before = _uploads_snapshot()
    r = client.post(
        "/api/v1/pronunciation/analyses",
        data={"text": "school"},
        files={"audio": ("rec.wav", sample_wav, "audio/wav")},
    )
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert _uploads_snapshot() == before


def test_missing_ffmpeg_for_compressed_audio_is_clear_400(client, sample_wav, monkeypatch):
    monkeypatch.setattr(analysis_mod, "_find_ffmpeg_executable", lambda: None)
    before = _uploads_snapshot()
    r = client.post(
        "/api/v1/pronunciation/analyses",
        data={"text": "school"},
        files={"audio": ("rec.webm", sample_wav, "audio/webm")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "audio_decode_unavailable"
    assert _uploads_snapshot() == before
