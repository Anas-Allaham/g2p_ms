"""Standalone DeepFilterNet/Silero endpoint contract and file lifecycle."""

from pathlib import Path

from api import analysis as analysis_mod
from api.errors import ServiceError

NO_AUTH = {"Authorization": ""}


def _uploads_snapshot():
    return set(analysis_mod.UPLOAD_DIR.glob("*"))


def _fake_clean_result(tmp_path: Path):
    processing = tmp_path / "request_audio_cleaning"
    processing.mkdir()
    cleaned_48k = processing / "cleaned_48k_mono.wav"
    cleaned_16k = processing / "cleaned_16k_mono.wav"
    cleaned_48k.write_bytes(b"cleaned-48k")
    cleaned_16k.write_bytes(b"cleaned-16k")
    return {
        "cleaned_audio_48k_path": cleaned_48k,
        "cleaned_audio_16k_path": cleaned_16k,
        "reduced_audio_path": cleaned_16k,
        "noise_reduction_applied": True,
        "preprocessing_pipeline": "ffmpeg_deepfilternet_silero_vad",
        "audio_processing": {
            "original_preserved": True,
            "processing_status": "completed",
            "speech_seconds": 1.25,
            "scoring_allowed": True,
            "rejection_reasons": [],
        },
        "cleanup_paths": [processing],
    }


def test_clean_audio_returns_48k_wav_and_removes_temporary_files(
    client, sample_wav, monkeypatch, tmp_path
):
    result = _fake_clean_result(tmp_path)
    monkeypatch.setattr(analysis_mod, "clean_recording", lambda _path: result)
    before = _uploads_snapshot()

    response = client.post(
        "/api/v1/audio/clean",
        files={"audio": ("learner sample.wav", sample_wav, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.content == b"cleaned-48k"
    assert response.headers["content-type"].startswith("audio/wav")
    assert 'filename="learner_sample_cleaned.wav"' in response.headers[
        "content-disposition"
    ]
    assert response.headers["x-audio-processing-pipeline"] == (
        "ffmpeg_deepfilternet_silero_vad"
    )
    assert response.headers["x-noise-reduction-applied"] == "true"
    assert response.headers["x-original-preserved"] == "true"
    assert response.headers["x-audio-scoring-allowed"] == "true"
    assert response.headers["x-audio-cleaning-backend"] == "deepfilternet"
    assert response.headers["x-audio-fallback-used"] == "false"
    assert not result["cleanup_paths"][0].exists()
    assert _uploads_snapshot() == before


def test_clean_audio_can_return_16k_stt_file(client, sample_wav, monkeypatch, tmp_path):
    result = _fake_clean_result(tmp_path)
    monkeypatch.setattr(analysis_mod, "clean_recording", lambda _path: result)

    response = client.post(
        "/api/v1/audio/clean?sample_rate=16000",
        files={"audio": ("recording.wav", sample_wav, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.content == b"cleaned-16k"


def test_clean_audio_reports_cleanvoice_fallback_headers(
    client, sample_wav, monkeypatch, tmp_path
):
    result = _fake_clean_result(tmp_path)
    result["preprocessing_pipeline"] = "ffmpeg_cleanvoice_fallback_silero_vad"
    result["audio_processing"].update(
        {"cleaning_backend": "cleanvoice", "fallback_used": True}
    )
    monkeypatch.setattr(analysis_mod, "clean_recording", lambda _path: result)

    response = client.post(
        "/api/v1/audio/clean",
        files={"audio": ("recording.wav", sample_wav, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["x-audio-cleaning-backend"] == "cleanvoice"
    assert response.headers["x-audio-fallback-used"] == "true"


def test_clean_audio_failure_does_not_expose_internal_path(
    client, sample_wav, monkeypatch
):
    def fail(_path):
        raise ServiceError(
            "The recording could not be cleaned. Please try again.",
            code="deepfilternet_processing_failed",
            status_code=502,
        )

    monkeypatch.setattr(analysis_mod, "clean_recording", fail)
    before = _uploads_snapshot()
    response = client.post(
        "/api/v1/audio/clean",
        files={"audio": ("recording.wav", sample_wav, "audio/wav")},
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "deepfilternet_processing_failed"
    assert "path" not in response.text.lower()
    assert _uploads_snapshot() == before


def test_clean_audio_reports_disabled_feature(client, sample_wav, monkeypatch):
    def disabled(_path):
        raise ServiceError(
            "Audio cleaning is disabled on this service.",
            code="audio_cleaning_disabled",
            status_code=503,
        )

    monkeypatch.setattr(analysis_mod, "clean_recording", disabled)
    response = client.post(
        "/api/v1/audio/clean",
        files={"audio": ("recording.wav", sample_wav, "audio/wav")},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "audio_cleaning_disabled"


def test_clean_audio_rejects_empty_upload_and_cleans_up(client):
    before = _uploads_snapshot()
    response = client.post(
        "/api/v1/audio/clean",
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert _uploads_snapshot() == before


def test_clean_audio_requires_service_authentication(client, sample_wav):
    response = client.post(
        "/api/v1/audio/clean",
        files={"audio": ("recording.wav", sample_wav, "audio/wav")},
        headers=NO_AUTH,
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_clean_audio_rejects_unsupported_media(client):
    response = client.post(
        "/api/v1/audio/clean",
        files={"audio": ("notes.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_clean_audio_validates_output_sample_rate(client, sample_wav):
    response = client.post(
        "/api/v1/audio/clean?sample_rate=22050",
        files={"audio": ("recording.wav", sample_wav, "audio/wav")},
    )
    assert response.status_code == 422


def test_clean_audio_openapi_declares_wav_response(client):
    operation = client.get("/openapi.json").json()["paths"][
        "/api/v1/audio/clean"
    ]["post"]
    assert "audio/wav" in operation["responses"]["200"]["content"]
