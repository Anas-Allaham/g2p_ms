from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from api import analysis as analysis_mod
from api.errors import ServiceError
from src.core.audio import audio_cleaning, cleanvoice_service


def test_enhance_recording_uses_pronunciation_safe_options(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    output = tmp_path / "recording_cleanvoice.wav"
    source.write_bytes(b"wav")
    observed = {}

    class FakeCleanvoice:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def process(self, path, **kwargs):
            observed["path"] = path
            observed["options"] = kwargs
            Path(kwargs["output_path"]).write_bytes(b"x" * 100)

    monkeypatch.setenv("CLEANVOICE_API_KEY", "test-key")
    monkeypatch.setenv("CLEANVOICE_FALLBACK_ENABLED", "1")
    monkeypatch.setattr(cleanvoice_service, "Cleanvoice", FakeCleanvoice)

    result = cleanvoice_service.enhance_recording(source, output)

    assert result == output
    assert observed["init"]["api_key"] == "test-key"
    assert observed["options"]["remove_noise"] is True
    assert observed["options"]["normalize"] is True
    assert observed["options"]["export_format"] == "wav"
    for option in (
        "fillers",
        "long_silences",
        "mouth_sounds",
        "breath",
        "stutters",
        "hesitations",
    ):
        assert observed["options"][option] is False


def test_cleanvoice_fallback_auto_enables_with_key_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("CLEANVOICE_FALLBACK_ENABLED", raising=False)
    monkeypatch.delenv("CLEANVOICE_ENABLED", raising=False)
    monkeypatch.delenv("CLEANVOICE_API_KEY", raising=False)
    assert cleanvoice_service.cleanvoice_fallback_configured() is False

    monkeypatch.setenv("CLEANVOICE_API_KEY", "test-key")
    assert cleanvoice_service.cleanvoice_fallback_configured() is True
    monkeypatch.setenv("CLEANVOICE_FALLBACK_ENABLED", "0")
    assert cleanvoice_service.cleanvoice_fallback_configured() is False


def test_cleanvoice_sdk_errors_do_not_expose_provider_details(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"wav")

    class FailingCleanvoice:
        def __init__(self, **_kwargs):
            pass

        def process(self, *_args, **_kwargs):
            raise RuntimeError("signed-url-with-sensitive-query")

    monkeypatch.setenv("CLEANVOICE_API_KEY", "test-key")
    monkeypatch.setattr(cleanvoice_service, "Cleanvoice", FailingCleanvoice)
    with pytest.raises(cleanvoice_service.CleanvoiceProcessingError) as caught:
        cleanvoice_service.enhance_recording(source, tmp_path / "clean.wav")
    assert "signed-url" not in str(caught.value)


def test_eligible_local_failure_uses_cleanvoice_fallback(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"original")
    expected = {"preprocessing_pipeline": "ffmpeg_cleanvoice_fallback_silero_vad"}

    class FailingCleaner:
        def process(self, _path):
            raise audio_cleaning.AudioCleaningError(
                "deepfilternet_processing_failed",
                "Noise reduction failed for this recording.",
                status_code=502,
            )

    monkeypatch.setenv("CLEANVOICE_API_KEY", "test-key")
    monkeypatch.setattr(audio_cleaning, "cleaner_from_settings", lambda _settings: FailingCleaner())
    monkeypatch.setattr(
        analysis_mod,
        "_clean_with_cleanvoice_fallback",
        lambda _path, *, primary_error_code: (
            expected
            if primary_error_code == "deepfilternet_processing_failed"
            else pytest.fail("unexpected primary error")
        ),
    )

    assert analysis_mod.clean_recording(source) is expected


def test_invalid_input_does_not_leave_service_for_cleanvoice(tmp_path, monkeypatch):
    source = tmp_path / "bad.wav"
    source.write_bytes(b"bad")

    class InvalidCleaner:
        def process(self, _path):
            raise audio_cleaning.AudioCleaningError(
                "unsupported_or_corrupt_audio",
                "The recording is corrupt or uses an unsupported audio format.",
                status_code=400,
            )

    monkeypatch.setenv("CLEANVOICE_API_KEY", "test-key")
    monkeypatch.setattr(audio_cleaning, "cleaner_from_settings", lambda _settings: InvalidCleaner())
    monkeypatch.setattr(
        analysis_mod,
        "_clean_with_cleanvoice_fallback",
        lambda *_args, **_kwargs: pytest.fail("invalid input must not be uploaded"),
    )
    with pytest.raises(ServiceError) as caught:
        analysis_mod.clean_recording(source)
    assert caught.value.code == "unsupported_or_corrupt_audio"


def test_cleanvoice_fallback_restores_48k_16k_and_metadata(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"immutable")

    def fake_convert(_source, output, *, sample_rate, **_kwargs):
        sf.write(
            str(output),
            np.full(sample_rate, 0.1, dtype=np.float32),
            sample_rate,
            subtype="PCM_16",
        )
        return Path(output)

    class FakeCleaner:
        def _run_silero_vad(self, _path, _device):
            return [{"start": 0.1, "end": 0.9}]

        def _quality_metadata(self, _path, segments, device):
            return {
                "duration_seconds": 1.0,
                "speech_seconds": 0.8,
                "speech_ratio": 0.8,
                "clipping_ratio": 0.0,
                "speech_segments": segments,
                "noise_reduction_applied": True,
                "processing_status": "completed",
                "scoring_allowed": True,
                "rejection_reasons": [],
                "original_preserved": True,
                "device": device,
            }

    monkeypatch.setattr(
        analysis_mod,
        "settings",
        SimpleNamespace(
            audio_cleaning_output_dir=tmp_path / "outputs",
            audio_cleaning_timeout_seconds=10,
            audio_cleaning_keep_intermediate_files=False,
            retain_audio=False,
        ),
    )
    monkeypatch.setattr(audio_cleaning, "convert_with_ffmpeg", fake_convert)
    monkeypatch.setattr(audio_cleaning, "cleaner_from_settings", lambda _settings: FakeCleaner())
    monkeypatch.setattr(
        cleanvoice_service,
        "enhance_recording",
        lambda input_path, output_path: fake_convert(
            input_path,
            output_path,
            sample_rate=48_000,
        ),
    )

    result = analysis_mod._clean_with_cleanvoice_fallback(
        source,
        primary_error_code="deepfilternet_processing_failed",
    )

    assert sf.info(str(result["cleaned_audio_48k_path"])).samplerate == 48_000
    assert sf.info(str(result["cleaned_audio_16k_path"])).samplerate == 16_000
    assert result["audio_processing"]["fallback_used"] is True
    assert result["audio_processing"]["cleaning_backend"] == "cleanvoice"
    assert result["audio_processing"]["primary_error_code"] == (
        "deepfilternet_processing_failed"
    )
    assert source.read_bytes() == b"immutable"
    analysis_mod.cleanup_audio_files(result["cleanup_paths"])


def test_both_cleaners_failing_returns_safe_error(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"original")

    class FailingCleaner:
        def process(self, _path):
            raise audio_cleaning.AudioCleaningError(
                "deepfilternet_initialization_failed",
                "The noise-reduction model is temporarily unavailable.",
                status_code=503,
            )

    monkeypatch.setenv("CLEANVOICE_API_KEY", "test-key")
    monkeypatch.setattr(audio_cleaning, "cleaner_from_settings", lambda _settings: FailingCleaner())
    monkeypatch.setattr(
        analysis_mod,
        "_clean_with_cleanvoice_fallback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("signed-provider-url")
        ),
    )
    with pytest.raises(ServiceError) as caught:
        analysis_mod.clean_recording(source)
    assert caught.value.code == "audio_cleaning_all_providers_failed"
    assert "signed-provider-url" not in str(caught.value)
