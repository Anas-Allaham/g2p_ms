from pathlib import Path

import numpy as np
import pytest

from src.core.audio import cleanvoice_service
from api import analysis as analysis_mod
from api import acoustic
from src.core.audio.audio_quality import AudioQualityDecision


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
            Path(kwargs["output_path"]).write_bytes(b"clean wav")

    monkeypatch.setenv("CLEANVOICE_API_KEY", "test-key")
    monkeypatch.setenv("CLEANVOICE_ENABLED", "1")
    monkeypatch.setattr(cleanvoice_service, "Cleanvoice", FakeCleanvoice)

    result = cleanvoice_service.enhance_recording(source, output)

    assert result == output
    assert observed["init"]["api_key"] == "test-key"
    assert observed["path"] == str(source)
    assert observed["options"]["remove_noise"] is True
    assert observed["options"]["normalize"] is True
    assert observed["options"]["export_format"] == "wav"
    for cutting_option in (
        "fillers", "long_silences", "mouth_sounds", "breath", "stutters"
    ):
        assert observed["options"][cutting_option] is False


def test_cleanvoice_auto_enables_only_when_key_is_present(monkeypatch):
    monkeypatch.delenv("CLEANVOICE_ENABLED", raising=False)
    monkeypatch.delenv("CLEANVOICE_API_KEY", raising=False)
    assert cleanvoice_service.cleanvoice_enabled() is False
    assert cleanvoice_service.cleanvoice_configured() is False

    monkeypatch.setenv("CLEANVOICE_API_KEY", "test-key")
    assert cleanvoice_service.cleanvoice_enabled() is True
    assert cleanvoice_service.cleanvoice_configured() is True


def test_sdk_errors_are_not_exposed(tmp_path, monkeypatch):
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


def test_process_recording_uses_cleanvoice_despite_quality_warning(tmp_path, monkeypatch):
    import librosa

    from src.core.audio import audio_quality

    source = tmp_path / "recording.wav"
    source.write_bytes(b"raw wav")
    loaded_paths = []

    def fake_load(path, **_kwargs):
        loaded_paths.append(Path(path))
        return np.ones(16000, dtype=np.float32), 16000

    def fake_enhance(_source, output):
        Path(output).write_bytes(b"clean wav")
        return Path(output)

    # convert_audio_to_wav / _apply_noise_reduction are module-level in analysis;
    # analyze_quality / cleanvoice hooks are re-imported per call, so patching the
    # source modules is enough. acoustic.transcribe is the mock seam for the model.
    monkeypatch.setattr(analysis_mod, "convert_audio_to_wav", lambda _path: source)
    monkeypatch.setattr(librosa, "load", fake_load)
    monkeypatch.setattr(
        audio_quality,
        "analyze_audio_quality",
        lambda *_args: AudioQualityDecision(False, 0.0, ["no_speech_modulation"], {}),
    )
    monkeypatch.setattr(cleanvoice_service, "cleanvoice_configured", lambda: True)
    monkeypatch.setattr(cleanvoice_service, "enhance_recording", fake_enhance)
    monkeypatch.setattr(
        analysis_mod,
        "_apply_noise_reduction",
        lambda *_args: pytest.fail("local cleanup must not run after Cleanvoice succeeds"),
    )
    monkeypatch.setattr(acoustic, "transcribe", lambda _path: "s")

    result = analysis_mod.process_recording(source)

    cleanvoice_path = source.with_name(source.stem + "_cleanvoice.wav")
    assert result["cleanvoice_applied"] is True
    assert result["quality_decision"].scorable is False
    assert result["predicted_ipa"] == "s"
    assert result["preprocessing_pipeline"] == "cleanvoice_noise_reduction_normalization"
    assert result["reduced_audio_path"] == cleanvoice_path
    # transcribe is mocked, so librosa.load is only called for the quality probe.
    assert loaded_paths == [source]
