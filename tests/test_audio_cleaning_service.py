"""Unit tests for the offline DeepFilterNet/Silero orchestration.

Normal tests never initialize or download either model. The real-model smoke
test is opt-in via RUN_AUDIO_MODEL_TESTS=1.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.core.audio import audio_cleaning as cleaning


def _options(tmp_path: Path, **overrides) -> cleaning.AudioCleaningOptions:
    values = {
        "enabled": True,
        "use_gpu": False,
        "min_duration_seconds": 0.5,
        "min_speech_seconds": 0.3,
        "max_clipping_ratio": 0.01,
        "keep_intermediate_files": False,
        "timeout_seconds": 30.0,
        "output_root": tmp_path / "outputs",
    }
    values.update(overrides)
    return cleaning.AudioCleaningOptions(**values)


def _install_fake_pipeline(
    monkeypatch,
    cleaner: cleaning.ExerciseAudioCleaner,
    *,
    duration: float = 1.0,
    amplitude: float = 0.2,
    segments=None,
):
    observed = {"deepfilter_calls": 0, "conversions": []}

    def fake_convert(source, output, *, sample_rate, timeout_seconds, ffmpeg_binary=None):
        del source, timeout_seconds, ffmpeg_binary
        observed["conversions"].append(sample_rate)
        count = max(1, int(sample_rate * duration))
        signal = np.full(count, amplitude, dtype=np.float32)
        sf.write(str(output), signal, sample_rate, subtype="PCM_16")
        return Path(output)

    def fake_deepfilter(source, output, device):
        del device
        observed["deepfilter_calls"] += 1
        shutil.copyfile(source, output)

    monkeypatch.setattr(cleaning, "convert_with_ffmpeg", fake_convert)
    monkeypatch.setattr(cleaner, "_run_deepfilternet", fake_deepfilter)
    monkeypatch.setattr(
        cleaner,
        "_run_silero_vad",
        lambda _path, _device: (
            [{"start": 0.1, "end": min(duration, 0.9)}]
            if segments is None
            else segments
        ),
    )
    return observed


def test_success_creates_48k_and_16k_and_preserves_original(tmp_path, monkeypatch):
    source = tmp_path / "student.webm"
    original = b"immutable original recording"
    source.write_bytes(original)
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))
    _install_fake_pipeline(monkeypatch, cleaner)

    result = cleaner.process(source, recording_id="attempt-42")

    assert source.read_bytes() == original
    assert result.original_audio_path == source.resolve()
    assert sf.info(str(result.cleaned_audio_48k_path)).samplerate == 48_000
    assert sf.info(str(result.cleaned_audio_16k_path)).samplerate == 16_000
    assert sf.info(str(result.cleaned_audio_48k_path)).channels == 1
    assert result.metadata["noise_reduction_applied"] is True
    assert result.metadata["cleaning_backend"] == "deepfilternet"
    assert result.metadata["fallback_used"] is False
    assert result.metadata["processing_status"] == "completed"
    assert result.metadata["original_preserved"] is True
    assert result.metadata["scoring_allowed"] is True


def test_missing_input_file_is_clear(tmp_path):
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))
    with pytest.raises(cleaning.AudioCleaningError) as caught:
        cleaner.process(tmp_path / "missing.wav")
    assert caught.value.code == "missing_input_file"
    assert caught.value.status_code == 400


def test_ffmpeg_failure_preserves_original_and_cleans_workdir(tmp_path, monkeypatch):
    source = tmp_path / "recording.webm"
    source.write_bytes(b"original")
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))

    def fail_conversion(*_args, **_kwargs):
        raise cleaning.AudioCleaningError(
            "unsupported_or_corrupt_audio",
            "The recording is corrupt or uses an unsupported audio format.",
            status_code=400,
        )

    monkeypatch.setattr(cleaning, "convert_with_ffmpeg", fail_conversion)
    with pytest.raises(cleaning.AudioCleaningError) as caught:
        cleaner.process(source)
    assert caught.value.code == "unsupported_or_corrupt_audio"
    assert source.read_bytes() == b"original"
    assert not list((tmp_path / "outputs").glob("*_audio_cleaning"))


def test_failed_processing_is_retryable(tmp_path, monkeypatch):
    source = tmp_path / "retry.wav"
    source.write_bytes(b"original")
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))
    _install_fake_pipeline(monkeypatch, cleaner)
    successful_convert = cleaning.convert_with_ffmpeg
    calls = {"count": 0}

    def flaky_convert(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise cleaning.AudioCleaningError(
                "processing_failed",
                "The recording could not be cleaned. Please try again.",
            )
        return successful_convert(*args, **kwargs)

    monkeypatch.setattr(cleaning, "convert_with_ffmpeg", flaky_convert)
    with pytest.raises(cleaning.AudioCleaningError):
        cleaner.process(source)
    result = cleaner.process(source)
    assert result.metadata["processing_status"] == "completed"


def test_failure_manifest_keeps_only_safe_error_when_debug_retention_is_on(
    tmp_path, monkeypatch
):
    source = tmp_path / "failure.wav"
    source.write_bytes(b"original")
    cleaner = cleaning.ExerciseAudioCleaner(
        _options(tmp_path, keep_intermediate_files=True)
    )
    _install_fake_pipeline(monkeypatch, cleaner)
    monkeypatch.setattr(
        cleaner,
        "_run_deepfilternet",
        lambda *_args: (_ for _ in ()).throw(
            cleaning.AudioCleaningError(
                "deepfilternet_processing_failed",
                "Noise reduction failed for this recording.",
                technical_message="private-model-path-and-trace",
            )
        ),
    )
    with pytest.raises(cleaning.AudioCleaningError):
        cleaner.process(source)
    manifests = list((tmp_path / "outputs").glob("*_audio_cleaning/result.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["processing_status"] == "failed"
    assert manifest["error_code"] == "deepfilternet_processing_failed"
    assert "private-model-path" not in json.dumps(manifest)


def test_no_speech_detected_is_metadata_not_pipeline_failure(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"original")
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))
    _install_fake_pipeline(monkeypatch, cleaner, segments=[])

    result = cleaner.process(source)

    assert result.metadata["processing_status"] == "completed"
    assert result.metadata["scoring_allowed"] is False
    assert result.metadata["rejection_reasons"] == ["no_speech_detected"]


def test_recording_too_short_and_insufficient_speech(tmp_path, monkeypatch):
    source = tmp_path / "short.wav"
    source.write_bytes(b"original")
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))
    _install_fake_pipeline(
        monkeypatch,
        cleaner,
        duration=0.4,
        segments=[{"start": 0.05, "end": 0.2}],
    )

    result = cleaner.process(source)

    assert result.metadata["scoring_allowed"] is False
    assert result.metadata["rejection_reasons"] == [
        "recording_too_short",
        "insufficient_speech",
    ]


def test_severe_clipping_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "clipped.wav"
    source.write_bytes(b"original")
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))
    _install_fake_pipeline(monkeypatch, cleaner, amplitude=1.0)

    result = cleaner.process(source)

    assert result.metadata["clipping_ratio"] > 0.99
    assert "severe_clipping" in result.metadata["rejection_reasons"]
    assert result.metadata["scoring_allowed"] is False


def test_low_speech_ratio_is_not_rejected_by_default(tmp_path, monkeypatch):
    source = tmp_path / "pause.wav"
    source.write_bytes(b"original")
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))
    _install_fake_pipeline(
        monkeypatch,
        cleaner,
        duration=4.0,
        segments=[{"start": 0.1, "end": 0.5}],
    )
    result = cleaner.process(source)
    assert result.metadata["speech_ratio"] == 0.1
    assert result.metadata["scoring_allowed"] is True


def test_json_safe_converts_numpy_and_torch_scalars():
    torch = pytest.importorskip("torch")
    value = {
        "bool": np.bool_(True),
        "float": np.float32(1.25),
        "int": np.int64(7),
        "tensor": torch.tensor(3.5),
    }
    converted = cleaning.json_safe(value)
    assert converted == {"bool": True, "float": 1.25, "int": 7, "tensor": 3.5}
    assert all(type(item) in {bool, float, int} for item in converted.values())


def test_gpu_request_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cleaning._select_device(True) == "cpu"


def test_deepfilternet_io_uses_soundfile_with_current_torchaudio(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    source = tmp_path / "source_48k.wav"
    output = tmp_path / "cleaned_48k.wav"
    samples = np.linspace(-0.25, 0.25, 4_800, dtype=np.float32)
    sf.write(str(source), samples, 48_000, subtype="PCM_16")
    observed = {}

    def fake_enhance(model, state, waveform, *, pad):
        observed.update(
            model=model,
            state=state,
            shape=tuple(waveform.shape),
            device=str(waveform.device),
            pad=pad,
        )
        return waveform * 0.5

    monkeypatch.setattr(
        cleaning._LazyModels,
        "deepfilter",
        lambda _device, **_kwargs: ("model", "state", fake_enhance),
    )
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))
    cleaner._run_deepfilternet(source, output, "cpu")

    info = sf.info(str(output))
    written, _ = sf.read(str(output), dtype="float32")
    assert observed == {
        "model": "model",
        "state": "state",
        "shape": (1, 4_800),
        "device": "cpu",
        "pad": True,
    }
    assert info.samplerate == 48_000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert np.max(np.abs(written)) <= 0.126


def test_deepfilternet_device_is_set_during_and_after_initialization(monkeypatch):
    monkeypatch.setenv("DEVICE", "existing-device")
    observed = {}

    class FakeModel:
        def to(self, device):
            observed["model_device"] = device
            return self

    class FakeConfig:
        def set(self, option, value, cast, section):
            observed["config_set"] = (option, value, cast, section)

    def fake_init(**kwargs):
        observed["init_device"] = os.environ.get("DEVICE")
        observed["init_kwargs"] = kwargs
        return FakeModel(), "state", "suffix"

    model, state = cleaning._initialize_deepfilter(fake_init, FakeConfig(), "cpu")

    assert isinstance(model, FakeModel)
    assert state == "state"
    assert observed["init_device"] == "cpu"
    assert observed["model_device"] == "cpu"
    assert observed["config_set"] == ("DEVICE", "cpu", str, "train")
    assert observed["init_kwargs"]["log_file"] is None
    assert os.environ["DEVICE"] == "existing-device"


def test_silero_vad_io_uses_soundfile_with_current_torchaudio(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    source = tmp_path / "cleaned_16k.wav"
    sf.write(str(source), np.zeros(16_000, dtype=np.float32), 16_000, subtype="PCM_16")
    observed = {}

    def fake_timestamps(waveform, model, *, sampling_rate, return_seconds):
        assert isinstance(waveform, torch.Tensor)
        observed.update(
            shape=tuple(waveform.shape),
            model=model,
            sampling_rate=sampling_rate,
            return_seconds=return_seconds,
        )
        return [{"start": 0.12345, "end": 0.98765}]

    monkeypatch.setattr(
        cleaning._LazyModels,
        "silero",
        lambda _device: ("model", fake_timestamps),
    )
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))

    segments = cleaner._run_silero_vad(source, "cpu")

    assert observed == {
        "shape": (16_000,),
        "model": "model",
        "sampling_rate": 16_000,
        "return_seconds": True,
    }
    assert segments == [{"start": 0.123, "end": 0.988}]


def test_completed_processing_is_idempotently_reused(tmp_path, monkeypatch):
    source = tmp_path / "same.wav"
    source.write_bytes(b"original")
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))
    observed = _install_fake_pipeline(monkeypatch, cleaner)

    first = cleaner.process(source, recording_id="stable")
    second = cleaner.process(source, recording_id="stable")

    assert first.reused is False
    assert second.reused is True
    assert observed["deepfilter_calls"] == 1
    assert first.cleaned_audio_16k_path == second.cleaned_audio_16k_path


def test_force_reruns_completed_processing(tmp_path, monkeypatch):
    source = tmp_path / "same.wav"
    source.write_bytes(b"original")
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))
    observed = _install_fake_pipeline(monkeypatch, cleaner)
    cleaner.process(source)
    cleaner.process(source, force=True)
    assert observed["deepfilter_calls"] == 2


def test_changed_completed_source_requires_force(tmp_path, monkeypatch):
    source = tmp_path / "same.wav"
    source.write_bytes(b"original")
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))
    _install_fake_pipeline(monkeypatch, cleaner)
    cleaner.process(source, recording_id="stable")
    source.write_bytes(b"changed")

    with pytest.raises(cleaning.AudioCleaningError) as caught:
        cleaner.process(source, recording_id="stable")
    assert caught.value.code == "source_changed_requires_force"


def test_concurrent_processing_of_same_recording_is_guarded(tmp_path, monkeypatch):
    source = tmp_path / "busy.wav"
    source.write_bytes(b"original")
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path, timeout_seconds=0.01))

    class BusyLock:
        def acquire(self, timeout):
            assert timeout == 1.0
            return False

        def release(self):
            pytest.fail("an unacquired lock must not be released")

    monkeypatch.setattr(cleaning, "_processing_lock", lambda _key: BusyLock())
    with pytest.raises(cleaning.AudioCleaningError) as caught:
        cleaner.process(source)
    assert caught.value.code == "processing_busy"
    assert caught.value.status_code == 409


def test_disabled_feature_does_not_load_models(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"original")
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path, enabled=False))
    monkeypatch.setattr(
        cleaning._LazyModels,
        "deepfilter",
        lambda *_args: pytest.fail("DeepFilterNet must remain lazy"),
    )
    monkeypatch.setattr(
        cleaning._LazyModels,
        "silero",
        lambda *_args: pytest.fail("Silero must remain lazy"),
    )
    with pytest.raises(cleaning.AudioCleaningError) as caught:
        cleaner.process(source)
    assert caught.value.code == "audio_cleaning_disabled"


def test_deepfilternet_failure_is_safe_and_preserves_original(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"original")
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))
    _install_fake_pipeline(monkeypatch, cleaner)

    def fail_model(*_args):
        raise cleaning.AudioCleaningError(
            "deepfilternet_processing_failed",
            "Noise reduction failed for this recording.",
            technical_message="C:/private/path/model-error",
            status_code=502,
        )

    monkeypatch.setattr(cleaner, "_run_deepfilternet", fail_model)
    with pytest.raises(cleaning.AudioCleaningError) as caught:
        cleaner.process(source)
    assert str(tmp_path) not in caught.value.user_message
    assert source.read_bytes() == b"original"


def test_silero_failure_is_reported(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"original")
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path))
    _install_fake_pipeline(monkeypatch, cleaner)
    monkeypatch.setattr(
        cleaner,
        "_run_silero_vad",
        lambda *_args: (_ for _ in ()).throw(
            cleaning.AudioCleaningError(
                "silero_vad_failed",
                "Speech detection failed for this recording.",
                status_code=502,
            )
        ),
    )
    with pytest.raises(cleaning.AudioCleaningError) as caught:
        cleaner.process(source)
    assert caught.value.code == "silero_vad_failed"


def test_ffmpeg_subprocess_command_generates_pcm_wav(tmp_path, monkeypatch):
    source = tmp_path / "source.webm"
    source.write_bytes(b"input")
    output = tmp_path / "output.wav"
    observed = {}
    monkeypatch.setattr(cleaning, "find_ffmpeg", lambda _configured=None: "ffmpeg-test")

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        sf.write(str(output), np.zeros(48_000, dtype=np.float32), 48_000, subtype="PCM_16")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cleaning.subprocess, "run", fake_run)
    cleaning.convert_with_ffmpeg(source, output, sample_rate=48_000, timeout_seconds=9)
    assert "pcm_s16le" in observed["command"]
    assert "48000" in observed["command"]
    assert observed["kwargs"]["timeout"] == 9


def test_ffmpeg_timeout_has_stable_error(tmp_path, monkeypatch):
    source = tmp_path / "source.webm"
    source.write_bytes(b"input")
    monkeypatch.setattr(cleaning, "find_ffmpeg", lambda _configured=None: "ffmpeg-test")
    monkeypatch.setattr(
        cleaning.subprocess,
        "run",
        lambda command, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(command, 1)
        ),
    )
    with pytest.raises(cleaning.AudioCleaningError) as caught:
        cleaning.convert_with_ffmpeg(
            source,
            tmp_path / "output.wav",
            sample_rate=48_000,
            timeout_seconds=1,
        )
    assert caught.value.code == "ffmpeg_timeout"


def test_unrelated_request_does_not_load_audio_models(client, monkeypatch):
    monkeypatch.setattr(
        cleaning._LazyModels,
        "deepfilter",
        lambda *_args: pytest.fail("unrelated request loaded DeepFilterNet"),
    )
    monkeypatch.setattr(
        cleaning._LazyModels,
        "silero",
        lambda *_args: pytest.fail("unrelated request loaded Silero"),
    )
    assert client.get("/health/live").status_code == 200


def test_audio_cleaning_environment_configuration(monkeypatch, tmp_path):
    from api.config import Settings

    monkeypatch.setenv("AUDIO_CLEANING_ENABLED", "1")
    monkeypatch.setenv("AUDIO_CLEANING_USE_GPU", "1")
    monkeypatch.setenv("AUDIO_CLEANING_MIN_DURATION_SECONDS", "0.75")
    monkeypatch.setenv("AUDIO_CLEANING_MIN_SPEECH_SECONDS", "0.4")
    monkeypatch.setenv("AUDIO_CLEANING_MAX_CLIPPING_RATIO", "0.02")
    monkeypatch.setenv("AUDIO_CLEANING_KEEP_INTERMEDIATE_FILES", "1")
    monkeypatch.setenv("AUDIO_CLEANING_TIMEOUT_SECONDS", "33")
    monkeypatch.setenv("AUDIO_CLEANING_OUTPUT_DIR", str(tmp_path))
    configured = Settings()
    assert configured.audio_cleaning_enabled is True
    assert configured.audio_cleaning_use_gpu is True
    assert configured.audio_cleaning_min_duration_seconds == 0.75
    assert configured.audio_cleaning_min_speech_seconds == 0.4
    assert configured.audio_cleaning_max_clipping_ratio == 0.02
    assert configured.audio_cleaning_keep_intermediate_files is True
    assert configured.audio_cleaning_timeout_seconds == 33
    assert configured.audio_cleaning_output_dir == tmp_path


def test_manual_cli_prints_json_result(tmp_path, monkeypatch, capsys):
    from scripts import clean_exercise_audio as cli

    source = tmp_path / "source.wav"
    source.write_bytes(b"original")
    processing = tmp_path / "result"
    processing.mkdir()
    result = cleaning.AudioCleaningResult(
        recording_id="manual",
        original_audio_path=source,
        normalized_audio_48k_path=processing / "source_48k_mono.wav",
        cleaned_audio_48k_path=processing / "cleaned_48k_mono.wav",
        cleaned_audio_16k_path=processing / "cleaned_16k_mono.wav",
        processing_directory=processing,
        metadata={
            "duration_seconds": 1.0,
            "speech_seconds": 0.8,
            "speech_ratio": 0.8,
            "clipping_ratio": 0.0,
            "scoring_allowed": True,
            "rejection_reasons": [],
        },
    )

    class FakeCleaner:
        def process(self, *_args, **_kwargs):
            return result

    monkeypatch.setattr(cli, "cleaner_from_settings", lambda _settings: FakeCleaner())
    code = cli.main([str(source), "--output-dir", str(tmp_path / "output")])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["summary"]["speech_seconds"] == 0.8
    assert payload["files"]["cleaned_48k"].endswith("cleaned_48k_mono.wav")


def test_analysis_uses_original_signal_for_quality_and_clean_16k_for_stt(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import librosa

    from api import acoustic, analysis as analysis_mod
    from src.core.audio import audio_quality
    from src.core.audio.audio_quality import AudioQualityDecision

    source = tmp_path / "source.webm"
    normalized = tmp_path / "source_48k.wav"
    cleaned_16k = tmp_path / "cleaned_16k.wav"
    source.write_bytes(b"original")
    normalized.write_bytes(b"normalized")
    cleaned_16k.write_bytes(b"cleaned")
    loaded = []
    transcribed = []
    monkeypatch.setattr(
        analysis_mod,
        "settings",
        SimpleNamespace(audio_cleaning_enabled=True),
    )
    monkeypatch.setattr(
        analysis_mod,
        "clean_recording",
        lambda _path: {
            "original_audio_path": source,
            "normalized_audio_48k_path": normalized,
            "cleaned_audio_48k_path": tmp_path / "cleaned_48k.wav",
            "cleaned_audio_16k_path": cleaned_16k,
            "reduced_audio_path": cleaned_16k,
            "noise_reduction_applied": True,
            "preprocessing_pipeline": "ffmpeg_deepfilternet_silero_vad",
            "audio_processing": {
                "processing_status": "completed",
                "scoring_allowed": True,
                "rejection_reasons": [],
            },
            "cleanup_paths": [],
        },
    )
    monkeypatch.setattr(
        librosa,
        "load",
        lambda path, **_kwargs: (
            loaded.append(Path(path)) or np.ones(16_000, dtype=np.float32),
            16_000,
        ),
    )
    monkeypatch.setattr(
        audio_quality,
        "analyze_audio_quality",
        lambda *_args: AudioQualityDecision(True, 1.0, [], {}),
    )
    monkeypatch.setattr(
        acoustic,
        "transcribe",
        lambda path: transcribed.append(Path(path)) or "s",
    )

    result = analysis_mod.process_recording(source)

    assert loaded == [normalized]
    assert transcribed == [cleaned_16k]
    assert result["original_audio_path"] == source


@pytest.mark.skipif(
    os.environ.get("RUN_AUDIO_MODEL_TESTS") != "1",
    reason="set RUN_AUDIO_MODEL_TESTS=1 to download/load real audio models",
)
def test_real_pipeline_smoke(sample_wav, tmp_path):
    source = tmp_path / "speech.wav"
    source.write_bytes(sample_wav)
    cleaner = cleaning.ExerciseAudioCleaner(_options(tmp_path, timeout_seconds=300))
    result = cleaner.process(source, force=True)
    assert result.cleaned_audio_48k_path.is_file()
    assert result.cleaned_audio_16k_path.is_file()
    assert isinstance(result.metadata["speech_segments"], list)
