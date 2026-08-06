"""Offline pronunciation-recording cleanup with DeepFilterNet and Silero VAD.

The input recording is immutable. Each recording gets an isolated processing
directory containing an FFmpeg-normalized source, cleaned 48 kHz and 16 kHz
PCM WAV files, and a JSON manifest. Heavy models are initialized lazily and
reused; importing this module never downloads or loads a model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import types
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ORIGINAL_SAMPLE_RATE = 48_000
STT_SAMPLE_RATE = 16_000
PCM_SUBTYPE = "PCM_16"
MAX_MODEL_DOWNLOAD_BYTES = 128 * 1024 * 1024


class AudioCleaningError(RuntimeError):
    """Internal failure with a stable code and a separate user-safe message."""

    def __init__(
        self,
        code: str,
        user_message: str,
        *,
        technical_message: Optional[str] = None,
        status_code: int = 500,
    ) -> None:
        super().__init__(technical_message or user_message)
        self.code = code
        self.user_message = user_message
        self.technical_message = technical_message or user_message
        self.status_code = status_code


@dataclass(frozen=True)
class AudioCleaningOptions:
    enabled: bool = False
    use_gpu: bool = False
    min_duration_seconds: float = 0.5
    min_speech_seconds: float = 0.3
    max_clipping_ratio: float = 0.01
    min_speech_ratio: Optional[float] = None
    keep_intermediate_files: bool = False
    timeout_seconds: float = 120.0
    output_root: Optional[Path] = None


@dataclass(frozen=True)
class AudioCleaningResult:
    recording_id: str
    original_audio_path: Path
    normalized_audio_48k_path: Path
    cleaned_audio_48k_path: Path
    cleaned_audio_16k_path: Path
    processing_directory: Path
    metadata: Dict[str, Any]
    reused: bool = False

    def to_dict(self, *, include_paths: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "recording_id": self.recording_id,
            "metadata": json_safe(self.metadata),
            "reused": bool(self.reused),
        }
        if include_paths:
            payload["files"] = {
                "original": str(self.original_audio_path),
                "normalized_48k": str(self.normalized_audio_48k_path),
                "cleaned_48k": str(self.cleaned_audio_48k_path),
                "cleaned_16k": str(self.cleaned_audio_16k_path),
                "processing_directory": str(self.processing_directory),
            }
        return payload


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy/Torch/dataclass values to JSON primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if hasattr(value, "detach") and hasattr(value, "numel"):
        detached = value.detach().cpu()
        if int(detached.numel()) == 1:
            return json_safe(detached.item())
        return json_safe(detached.tolist())
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def find_ffmpeg(configured_binary: Optional[str] = None) -> str:
    """Return an external FFmpeg executable or raise a helpful safe error."""
    configured = (configured_binary or os.environ.get("FFMPEG_BINARY", "")).strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate)
        found = shutil.which(configured)
        if found:
            return found
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise AudioCleaningError(
        "ffmpeg_unavailable",
        "Audio processing is unavailable because FFmpeg is not installed.",
        technical_message="FFmpeg was not found in FFMPEG_BINARY or PATH.",
        status_code=503,
    )


def convert_with_ffmpeg(
    source: Path,
    output: Path,
    *,
    sample_rate: int,
    timeout_seconds: float,
    ffmpeg_binary: Optional[str] = None,
) -> Path:
    """Create mono PCM-16 WAV at ``sample_rate`` without modifying ``source``."""
    source = Path(source)
    output = Path(output)
    if not source.is_file():
        raise AudioCleaningError(
            "missing_input_file",
            "The uploaded recording could not be found.",
            status_code=400,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        find_ffmpeg(ffmpeg_binary),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=max(1.0, float(timeout_seconds)),
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioCleaningError(
            "ffmpeg_timeout",
            "The recording took too long to decode.",
            technical_message=f"FFmpeg exceeded {timeout_seconds:.1f}s.",
            status_code=504,
        ) from exc
    except (subprocess.CalledProcessError, OSError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise AudioCleaningError(
            "unsupported_or_corrupt_audio",
            "The recording is corrupt or uses an unsupported audio format.",
            technical_message=f"FFmpeg conversion failed: {stderr[-1000:]}",
            status_code=400,
        ) from exc
    _validate_wav(output, sample_rate)
    return output


def _validate_wav(path: Path, expected_sample_rate: int) -> None:
    if not path.is_file() or path.stat().st_size <= 44:
        raise AudioCleaningError(
            "invalid_generated_output",
            "Audio processing did not produce a valid recording.",
            technical_message="Generated WAV is missing or empty.",
        )
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if (
            int(info.samplerate) != int(expected_sample_rate)
            or int(info.channels) != 1
            or int(info.frames) <= 0
            or str(info.subtype) != PCM_SUBTYPE
        ):
            raise ValueError(
                f"expected mono/{expected_sample_rate}, got "
                f"{info.channels}/{info.samplerate}/{info.frames}/{info.subtype}"
            )
    except AudioCleaningError:
        raise
    except Exception as exc:
        raise AudioCleaningError(
            "invalid_generated_output",
            "Audio processing did not produce a valid recording.",
            technical_message=f"Generated WAV validation failed: {exc!r}",
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_device(use_gpu: bool) -> str:
    if not use_gpu:
        return "cpu"
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _install_deepfilter_torchaudio_compat() -> None:
    """Provide the metadata import removed by torchaudio 2.9+.

    DeepFilterNet 0.5.6 imports ``torchaudio.backend.common.AudioMetaData``
    while loading its enhancement module. The service deliberately does not
    use DeepFilterNet's torchaudio-based I/O, but the import still has to
    succeed with current torchaudio releases.
    """
    try:
        from torchaudio.backend.common import AudioMetaData  # noqa: F401

        return
    except (ImportError, ModuleNotFoundError):
        import torchaudio

    class AudioMetaData(NamedTuple):
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    backend_module = sys.modules.get("torchaudio.backend")
    if backend_module is None:
        backend_module = types.ModuleType("torchaudio.backend")
        sys.modules["torchaudio.backend"] = backend_module
    common_module = sys.modules.get("torchaudio.backend.common")
    if common_module is None:
        common_module = types.ModuleType("torchaudio.backend.common")
        sys.modules["torchaudio.backend.common"] = common_module
    common_module.AudioMetaData = AudioMetaData
    backend_module.common = common_module
    if not hasattr(torchaudio, "backend"):
        torchaudio.backend = backend_module


def _initialize_deepfilter(
    init_df: Callable[..., Tuple[Any, Any, str]],
    df_config: Any,
    device: str,
) -> Tuple[Any, Any]:
    """Initialize DeepFilterNet with a device before its config exists."""
    previous_device = os.environ.get("DEVICE")
    os.environ["DEVICE"] = device
    try:
        model, state, *_ = init_df(
            post_filter=False,
            log_level="ERROR",
            log_file=None,
            config_allow_defaults=True,
        )
    finally:
        if previous_device is None:
            os.environ.pop("DEVICE", None)
        else:
            os.environ["DEVICE"] = previous_device

    # init_df has now loaded the model config. Persist the selected device for
    # enhance(), whose internal get_device() calls happen after env restoration.
    df_config.set("DEVICE", device, str, "train")
    if hasattr(model, "to"):
        model = model.to(device)
    return model, state


def _download_deepfilter_file(
    url: str,
    download_dir: str,
    *,
    extract: bool,
    timeout_seconds: float,
) -> str:
    """Download the official model with hard size/time and safe-ZIP limits."""
    import requests

    # Avoid an extra GitHub HTML redirect, which can stall indefinitely in
    # some Windows/Conda proxy configurations.
    prefix = "https://github.com/Rikorose/DeepFilterNet/raw/"
    if url.startswith(prefix):
        url = "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/" + url[len(prefix) :]
    destination_dir = Path(download_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / url.rsplit("/", 1)[-1]
    # Keep one stable partial file so a later request can resume after a
    # bounded timeout instead of restarting a slow model download at byte 0.
    partial = destination_dir / f".{destination.name}.part"
    started = time.monotonic()
    resume_at = partial.stat().st_size if partial.is_file() else 0
    downloaded = resume_at
    timeout = max(5.0, float(timeout_seconds))
    completed = False
    try:
        headers = {"Range": f"bytes={resume_at}-"} if resume_at else None
        with requests.get(
            url,
            stream=True,
            headers=headers,
            timeout=(min(10.0, timeout), timeout),
        ) as response:
            response.raise_for_status()
            resumed = resume_at > 0 and response.status_code == 206
            if not resumed:
                resume_at = 0
                downloaded = 0
            declared_size = int(response.headers.get("content-length", "0") or 0)
            if resume_at + declared_size > MAX_MODEL_DOWNLOAD_BYTES:
                raise ValueError("DeepFilterNet model archive exceeds the size limit")
            with partial.open("ab" if resumed else "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > MAX_MODEL_DOWNLOAD_BYTES:
                        raise ValueError("DeepFilterNet model archive exceeds the size limit")
                    if time.monotonic() - started > timeout:
                        raise TimeoutError("DeepFilterNet model download exceeded its timeout")
                    handle.write(chunk)
        if downloaded == 0:
            raise ValueError("DeepFilterNet model download was empty")
        if extract:
            with zipfile.ZipFile(partial) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise ValueError(f"Corrupt DeepFilterNet archive member: {bad_member}")
                root = str(destination_dir)
                for member in archive.infolist():
                    target = str((destination_dir / member.filename).resolve())
                    if os.path.commonpath((root, target)) != root:
                        raise ValueError("Unsafe path in DeepFilterNet model archive")
                    if stat.S_ISLNK(member.external_attr >> 16):
                        raise ValueError("Symlinks are not allowed in the model archive")
                archive.extractall(destination_dir)
            destination.unlink(missing_ok=True)
        else:
            os.replace(partial, destination)
        completed = True
        return str(destination)
    except (ValueError, zipfile.BadZipFile):
        partial.unlink(missing_ok=True)
        raise
    finally:
        if completed:
            partial.unlink(missing_ok=True)


class _LazyModels:
    """Process-wide lazy model cache with serialized initialization/inference."""

    _deepfilter_lock = threading.RLock()
    _deepfilter_inference_lock = threading.Lock()
    _deepfilter: Dict[str, Tuple[Any, Any, Callable[..., Any]]] = {}
    _silero_lock = threading.RLock()
    _silero_inference_lock = threading.Lock()
    _silero: Dict[str, Tuple[Any, Callable[..., Any]]] = {}

    @classmethod
    def deepfilter(cls, device: str, *, timeout_seconds: float = 120.0):
        with cls._deepfilter_lock:
            cached = cls._deepfilter.get(device)
            if cached is not None:
                return cached
            try:
                _install_deepfilter_torchaudio_compat()
                import importlib

                from df.config import config as df_config
                from df.enhance import enhance, init_df

                enhance_module = importlib.import_module("df.enhance")
                enhance_module.download_file = lambda url, directory, extract=False: (
                    _download_deepfilter_file(
                        url,
                        directory,
                        extract=extract,
                        timeout_seconds=timeout_seconds,
                    )
                )
                model, state = _initialize_deepfilter(init_df, df_config, device)
                cached = (model, state, enhance)
                cls._deepfilter[device] = cached
                return cached
            except (Exception, SystemExit) as exc:
                raise AudioCleaningError(
                    "deepfilternet_initialization_failed",
                    "The noise-reduction model is temporarily unavailable.",
                    technical_message=f"DeepFilterNet initialization failed: {exc!r}",
                    status_code=503,
                ) from exc

    @classmethod
    def silero(cls, device: str):
        # Silero is intentionally kept on CPU; it is tiny and this avoids model
        # contention with DeepFilterNet/Wav2Vec2 on small GPUs.
        cache_key = "cpu"
        with cls._silero_lock:
            cached = cls._silero.get(cache_key)
            if cached is not None:
                return cached
            try:
                from silero_vad import (
                    get_speech_timestamps,
                    load_silero_vad,
                )

                model = load_silero_vad()
                if hasattr(model, "to"):
                    model = model.to("cpu")
                cached = (model, get_speech_timestamps)
                cls._silero[cache_key] = cached
                return cached
            except (Exception, SystemExit) as exc:
                raise AudioCleaningError(
                    "silero_vad_initialization_failed",
                    "Speech detection is temporarily unavailable.",
                    technical_message=f"Silero VAD initialization failed: {exc!r}",
                    status_code=503,
                ) from exc

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._deepfilter_lock, cls._silero_lock:
            cls._deepfilter.clear()
            cls._silero.clear()


_LOCKS_GUARD = threading.Lock()
_PROCESSING_LOCKS: Dict[str, threading.Lock] = {}


def _processing_lock(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _PROCESSING_LOCKS.setdefault(key, threading.Lock())


def _safe_recording_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")[:80]
    return cleaned or hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


class ExerciseAudioCleaner:
    """Reusable, idempotent offline audio-cleaning pipeline."""

    def __init__(self, options: AudioCleaningOptions) -> None:
        self.options = options

    def process(
        self,
        input_path: Path,
        *,
        output_dir: Optional[Path] = None,
        recording_id: Optional[str] = None,
        force: bool = False,
    ) -> AudioCleaningResult:
        source = Path(input_path).expanduser().resolve()
        if not self.options.enabled:
            raise AudioCleaningError(
                "audio_cleaning_disabled",
                "Audio cleaning is disabled on this service.",
                status_code=503,
            )
        if not source.is_file():
            raise AudioCleaningError(
                "missing_input_file",
                "The uploaded recording could not be found.",
                status_code=400,
            )

        default_identifier = (
            f"{source.stem}-"
            f"{hashlib.sha256(str(source).casefold().encode('utf-8')).hexdigest()[:12]}"
        )
        identifier = _safe_recording_id(recording_id or default_identifier)
        root = Path(output_dir or self.options.output_root or source.parent).expanduser().resolve()
        processing_dir = root / f"{identifier}_audio_cleaning"
        lock = _processing_lock(str(processing_dir).casefold())
        acquired = lock.acquire(timeout=max(1.0, self.options.timeout_seconds))
        if not acquired:
            raise AudioCleaningError(
                "processing_busy",
                "This recording is already being processed.",
                status_code=409,
            )
        try:
            return self._process_locked(source, processing_dir, identifier, force)
        finally:
            lock.release()

    def _process_locked(
        self,
        source: Path,
        processing_dir: Path,
        recording_id: str,
        force: bool,
    ) -> AudioCleaningResult:
        normalized_48k = processing_dir / "source_48k_mono.wav"
        cleaned_48k = processing_dir / "cleaned_48k_mono.wav"
        cleaned_16k = processing_dir / "cleaned_16k_mono.wav"
        manifest_path = processing_dir / "result.json"

        if not force:
            reused = self._read_completed_manifest(
                manifest_path,
                source,
                normalized_48k,
                cleaned_48k,
                cleaned_16k,
                processing_dir,
                recording_id,
            )
            if reused is not None:
                return reused

        try:
            if processing_dir.exists():
                shutil.rmtree(processing_dir)
            processing_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise AudioCleaningError(
                "processing_directory_unavailable",
                "Temporary audio storage is unavailable.",
                technical_message=f"Could not create processing directory: {exc!r}",
                status_code=503,
            ) from exc
        started = time.monotonic()
        device = _select_device(self.options.use_gpu)
        try:
            source_hash = _sha256(source)
            self._check_disk_space(source, processing_dir)
            self._write_manifest(
                manifest_path,
                {
                    "recording_id": recording_id,
                    "processing_status": "pending",
                    "source_sha256": source_hash,
                },
            )
            logger.info(
                "audio_cleaning_started",
                extra={"recording_id": recording_id, "device": device},
            )
            remaining = self._remaining_timeout(started)
            convert_with_ffmpeg(
                source,
                normalized_48k,
                sample_rate=ORIGINAL_SAMPLE_RATE,
                timeout_seconds=remaining,
            )
            self._ensure_original_unchanged(source, source_hash)
            self._run_deepfilternet(normalized_48k, cleaned_48k, device)
            self._remaining_timeout(started)
            convert_with_ffmpeg(
                cleaned_48k,
                cleaned_16k,
                sample_rate=STT_SAMPLE_RATE,
                timeout_seconds=self._remaining_timeout(started),
            )
            speech_segments = self._run_silero_vad(cleaned_16k, device)
            metadata = self._quality_metadata(normalized_48k, speech_segments, device)
            self._ensure_original_unchanged(source, source_hash)
            self._write_manifest(
                manifest_path,
                {
                    "recording_id": recording_id,
                    "processing_status": "completed",
                    "source_sha256": source_hash,
                    "files": {
                        "normalized_48k": normalized_48k.name,
                        "cleaned_48k": cleaned_48k.name,
                        "cleaned_16k": cleaned_16k.name,
                    },
                    "metadata": metadata,
                },
            )
            logger.info(
                "audio_cleaning_completed",
                extra={
                    "recording_id": recording_id,
                    "device": device,
                    "scoring_allowed": metadata["scoring_allowed"],
                },
            )
            return AudioCleaningResult(
                recording_id=recording_id,
                original_audio_path=source,
                normalized_audio_48k_path=normalized_48k,
                cleaned_audio_48k_path=cleaned_48k,
                cleaned_audio_16k_path=cleaned_16k,
                processing_directory=processing_dir,
                metadata=metadata,
            )
        except AudioCleaningError as exc:
            self._record_failure(manifest_path, recording_id, exc)
            logger.error(
                "audio_cleaning_failed",
                extra={"recording_id": recording_id, "error_code": exc.code},
                exc_info=True,
            )
            if not self.options.keep_intermediate_files:
                shutil.rmtree(processing_dir, ignore_errors=True)
            raise
        except Exception as exc:
            wrapped = AudioCleaningError(
                "processing_failed",
                "The recording could not be cleaned. Please try again.",
                technical_message=f"Unexpected audio-cleaning failure: {exc!r}",
            )
            self._record_failure(manifest_path, recording_id, wrapped)
            logger.error(
                "audio_cleaning_failed",
                extra={"recording_id": recording_id, "error_code": wrapped.code},
                exc_info=True,
            )
            if not self.options.keep_intermediate_files:
                shutil.rmtree(processing_dir, ignore_errors=True)
            raise wrapped from exc

    def _read_completed_manifest(
        self,
        manifest_path: Path,
        source: Path,
        normalized_48k: Path,
        cleaned_48k: Path,
        cleaned_16k: Path,
        processing_dir: Path,
        recording_id: str,
    ) -> Optional[AudioCleaningResult]:
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("processing_status") != "completed":
                return None
            if manifest.get("source_sha256") != _sha256(source):
                raise AudioCleaningError(
                    "source_changed_requires_force",
                    "This completed recording changed; rerun explicitly with force.",
                    status_code=409,
                )
            _validate_wav(normalized_48k, ORIGINAL_SAMPLE_RATE)
            _validate_wav(cleaned_48k, ORIGINAL_SAMPLE_RATE)
            _validate_wav(cleaned_16k, STT_SAMPLE_RATE)
            return AudioCleaningResult(
                recording_id=recording_id,
                original_audio_path=source,
                normalized_audio_48k_path=normalized_48k,
                cleaned_audio_48k_path=cleaned_48k,
                cleaned_audio_16k_path=cleaned_16k,
                processing_directory=processing_dir,
                metadata=json_safe(manifest["metadata"]),
                reused=True,
            )
        except AudioCleaningError:
            raise
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _run_deepfilternet(self, source_48k: Path, cleaned_48k: Path, device: str) -> None:
        try:
            import soundfile as sf
            import torch

            model, state, enhance = _LazyModels.deepfilter(
                device,
                timeout_seconds=self.options.timeout_seconds,
            )
            with _LazyModels._deepfilter_inference_lock:
                samples, sample_rate = sf.read(
                    str(source_48k),
                    dtype="float32",
                    always_2d=True,
                )
                if int(sample_rate) != ORIGINAL_SAMPLE_RATE or samples.shape[1] != 1:
                    raise ValueError("DeepFilterNet input must be mono 48 kHz audio")
                audio = torch.from_numpy(np.ascontiguousarray(samples.T)).to(device)
                enhanced = enhance(model, state, audio, pad=True)
                output = enhanced.detach().to("cpu", dtype=torch.float32).numpy()
                if output.ndim == 2:
                    output = output.T
                output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
                sf.write(
                    str(cleaned_48k),
                    output,
                    ORIGINAL_SAMPLE_RATE,
                    subtype=PCM_SUBTYPE,
                )
            _validate_wav(cleaned_48k, ORIGINAL_SAMPLE_RATE)
        except AudioCleaningError:
            raise
        except (Exception, SystemExit) as exc:
            raise AudioCleaningError(
                "deepfilternet_processing_failed",
                "Noise reduction failed for this recording.",
                technical_message=f"DeepFilterNet enhancement failed: {exc!r}",
                status_code=502,
            ) from exc

    def _run_silero_vad(self, cleaned_16k: Path, device: str) -> List[Dict[str, float]]:
        try:
            import soundfile as sf
            import torch

            model, get_speech_timestamps = _LazyModels.silero(device)
            with _LazyModels._silero_inference_lock:
                samples, sample_rate = sf.read(
                    str(cleaned_16k),
                    dtype="float32",
                    always_2d=False,
                )
                if int(sample_rate) != STT_SAMPLE_RATE or np.asarray(samples).ndim != 1:
                    raise ValueError("Silero VAD input must be mono 16 kHz audio")
                waveform = torch.from_numpy(np.ascontiguousarray(samples))
                raw_segments = get_speech_timestamps(
                    waveform,
                    model,
                    sampling_rate=STT_SAMPLE_RATE,
                    return_seconds=True,
                )
            return [
                {
                    "start": round(float(segment["start"]), 3),
                    "end": round(float(segment["end"]), 3),
                }
                for segment in raw_segments
                if float(segment["end"]) >= float(segment["start"])
            ]
        except AudioCleaningError:
            raise
        except (Exception, SystemExit) as exc:
            raise AudioCleaningError(
                "silero_vad_failed",
                "Speech detection failed for this recording.",
                technical_message=f"Silero VAD failed: {exc!r}",
                status_code=502,
            ) from exc

    def _quality_metadata(
        self,
        normalized_48k: Path,
        speech_segments: Sequence[Mapping[str, float]],
        device: str,
    ) -> Dict[str, Any]:
        try:
            import soundfile as sf

            audio, sample_rate = sf.read(str(normalized_48k), dtype="float32", always_2d=False)
        except Exception as exc:
            raise AudioCleaningError(
                "audio_measurement_failed",
                "Audio quality could not be measured.",
                technical_message=f"soundfile could not read normalized WAV: {exc!r}",
            ) from exc
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if samples.size == 0 or int(sample_rate) <= 0:
            raise AudioCleaningError(
                "invalid_generated_output",
                "Audio processing did not produce a valid recording.",
            )

        duration = float(samples.size / float(sample_rate))
        clipping_ratio = float(np.mean(np.abs(samples) >= 0.999))
        speech_seconds = float(
            sum(max(0.0, float(item["end"]) - float(item["start"])) for item in speech_segments)
        )
        speech_seconds = min(duration, speech_seconds)
        speech_ratio = float(speech_seconds / duration) if duration > 0 else 0.0

        reasons: List[str] = []
        if duration < self.options.min_duration_seconds:
            reasons.append("recording_too_short")
        if speech_seconds <= 0.0:
            reasons.append("no_speech_detected")
        elif speech_seconds < self.options.min_speech_seconds:
            reasons.append("insufficient_speech")
        if clipping_ratio > self.options.max_clipping_ratio:
            reasons.append("severe_clipping")
        if (
            self.options.min_speech_ratio is not None
            and speech_ratio < self.options.min_speech_ratio
        ):
            reasons.append("speech_ratio_below_threshold")

        return json_safe(
            {
                "duration_seconds": round(duration, 3),
                "speech_seconds": round(speech_seconds, 3),
                "speech_ratio": round(speech_ratio, 6),
                "clipping_ratio": round(clipping_ratio, 8),
                "speech_segments": list(speech_segments),
                "noise_reduction_applied": True,
                "processing_status": "completed",
                "scoring_allowed": not reasons,
                "rejection_reasons": reasons,
                "original_preserved": True,
                "original_sample_rate": ORIGINAL_SAMPLE_RATE,
                "stt_sample_rate": STT_SAMPLE_RATE,
                "device": device,
                "pipeline": "ffmpeg_deepfilternet_silero_vad",
                "cleaning_backend": "deepfilternet",
                "fallback_used": False,
                "speech_detection_method": "silero_vad",
            }
        )

    def _remaining_timeout(self, started: float) -> float:
        remaining = float(self.options.timeout_seconds) - (time.monotonic() - started)
        if remaining <= 0:
            raise AudioCleaningError(
                "processing_timeout",
                "Audio processing timed out. Please try a shorter recording.",
                status_code=504,
            )
        return remaining

    @staticmethod
    def _ensure_original_unchanged(source: Path, expected_hash: str) -> None:
        if _sha256(source) != expected_hash:
            raise AudioCleaningError(
                "original_audio_changed",
                "The original recording changed during processing.",
                technical_message="Input SHA-256 changed while cleaning.",
            )

    @staticmethod
    def _check_disk_space(source: Path, processing_dir: Path) -> None:
        try:
            free = shutil.disk_usage(processing_dir.parent).free
        except OSError as exc:
            raise AudioCleaningError(
                "disk_space_check_failed",
                "Audio storage is temporarily unavailable.",
                technical_message=f"Disk usage check failed: {exc!r}",
                status_code=503,
            ) from exc
        required = max(source.stat().st_size * 8, 32 * 1024 * 1024)
        if free < required:
            raise AudioCleaningError(
                "insufficient_disk_space",
                "There is not enough temporary storage to process this recording.",
                status_code=507,
            )

    @staticmethod
    def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
        temp_path = path.with_suffix(".tmp")
        try:
            temp_path.write_text(
                json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(path)
        except OSError as exc:
            raise AudioCleaningError(
                "manifest_write_failed",
                "Audio processing state could not be saved.",
                technical_message=f"Could not atomically write manifest: {exc!r}",
            ) from exc

    def _record_failure(
        self,
        manifest_path: Path,
        recording_id: str,
        error: AudioCleaningError,
    ) -> None:
        try:
            self._write_manifest(
                manifest_path,
                {
                    "recording_id": recording_id,
                    "processing_status": "failed",
                    "error_code": error.code,
                    "user_message": error.user_message,
                },
            )
        except AudioCleaningError:
            logger.exception(
                "audio_cleaning_failure_manifest_write_failed",
                extra={"recording_id": recording_id},
            )


def cleaner_from_settings(settings: Any) -> ExerciseAudioCleaner:
    """Build a cleaner from the service Settings object without global models."""
    return ExerciseAudioCleaner(
        AudioCleaningOptions(
            enabled=bool(settings.audio_cleaning_enabled),
            use_gpu=bool(settings.audio_cleaning_use_gpu),
            min_duration_seconds=float(settings.audio_cleaning_min_duration_seconds),
            min_speech_seconds=float(settings.audio_cleaning_min_speech_seconds),
            max_clipping_ratio=float(settings.audio_cleaning_max_clipping_ratio),
            min_speech_ratio=settings.audio_cleaning_min_speech_ratio,
            keep_intermediate_files=bool(settings.audio_cleaning_keep_intermediate_files),
            timeout_seconds=float(settings.audio_cleaning_timeout_seconds),
            output_root=Path(settings.audio_cleaning_output_dir).expanduser(),
        )
    )
