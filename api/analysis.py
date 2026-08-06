"""
Audio analysis pipeline — the ported acoustic/scoring wiring from the Flask
app, minus HTTP and persistence.

Flow: preserve the upload -> optional DeepFilterNet 48 kHz cleanup -> Silero
VAD + 16 kHz STT input -> Wav2Vec2 transcription -> reference G2P -> phoneme
alignment -> provisional metrics. The original signal remains the quality /
scoring evidence source; cleaned audio is never silently substituted for it.

Nothing here writes to the database or returns audio. Every on-disk artifact
this produces is registered in a ``cleanup_paths`` list the caller deletes in
an outer ``finally`` block, so recordings stay private and temporary on every
success or failure.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import settings
from .errors import AudioDecodeError, ServiceError

logger = logging.getLogger(__name__)

SERVICE_ROOT = Path(__file__).resolve().parent.parent

# Uploads are ephemeral and private: written here, then deleted in the caller's
# finally block. Never place this on a persistent Modal Volume.
UPLOAD_DIR = Path(os.environ.get("PRONUNCIATION_UPLOAD_DIR", str(SERVICE_ROOT / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def new_upload_path(mimetype: str) -> Path:
    """A unique temp path for one incoming recording."""
    return UPLOAD_DIR / f"{uuid.uuid4().hex}{extension_for_mimetype(mimetype)}"

def _find_ffmpeg_executable() -> Optional[str]:
    try:
        from src.core.audio.audio_cleaning import find_ffmpeg

        return find_ffmpeg()
    except Exception:
        return None


def convert_audio_to_wav(input_path: Path) -> Path:
    """Compatibility conversion used when the new cleaner is disabled."""
    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_converted.wav")
    if output_path.exists():
        return output_path
    if input_path.suffix.lower() in {".wav", ".wave"}:
        return input_path
    ffmpeg = _find_ffmpeg_executable()
    if ffmpeg is None:
        raise AudioDecodeError(
            "This recording is a compressed format that needs FFmpeg to decode. "
            "Install FFmpeg and place it on PATH (or configure FFMPEG_BINARY)."
        )
    try:
        from src.core.audio.audio_cleaning import convert_with_ffmpeg

        return convert_with_ffmpeg(
            input_path,
            output_path,
            sample_rate=16_000,
            timeout_seconds=settings.audio_cleaning_timeout_seconds,
            ffmpeg_binary=ffmpeg,
        )
    except Exception as exc:
        raise AudioDecodeError("Could not decode the uploaded recording into WAV.") from exc


def extension_for_mimetype(mimetype: str) -> str:
    mimetype = (mimetype or "").lower()
    if "wav" in mimetype or "wave" in mimetype:
        return ".wav"
    if "ogg" in mimetype:
        return ".ogg"
    if "mp4" in mimetype or "mpeg" in mimetype or "aac" in mimetype:
        return ".m4a"
    if "webm" in mimetype or "opus" in mimetype:
        return ".webm"
    return ".webm"


def cleanup_audio_files(paths: List[Optional[Path]]) -> None:
    """Delete private request artifacts unless explicit local retention is on."""
    if settings.retain_audio or settings.audio_cleaning_keep_intermediate_files:
        return
    seen = set()
    for path in paths:
        try:
            if path is None:
                continue
            resolved = Path(path)
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_dir():
                shutil.rmtree(resolved)
            elif resolved.exists():
                resolved.unlink()
        except Exception as exc:
            print("Could not delete temporary audio:", repr(exc))


def candidate_cleanup_paths(audio_path: Path) -> List[Path]:
    """Compatibility files plus the immutable uploaded original."""
    audio_path = Path(audio_path)
    return [
        audio_path,
        audio_path.with_name(audio_path.stem + "_converted.wav"),
    ]


_CLEANVOICE_FALLBACK_ERROR_CODES = frozenset(
    {
        "deepfilternet_initialization_failed",
        "deepfilternet_processing_failed",
        "silero_vad_initialization_failed",
        "silero_vad_failed",
        "audio_measurement_failed",
        "invalid_generated_output",
        "processing_timeout",
        "processing_failed",
    }
)


def _energy_speech_segments(path: Path) -> List[Dict[str, float]]:
    """Conservative VAD fallback used only if Silero itself is unavailable."""
    import numpy as np
    import soundfile as sf

    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0 or int(sample_rate) <= 0:
        return []
    frame_size = max(1, int(sample_rate * 0.03))
    frame_count = int(np.ceil(audio.size / frame_size))
    padded = np.pad(audio, (0, frame_count * frame_size - audio.size))
    rms = np.sqrt(np.mean(padded.reshape(frame_count, frame_size) ** 2, axis=1))
    peak_rms = float(np.max(rms)) if rms.size else 0.0
    if peak_rms < 0.003:
        return []
    noise_floor = float(np.percentile(rms, 20))
    threshold = max(0.003, min(noise_floor * 2.5, peak_rms * 0.35))
    active = rms >= threshold
    segments: List[Dict[str, float]] = []
    start: Optional[int] = None
    for index, is_active in enumerate(active.tolist() + [False]):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            segments.append(
                {
                    "start": round(start * frame_size / sample_rate, 3),
                    "end": round(min(index * frame_size, audio.size) / sample_rate, 3),
                }
            )
            start = None
    return segments


def _clean_with_cleanvoice_fallback(
    audio_path: Path,
    *,
    primary_error_code: str,
) -> Dict[str, Any]:
    """Run external enhancement, then restore the service's exact WAV contract."""
    from src.core.audio.audio_cleaning import (
        AudioCleaningError,
        ORIGINAL_SAMPLE_RATE,
        STT_SAMPLE_RATE,
        _sha256,
        cleaner_from_settings,
        convert_with_ffmpeg,
    )
    from src.core.audio.cleanvoice_service import enhance_recording

    source = Path(audio_path).resolve()
    source_hash = _sha256(source)
    root = Path(settings.audio_cleaning_output_dir).expanduser().resolve()
    processing_dir = root / (
        f"{source.stem}-{uuid.uuid4().hex}_cleanvoice_fallback"
    )
    normalized_48k = processing_dir / "source_48k_mono.wav"
    provider_output = processing_dir / "cleanvoice_output.wav"
    cleaned_48k = processing_dir / "cleaned_48k_mono.wav"
    cleaned_16k = processing_dir / "cleaned_16k_mono.wav"
    try:
        processing_dir.mkdir(parents=True, exist_ok=False)
        timeout = float(settings.audio_cleaning_timeout_seconds)
        convert_with_ffmpeg(
            source,
            normalized_48k,
            sample_rate=ORIGINAL_SAMPLE_RATE,
            timeout_seconds=timeout,
        )
        enhance_recording(normalized_48k, provider_output)
        convert_with_ffmpeg(
            provider_output,
            cleaned_48k,
            sample_rate=ORIGINAL_SAMPLE_RATE,
            timeout_seconds=timeout,
        )
        convert_with_ffmpeg(
            cleaned_48k,
            cleaned_16k,
            sample_rate=STT_SAMPLE_RATE,
            timeout_seconds=timeout,
        )

        cleaner = cleaner_from_settings(settings)
        vad_method = "silero_vad"
        try:
            segments = cleaner._run_silero_vad(cleaned_16k, "cpu")
        except AudioCleaningError:
            segments = _energy_speech_segments(cleaned_16k)
            vad_method = "energy_vad"
        metadata = cleaner._quality_metadata(normalized_48k, segments, "external")
        if _sha256(source) != source_hash:
            raise AudioCleaningError(
                "original_audio_changed",
                "The original recording changed during processing.",
            )
        metadata.update(
            {
                "pipeline": f"ffmpeg_cleanvoice_fallback_{vad_method}",
                "cleaning_backend": "cleanvoice",
                "fallback_used": True,
                "primary_error_code": primary_error_code,
                "speech_detection_method": vad_method,
            }
        )
        return {
            "original_audio_path": source,
            "normalized_audio_48k_path": normalized_48k,
            "cleaned_audio_48k_path": cleaned_48k,
            "cleaned_audio_16k_path": cleaned_16k,
            "reduced_audio_path": cleaned_16k,
            "noise_reduction_applied": True,
            "preprocessing_pipeline": metadata["pipeline"],
            "audio_processing": metadata,
            "cleanup_paths": [processing_dir],
        }
    except Exception:
        if not (
            settings.retain_audio
            or settings.audio_cleaning_keep_intermediate_files
        ):
            shutil.rmtree(processing_dir, ignore_errors=True)
        raise


def clean_recording(audio_path: Path, wav_path: Optional[Path] = None) -> Dict[str, Any]:
    """Create cleaned 48/16 kHz WAVs and VAD/quality metadata."""
    del wav_path  # kept only for backwards-compatible callers
    from src.core.audio.audio_cleaning import AudioCleaningError, cleaner_from_settings
    from src.core.audio.cleanvoice_service import cleanvoice_fallback_configured

    try:
        cleaned = cleaner_from_settings(settings).process(Path(audio_path))
    except AudioCleaningError as exc:
        if (
            exc.code in _CLEANVOICE_FALLBACK_ERROR_CODES
            and cleanvoice_fallback_configured()
        ):
            logger.warning(
                "local_audio_cleaning_failed_using_cleanvoice_fallback",
                extra={"primary_error_code": exc.code},
            )
            try:
                return _clean_with_cleanvoice_fallback(
                    Path(audio_path),
                    primary_error_code=exc.code,
                )
            except Exception as fallback_exc:
                logger.error(
                    "cleanvoice_audio_fallback_failed",
                    extra={"primary_error_code": exc.code},
                    exc_info=True,
                )
                raise ServiceError(
                    "Audio cleaning failed locally and with the configured fallback.",
                    code="audio_cleaning_all_providers_failed",
                    status_code=502,
                ) from fallback_exc
        raise ServiceError(
            exc.user_message,
            code=exc.code,
            status_code=exc.status_code,
        ) from exc
    return {
        "original_audio_path": cleaned.original_audio_path,
        "normalized_audio_48k_path": cleaned.normalized_audio_48k_path,
        "cleaned_audio_48k_path": cleaned.cleaned_audio_48k_path,
        "cleaned_audio_16k_path": cleaned.cleaned_audio_16k_path,
        # Compatibility key used by the existing acoustic transcription seam.
        "reduced_audio_path": cleaned.cleaned_audio_16k_path,
        "noise_reduction_applied": True,
        "preprocessing_pipeline": cleaned.metadata["pipeline"],
        "audio_processing": cleaned.metadata,
        "cleanup_paths": [cleaned.processing_directory],
    }


def process_recording(audio_path: Path) -> Dict[str, Any]:
    """Inspect original evidence, clean for STT when enabled, then transcribe."""
    import librosa

    from src.core.audio.audio_quality import analyze_audio_quality
    from . import acoustic

    if settings.audio_cleaning_enabled:
        cleaned = clean_recording(audio_path)
        quality_source = cleaned["normalized_audio_48k_path"]
    else:
        wav_path = convert_audio_to_wav(audio_path)
        quality_source = wav_path
        cleaned = {
            "original_audio_path": Path(audio_path),
            "normalized_audio_48k_path": None,
            "cleaned_audio_48k_path": None,
            "cleaned_audio_16k_path": wav_path,
            "reduced_audio_path": wav_path,
            "noise_reduction_applied": False,
            "preprocessing_pipeline": "audio_cleaning_disabled",
            "audio_processing": {
                "processing_status": "disabled",
                "noise_reduction_applied": False,
                "scoring_allowed": True,
                "rejection_reasons": [],
                "pipeline": "audio_cleaning_disabled",
            },
            "cleanup_paths": candidate_cleanup_paths(Path(audio_path)),
        }

    raw_audio, sr = librosa.load(str(quality_source), sr=16000, mono=True)
    decision = analyze_audio_quality(raw_audio, sr)
    cleaning_metadata = cleaned["audio_processing"]
    if not cleaning_metadata.get("scoring_allowed", True):
        decision.scorable = False
        decision.quality_weight = 0.0
        for reason in cleaning_metadata.get("rejection_reasons", []):
            if reason not in decision.reasons:
                decision.reasons.append(reason)
    decision.metrics["audio_cleaning"] = cleaning_metadata

    result: Dict[str, Any] = {
        "quality_decision": decision,
        "predicted_ipa": None,
    }
    result.update(cleaned)
    result["predicted_ipa"] = acoustic.transcribe(result["reduced_audio_path"])
    return result


def analyze_recording(
    user_text: str,
    audio_path: Path,
    cleanup_paths: List[Optional[Path]],
    reference: Any = None,
) -> Dict[str, Any]:
    """Run the full stateless analysis for one saved upload and return an
    ``analysis`` bundle: the client-facing fields plus the internal alignment /
    metrics / decision needed to persist an attempt. Appends every produced
    artifact to ``cleanup_paths``. Does NOT persist anything."""
    from src.core.g2p.phoneme_vectors_professional import scoring_engine, scoring_trusted
    from src.core.scoring.scoring import align_phonemes, calculate_metrics, to_api_alignment
    from src.core.g2p.tokenization import (
        ipa_reading_guide,
        tokenize_ctc_prediction,
        tokenize_reference_ipa,
    )

    if reference is None:
        from .reference_validation import resolve_supported_reference

        reference = resolve_supported_reference(user_text)
    reference_ipa = reference.ipa

    recording = process_recording(audio_path)
    cleanup_paths.extend(recording.get("cleanup_paths", []))
    decision = recording["quality_decision"]
    predicted_ipa = recording["predicted_ipa"]

    engine_name = scoring_engine()
    engine_trusted = scoring_trusted()

    reference_reason_parts = [
        *(f"unresolved:{word}" for word in reference.unresolved_heteronyms),
        *(f"unsupported:{word}" for word in reference.unsupported_heteronyms),
        *(f"oov:{word}" for word in reference.oov_words),
    ]
    reference_reason = ",".join(reference_reason_parts) or None

    ref_seq = tokenize_reference_ipa(reference_ipa)
    hyp_seq = tokenize_ctc_prediction(predicted_ipa)
    rows = align_phonemes(ref_seq, hyp_seq)
    metrics = calculate_metrics(rows)
    api_alignment = to_api_alignment(rows)

    return {
        "reference": reference,
        "reference_ipa": reference_ipa,
        "predicted_ipa": predicted_ipa,
        "decision": decision,
        "engine_name": engine_name,
        "engine_trusted": engine_trusted,
        "reference_reason": reference_reason,
        "rows": rows,
        "api_alignment": api_alignment,
        "metrics": metrics,
        "recording": recording,
        "reference_guide": ipa_reading_guide(reference_ipa),
        "predicted_guide": ipa_reading_guide(predicted_ipa),
    }


def mastery_note_for(analysis: Dict[str, Any]) -> Optional[str]:
    decision = analysis["decision"]
    if not analysis["engine_trusted"]:
        return (
            "PanPhon unavailable or incomplete: showing provisional fallback scores only; "
            "mastery was NOT updated."
        )
    if not analysis["reference"].reference_g2p_trusted:
        return "The reference pronunciation is unresolved or unsupported; mastery was NOT updated."
    if not decision.scorable:
        return (
            "Audio quality warnings were detected, but the recording was still processed. "
            "The score is shown without updating progress."
        )
    return None


def build_analysis_payload(
    analysis: Dict[str, Any], user_text: str, mastery_updated: bool, persisted: bool
) -> Dict[str, Any]:
    """Assemble the client-facing analysis payload. Processed audio is never
    included or retained."""
    reference = analysis["reference"]
    decision = analysis["decision"]
    recording = analysis["recording"]
    internal_payload = {
        "scorable": True,
        "quality_warning": not decision.scorable,
        "persisted": persisted,
        "text": user_text,
        "reference_ipa": analysis["reference_ipa"],
        "predicted_ipa": analysis["predicted_ipa"],
        "reference_guide": analysis["reference_guide"],
        "predicted_guide": analysis["predicted_guide"],
        **reference.to_dict(),
        "scoring_engine": analysis["engine_name"],
        "scoring_trusted": analysis["engine_trusted"],
        "mastery_updated": mastery_updated,
        "mastery_note": mastery_note_for(analysis),
        "noise_reduction_applied": recording["noise_reduction_applied"],
        "preprocessing_pipeline": recording["preprocessing_pipeline"],
        "audio_processing": recording.get("audio_processing", {}),
        "audio_quality": decision.to_dict(),
        "alignment": analysis["api_alignment"],
        "metrics": analysis["metrics"],
    }
    from .arpabet import to_public_arpabet
    from .pronunciation_feedback import with_pronunciation_errors

    return with_pronunciation_errors(to_public_arpabet(internal_payload))
