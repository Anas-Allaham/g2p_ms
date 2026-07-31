"""
Audio analysis pipeline — the ported acoustic/scoring wiring from the Flask
app, minus HTTP and persistence.

Flow (unchanged behaviour): convert to 16 kHz mono WAV -> advisory quality
diagnostics -> optional Cleanvoice / local noise reduction -> Wav2Vec2
transcription -> reference G2P -> phoneme alignment -> provisional metrics.

Nothing here writes to the database or returns audio. Every on-disk artifact
this produces is registered in a ``cleanup_paths`` list the caller deletes in
an outer ``finally`` block, so recordings stay private and temporary on every
success or failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import settings
from .errors import AudioDecodeError, CleanvoiceUnavailableError

SERVICE_ROOT = Path(__file__).resolve().parent.parent
VOICE_FILTERING_DIR = SERVICE_ROOT / "voice-filtering"

# Uploads are ephemeral and private: written here, then deleted in the caller's
# finally block. Never place this on a persistent Modal Volume.
UPLOAD_DIR = Path(os.environ.get("PRONUNCIATION_UPLOAD_DIR", str(SERVICE_ROOT / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def new_upload_path(mimetype: str) -> Path:
    """A unique temp path for one incoming recording."""
    return UPLOAD_DIR / f"{uuid.uuid4().hex}{extension_for_mimetype(mimetype)}"

try:
    import noisereduce as nr
except Exception:  # pragma: no cover
    nr = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

if str(VOICE_FILTERING_DIR) not in sys.path:
    sys.path.insert(0, str(VOICE_FILTERING_DIR))

try:
    from audio_filter_safe import process_audio_file as safe_process_audio_file
except Exception as exc:  # pragma: no cover
    safe_process_audio_file = None
    print("audio_filter_safe is unavailable. Falling back to legacy preprocessing.")
    print("Reason:", repr(exc))


def _find_ffmpeg_executable() -> Optional[str]:
    import os

    configured = os.environ.get("FFMPEG_BINARY", "").strip()
    if configured:
        configured_path = Path(configured)
        if configured_path.exists():
            return str(configured_path)
        found = shutil.which(configured)
        if found:
            return found
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def convert_audio_to_wav(input_path: Path) -> Path:
    """Convert browser audio to 16 kHz mono WAV when FFmpeg is available."""
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
            "Install FFmpeg or `pip install imageio-ffmpeg` on the service host."
        )
    command = [ffmpeg, "-y", "-i", str(input_path), "-ac", "1", "-ar", "16000", str(output_path)]
    try:
        completed = subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        tail = detail[-1] if detail else "FFmpeg could not decode the uploaded recording."
        raise AudioDecodeError(f"Could not decode the uploaded recording: {tail}") from exc
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise AudioDecodeError("Could not decode the uploaded recording into WAV.")
    return output_path


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


def _apply_noise_reduction(wav_path: Path, reduced_path: Path):
    """Best-effort mild noise reduction. Returns
    (model_input_path, noise_reduction_applied, preprocessing_pipeline)."""
    import librosa

    if safe_process_audio_file is not None:
        try:
            safe_process_audio_file(input_path=wav_path, output_path=reduced_path, use_noise_reduce=True)
            if reduced_path.exists():
                applied = nr is not None
                pipeline = "audio_filter_safe" if applied else "audio_filter_safe_without_noisereduce"
                return reduced_path, applied, pipeline
        except Exception as exc:
            print("audio_filter_safe failed. Falling back to legacy noisereduce.")
            print("Reason:", repr(exc))

    audio_array, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    if nr is not None:
        reduced = nr.reduce_noise(y=audio_array, sr=sr, stationary=False, prop_decrease=0.35)
        pipeline, applied = "legacy_noisereduce", True
    else:
        reduced, pipeline, applied = audio_array, "raw_audio_fallback", False
    if sf is not None:
        sf.write(str(reduced_path), reduced, sr)
        if reduced_path.exists():
            return reduced_path, applied, pipeline
    return wav_path, applied, pipeline


def cleanup_audio_files(paths: List[Optional[Path]]) -> None:
    """Delete temporary recording files (original, converted, reduced,
    cleanvoice) unless retention is explicitly enabled. Recordings are private
    by default. Never raises."""
    if settings.retain_audio:
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
            if resolved.exists():
                resolved.unlink()
        except Exception as exc:
            print("Could not delete temporary audio:", repr(exc))


def candidate_cleanup_paths(audio_path: Path) -> List[Path]:
    """Every filename the recording pipeline can create for one upload."""
    audio_path = Path(audio_path)
    return [
        audio_path,
        audio_path.with_name(audio_path.stem + "_converted.wav"),
        audio_path.with_name(audio_path.stem + "_cleanvoice.wav"),
        audio_path.with_name(audio_path.stem + "_reduced.wav"),
    ]


def process_recording(audio_path: Path) -> Dict[str, Any]:
    """Convert -> inspect quality -> enhance -> transcribe. Mirrors the Flask
    pipeline exactly. ``cleanup_paths`` lists every on-disk artifact produced so
    the caller can delete them (private/temporary by default)."""
    import librosa

    from src.core.audio.audio_quality import analyze_audio_quality
    from src.core.audio.cleanvoice_service import (
        CleanvoiceProcessingError,
        cleanvoice_configured,
        cleanvoice_strict,
        enhance_recording,
    )

    from . import acoustic

    wav_path = convert_audio_to_wav(audio_path)
    raw_audio, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    decision = analyze_audio_quality(raw_audio, sr)

    result: Dict[str, Any] = {
        "quality_decision": decision,
        "predicted_ipa": None,
        "reduced_audio_path": None,
        "noise_reduction_applied": False,
        "preprocessing_pipeline": "not_processed",
        "cleanvoice_applied": False,
        "cleanvoice_error": None,
        "cleanup_paths": [audio_path, wav_path],
    }
    cleanvoice_path = audio_path.with_name(audio_path.stem + "_cleanvoice.wav")
    result["cleanup_paths"].append(cleanvoice_path)
    applied = False
    pipeline = "raw_audio_fallback"

    if cleanvoice_configured():
        try:
            model_input_path = enhance_recording(wav_path, cleanvoice_path)
            applied = True
            pipeline = "cleanvoice_noise_reduction_normalization"
            result["cleanvoice_applied"] = True
        except CleanvoiceProcessingError as exc:
            result["cleanvoice_error"] = str(exc)
            if cleanvoice_strict():
                raise CleanvoiceUnavailableError(str(exc))
            reduced_path = audio_path.with_name(audio_path.stem + "_reduced.wav")
            model_input_path, applied, local_pipeline = _apply_noise_reduction(wav_path, reduced_path)
            pipeline = f"cleanvoice_failed_fallback:{local_pipeline}"
            result["cleanup_paths"].append(reduced_path)
    else:
        reduced_path = audio_path.with_name(audio_path.stem + "_reduced.wav")
        model_input_path, applied, pipeline = _apply_noise_reduction(wav_path, reduced_path)
        result["cleanup_paths"].append(reduced_path)

    result["noise_reduction_applied"] = applied
    result["preprocessing_pipeline"] = pipeline
    result["reduced_audio_path"] = model_input_path if model_input_path.exists() else None
    result["predicted_ipa"] = acoustic.transcribe(model_input_path)
    return result


def analyze_recording(user_text: str, audio_path: Path, cleanup_paths: List[Optional[Path]]) -> Dict[str, Any]:
    """Run the full stateless analysis for one saved upload and return an
    ``analysis`` bundle: the client-facing fields plus the internal alignment /
    metrics / decision needed to persist an attempt. Appends every produced
    artifact to ``cleanup_paths``. Does NOT persist anything."""
    from src.core.g2p.g2p_service import g2p_convert_with_metadata
    from src.core.g2p.phoneme_vectors_professional import scoring_engine, scoring_trusted
    from src.core.scoring.scoring import align_phonemes, calculate_metrics, to_api_alignment
    from src.core.g2p.tokenization import (
        ipa_reading_guide,
        tokenize_ctc_prediction,
        tokenize_reference_ipa,
    )

    reference = g2p_convert_with_metadata(user_text)
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
        "cleanvoice_applied": recording.get("cleanvoice_applied", False),
        "cleanvoice_error": recording.get("cleanvoice_error"),
        "audio_quality": decision.to_dict(),
        "alignment": analysis["api_alignment"],
        "metrics": analysis["metrics"],
    }
    from .arpabet import to_public_arpabet

    return to_public_arpabet(internal_payload)
