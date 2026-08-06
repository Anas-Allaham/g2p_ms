"""Authenticated endpoint for downloading a cleaned pronunciation WAV."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .. import analysis as analysis_mod
from ..config import API_VERSION
from ..dependencies import save_upload
from ..errors import ValidationError
from ..security import require_service_auth

router = APIRouter(
    prefix=f"/api/{API_VERSION}",
    tags=["audio"],
    dependencies=[Depends(require_service_auth)],
)


def _download_name(upload_name: Optional[str]) -> str:
    stem = Path(upload_name or "recording").stem
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-")[:80]
    return f"{safe_stem or 'recording'}_cleaned.wav"


@router.post(
    "/audio/clean",
    response_class=FileResponse,
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Pronunciation-safe cleaned mono WAV.",
        }
    },
    summary="Clean a pronunciation recording",
)
def clean_audio(
    audio: UploadFile = File(..., description="Audio recording to clean."),
    sample_rate: int = Query(
        48000,
        description="Return 48 kHz for playback or 16 kHz for STT/alignment.",
    ),
):
    """Return a locally cleaned WAV, with configured Cleanvoice fallback."""
    if sample_rate not in {16000, 48000}:
        raise ValidationError(
            "sample_rate must be either 16000 or 48000.",
            details={"field": "sample_rate"},
        )
    cleanup_paths: List[Optional[Path]] = []
    audio_path = analysis_mod.new_upload_path(audio.content_type or "")
    cleanup_paths.extend(analysis_mod.candidate_cleanup_paths(audio_path))

    try:
        save_upload(audio, audio_path)
        cleaned = analysis_mod.clean_recording(audio_path)
        cleanup_paths.extend(cleaned["cleanup_paths"])
        output_path = Path(
            cleaned[
                "cleaned_audio_48k_path"
                if sample_rate == 48000
                else "cleaned_audio_16k_path"
            ]
        )
        metadata = cleaned["audio_processing"]
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=_download_name(audio.filename),
            headers={
                "X-Audio-Processing-Pipeline": str(
                    cleaned["preprocessing_pipeline"]
                ),
                "X-Noise-Reduction-Applied": str(
                    bool(cleaned["noise_reduction_applied"])
                ).lower(),
                "X-Original-Preserved": str(
                    bool(metadata["original_preserved"])
                ).lower(),
                "X-Audio-Processing-Status": str(metadata["processing_status"]),
                "X-Audio-Cleaning-Backend": str(
                    metadata.get("cleaning_backend", "deepfilternet")
                ),
                "X-Audio-Fallback-Used": str(
                    bool(metadata.get("fallback_used", False))
                ).lower(),
                "X-Audio-Speech-Seconds": str(metadata["speech_seconds"]),
                "X-Audio-Scoring-Allowed": str(
                    bool(metadata["scoring_allowed"])
                ).lower(),
                "X-Audio-Rejection-Reasons": ",".join(
                    metadata["rejection_reasons"]
                ),
            },
            background=BackgroundTask(
                analysis_mod.cleanup_audio_files,
                cleanup_paths,
            ),
        )
    except Exception:
        analysis_mod.cleanup_audio_files(cleanup_paths)
        raise
