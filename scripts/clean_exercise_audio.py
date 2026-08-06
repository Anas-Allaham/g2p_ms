"""Manual smoke-test CLI for the offline audio-cleaning pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from api.config import settings
from src.core.audio.audio_cleaning import (
    AudioCleaningError,
    cleaner_from_settings,
    json_safe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create DeepFilterNet-cleaned 48 kHz and 16 kHz WAV files."
    )
    parser.add_argument("input", type=Path, help="Source recording (never modified).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Parent directory for the isolated processing directory.",
    )
    parser.add_argument("--recording-id", help="Stable id used for idempotent reruns.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun a completed recording instead of reusing its manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = cleaner_from_settings(settings).process(
            args.input,
            output_dir=args.output_dir,
            recording_id=args.recording_id,
            force=args.force,
        )
    except AudioCleaningError as exc:
        print(
            json.dumps(
                {
                    "processing_status": "failed",
                    "error_code": exc.code,
                    "message": exc.user_message,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    payload = result.to_dict(include_paths=True)
    payload["summary"] = {
        "duration_seconds": result.metadata["duration_seconds"],
        "speech_seconds": result.metadata["speech_seconds"],
        "speech_ratio": result.metadata["speech_ratio"],
        "clipping_ratio": result.metadata["clipping_ratio"],
        "scoring_allowed": result.metadata["scoring_allowed"],
        "rejection_reasons": result.metadata["rejection_reasons"],
    }
    print(json.dumps(json_safe(payload), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
