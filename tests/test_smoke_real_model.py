"""
Optional real-model smoke test.

Runs the FULL pipeline (no mocked transcription) against the bundled WAV
fixture. Skipped automatically when the trained model weight is not present, so
the mocked CI suite stays green without it.

Run explicitly with the weight installed:
    python -m pytest tests/test_smoke_real_model.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SERVICE_API_KEY", "test-service-key")

from api import acoustic  # noqa: E402

WEIGHT_PRESENT = acoustic.model_weight_present()
FIXTURE_WAV = ROOT / "tests" / "fixtures" / "speech_sample.wav"

pytestmark = pytest.mark.skipif(
    not WEIGHT_PRESENT, reason="model weight not installed; real-model smoke test skipped"
)


def test_real_model_transcribes_and_scores(tmp_path):
    import shutil

    from src.core.persistence import db
    from fastapi.testclient import TestClient

    from api.main import app

    original = db.DB_PATH
    db.set_database_for_testing(tmp_path / "smoke.db")
    try:
        with TestClient(app) as c:
            c.headers.update({"Authorization": f"Bearer {os.environ['SERVICE_API_KEY']}"})
            r = c.post(
                "/api/v1/pronunciation/analyses",
                data={"text": "school"},
                files={"audio": ("s.wav", FIXTURE_WAV.read_bytes(), "audio/wav")},
            )
            assert r.status_code == 200
            data = r.json()["data"]
            assert data["predicted_ipa"]  # real transcription produced something
            assert "utterance_score" in data["metrics"]
    finally:
        db.set_database_for_testing(original)
