"""
Shared test fixtures.

Makes the service root importable (so ``import scoring`` / ``import db`` work),
sets a service API key BEFORE the app config is imported, and provides a temp
database plus an authenticated FastAPI TestClient with the acoustic model
mocked (the CI suite never loads torch/Wav2Vec2 weights).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must be set before api.config imports settings.
os.environ.setdefault("SERVICE_API_KEY", "test-service-key")
os.environ.setdefault("RETAIN_AUDIO", "0")
# Integration/model tests opt in explicitly. A developer's local .env may
# enable the real cleaning pipeline, but the deterministic suite must not load
# FFmpeg or download DeepFilterNet/Silero models.
os.environ["AUDIO_CLEANING_ENABLED"] = "0"

import pytest

TEST_API_KEY = os.environ["SERVICE_API_KEY"]
AUTH = {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture
def temp_db(tmp_path):
    """Point db at a fresh temp SQLite file for one test, then restore."""
    from src.core.persistence import db

    original = db.DB_PATH
    db.set_database_for_testing(tmp_path / "test_pron.db")
    db.init_db()
    try:
        yield db
    finally:
        db.set_database_for_testing(original)


FIXTURE_WAV = ROOT / "tests" / "fixtures" / "speech_sample.wav"


@pytest.fixture
def auth():
    return dict(AUTH)


@pytest.fixture
def sample_wav():
    """Real 16 kHz WAV bytes so the convert/quality path runs for real; only the
    acoustic transcription is mocked in the client fixture."""
    return FIXTURE_WAV.read_bytes()


@pytest.fixture(scope="session")
def seeded_template(tmp_path_factory):
    """Seed the exercise bank ONCE per test session into a template DB file.
    Per-test client DBs are copied from this, so the expensive 178-sentence G2P
    tagging runs a single time instead of on every client test."""
    from src.core.persistence import db
    from api.bootstrap import seed_exercise_bank

    template = tmp_path_factory.mktemp("seed") / "template.db"
    original = db.DB_PATH
    db.set_database_for_testing(template)
    db.init_db()
    seed_exercise_bank()
    db.checkpoint()
    db.set_database_for_testing(original)
    return template


@pytest.fixture
def client(tmp_path, monkeypatch, seeded_template):
    """Authenticated TestClient on a fresh per-test DB copied from the seeded
    template, with the acoustic model mocked so analysis needs no weights."""
    import shutil

    from fastapi.testclient import TestClient

    from src.core.persistence import db
    from api import acoustic
    from api.main import app

    client_db = tmp_path / "client_pron.db"
    shutil.copyfile(seeded_template, client_db)

    original = db.DB_PATH
    db.set_database_for_testing(client_db)

    # Pretend the trained model is present + fast; transcribe returns fixed IPA.
    monkeypatch.setattr(acoustic, "model_config_present", lambda: True)
    monkeypatch.setattr(acoustic, "model_weight_present", lambda: True)
    monkeypatch.setattr(acoustic, "model_loaded", lambda: True)
    monkeypatch.setattr(acoustic, "transcribe", lambda _path: "skuːl")

    try:
        # raise_server_exceptions=False so the registered 500 handler's envelope
        # is returned (like a real server) instead of re-raising into the test.
        with TestClient(app, raise_server_exceptions=False) as c:
            c.headers.update(AUTH)
            yield c
    finally:
        db.set_database_for_testing(original)
