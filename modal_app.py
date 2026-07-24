"""
Modal deployment for the pronunciation AI microservice (FastAPI / ASGI).

Development:
    modal serve modal_app.py

Production (does NOT deploy automatically — run explicitly):
    modal deploy modal_app.py

Notes / constraints (see MICROSERVICE_REPORT.md):
  * The SQLite database lives on a Modal Volume at /data/pronunciation_ai.db.
    Modal Volumes are not built for concurrent modification of one SQLite file,
    so this v1 runs single-writer: max_containers=1, one active input,
    600s request timeout. Move to PostgreSQL before scaling to many containers.
  * Secrets carry SERVICE_API_KEY and optional provider keys — never the .env.
  * The 378 MB model weight ships in its own cached image layer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

APP_NAME = "pronunciation-ai-service"
APP_DIR = "/root/pronunciation-ai-service"
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model" / "my_wav2vec2_phoneme_model"
MODEL_REMOTE_DIR = f"{APP_DIR}/model/my_wav2vec2_phoneme_model"

# Never upload local secrets, recordings, databases, caches, or test artifacts.
# The model is copied in its own cached layer, so source-only changes don't
# re-upload the weight.
IMAGE_IGNORES = [
    ".git", ".git/**",
    ".env", ".env.*", "!.env.example",
    ".pytest_cache", ".pytest_cache/**",
    "**/__pycache__", "**/__pycache__/**", "**/*.pyc",
    "tests", "tests/**",
    "pronunciation_ai.db", "pronunciation_ai.db-*", "pronunciation_ai.db.*",
    "uploads/**",
    "model/**",
]

runtime_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(PROJECT_DIR / "requirements-deploy.txt"))
    .env(
        {
            "PRONUNCIATION_DB_PATH": "/data/pronunciation_ai.db",
            "PRONUNCIATION_UPLOAD_DIR": "/tmp/pronunciation_uploads",
            "CLEANVOICE_ENABLED": "0",
            "CLEANVOICE_STRICT": "0",
            "RETAIN_AUDIO": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(str(MODEL_DIR), remote_path=MODEL_REMOTE_DIR, copy=True)
    .add_local_dir(str(PROJECT_DIR), remote_path=APP_DIR, copy=True, ignore=IMAGE_IGNORES)
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("pronunciation-ai-data", create_if_missing=True)


@app.function(
    image=runtime_image,
    secrets=[modal.Secret.from_name("pronunciation-ai-secrets")],
    volumes={"/data": data_volume},
    cpu=1.0,
    memory=4096,
    timeout=600,
    scaledown_window=60,
    max_containers=1,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def web():
    """Expose the FastAPI app at a public Modal HTTPS URL."""
    os.chdir(APP_DIR)
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)

    # Bootstrap (schema + seed) runs in the app's lifespan; explicitly commit
    # the Volume afterward so the seeded database is durable.
    from api.main import app as fastapi_app

    return fastapi_app
