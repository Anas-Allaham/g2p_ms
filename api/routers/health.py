"""
Public health endpoints (no auth).

  * GET /health/live   — process liveness only (is the ASGI app up).
  * GET /health/ready  — minimal readiness: model files, G2P assets, scoring
                         trust, database reachability, and a populated
                         exercise bank. Returns 200 only when all are ready,
                         503 otherwise, so an orchestrator can gate traffic.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse

from ..envelopes import success

router = APIRouter(tags=["health"])


@router.get("/health/live")
def live(request: Request):
    return success({"status": "alive"}, request)


def readiness_snapshot() -> dict:
    """Cheap readiness probe shared by /health/ready and /capabilities."""
    from src.core.persistence import db
    from src.core.g2p.g2p_service import HETERONYMS_PATH, IPA_DICT_PATH, get_g2p_mode
    from src.core.g2p.phoneme_vectors_professional import scoring_engine, scoring_trusted, validate_panphon_inventory

    from .. import acoustic

    try:
        bank_count = db.count_exercise_bank()
        db_ok = True
    except Exception as exc:  # pragma: no cover
        bank_count = 0
        db_ok = f"error: {exc!r}"

    g2p_ok = HETERONYMS_PATH.exists() and IPA_DICT_PATH.exists()
    panphon = validate_panphon_inventory()

    checks = {
        "model_config_present": acoustic.model_config_present(),
        "model_weight_present": acoustic.model_weight_present(),
        "g2p_assets_present": g2p_ok,
        "scoring_engine": scoring_engine(),
        "scoring_trusted": scoring_trusted(),
        "panphon_inventory_ok": panphon["ok"],
        "database_ok": db_ok is True,
        "exercise_bank_count": bank_count,
        "exercise_bank_populated": bank_count > 0,
        "g2p_mode": get_g2p_mode(),
    }
    ready = bool(
        checks["model_config_present"]
        and checks["model_weight_present"]
        and checks["g2p_assets_present"]
        and checks["database_ok"]
        and checks["exercise_bank_populated"]
    )
    return {"ready": ready, "checks": checks}


@router.get("/health/ready")
def ready(request: Request):
    snapshot = readiness_snapshot()
    body = success(snapshot, request)
    if not snapshot["ready"]:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body)
    return body
