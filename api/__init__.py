"""
FastAPI application package for the pronunciation AI microservice.

The authoritative domain modules (db, services, scoring, mastery, g2p_service,
...) live flat at the service root and are reused verbatim. Ensure that root is
importable regardless of the process CWD (e.g. under Modal, which chdirs), so
``import db`` / ``import services`` resolve from anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))
