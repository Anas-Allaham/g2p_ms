"""
Startup bootstrap: initialize the database schema, warm the G2P engine,
validate the scoring inventory, and seed the shared exercise bank on first run.

Seeding populates ONLY the public exercise corpus from
``data/seed_sentences.txt``. It never imports users, attempts, or any private
state — a fresh database starts with zero subjects.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict

SERVICE_ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = SERVICE_ROOT / "data" / "seed_sentences.txt"
EXERCISE_SEED_VERSION_KEY = "exercise_seed_version"
EXERCISE_SEED_VERSION = "complex-lessons-v2"


def validate_inventory() -> Dict[str, Any]:
    """Log-and-return the same startup validation the Flask app performed:
    bank phonemes map into the canonical inventory, PanPhon can vectorize every
    assessable phoneme, and the heteronym lexicon is schema-valid."""
    from src.core.persistence import db
    from src.core.g2p.g2p_service import validate_heteronym_lexicon
    from src.core.g2p.phoneme_vectors_professional import (
        panphon_available,
        validate_g2p_inventory,
        validate_panphon_inventory,
    )

    report = validate_g2p_inventory(db.get_all_bank_phonemes())
    if not report["ok"]:
        print("WARNING: exercise-bank phonemes outside canonical inventory:", report["unsupported"])

    panphon_report = validate_panphon_inventory()
    if not panphon_available():
        print("NOTE: PanPhon not installed -- scoring runs in the untrusted "
              "fallback_features mode; mastery will NOT be updated.")
    elif not panphon_report["ok"]:
        print("WARNING: PanPhon installed but cannot vectorize:", panphon_report["failures"],
              "-- scoring is UNTRUSTED (fallback_features).")
    else:
        print("PanPhon validated: all assessable phonemes vectorize -- scoring engine is TRUSTED.")

    heteronym_report = validate_heteronym_lexicon()
    if not heteronym_report["fully_supported"]:
        print("Heteronym lexicon validated with explicitly unsupported contrasts:",
              heteronym_report["unsupported_contrasts"])
    return {"inventory": report, "panphon": panphon_report, "heteronyms": heteronym_report}


def seed_exercise_bank() -> int:
    """Tag ``data/seed_sentences.txt`` with the service's own G2P pipeline and
    load the accepted sentences into the exercise bank. Returns how many were
    inserted. Idempotent: duplicates are skipped."""
    from src.core.g2p import content
    from src.core.persistence import db
    from src.core.g2p.g2p_service import g2p_convert_with_metadata, load_g2p_engine
    from src.core.g2p.tokenization import ipa_to_tokens

    load_g2p_engine()
    if not SEED_PATH.exists():
        print(f"WARNING: seed file not found: {SEED_PATH}; exercise bank left empty.")
        return 0

    with SEED_PATH.open("r", encoding="utf-8") as handle:
        sentences = [line.strip() for line in handle if line.strip()]

    inserted = 0
    for text in sentences:
        tagged = content.tag_sentence(text, g2p_convert_with_metadata, ipa_to_tokens)
        if not content.is_valid_tagging(tagged):
            continue
        sentence_id = db.insert_sentence(
            text=tagged["text"], reference_ipa=tagged["reference_ipa"],
            word_count=tagged["word_count"], level_proxy=tagged["level_proxy"],
            phoneme_counts=tagged["phoneme_counts"], source="retrieval",
        )
        if sentence_id is not None:
            inserted += 1
    return inserted


def ensure_seeded() -> None:
    """Import the current corpus once per seed version.

    Existing Modal Volumes keep their learner data and assignments. Bumping
    ``EXERCISE_SEED_VERSION`` adds only new, verified lessons because sentence
    text is unique and ``seed_exercise_bank`` skips duplicates.
    """
    from src.core.persistence import db

    current_version = db.get_service_metadata(EXERCISE_SEED_VERSION_KEY)
    if db.count_exercise_bank() == 0 or current_version != EXERCISE_SEED_VERSION:
        inserted = seed_exercise_bank()
        db.set_service_metadata(EXERCISE_SEED_VERSION_KEY, EXERCISE_SEED_VERSION)
        print(
            f"Updated exercise bank to {EXERCISE_SEED_VERSION}: "
            f"inserted {inserted} sentence(s); total {db.count_exercise_bank()}."
        )


def bootstrap() -> None:
    """Full startup sequence, safe to call more than once."""
    from src.core.persistence import db

    db.init_db()
    from src.core.g2p.g2p_service import load_g2p_engine

    # NeMo is required in normal local and deployed execution.  Do not hide a
    # broken/missing backend and then accept traffic with a different G2P mode.
    load_g2p_engine()
    ensure_seeded()
    validate_inventory()
    db.checkpoint()
