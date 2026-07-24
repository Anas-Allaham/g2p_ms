"""
SQLite persistence for the anonymous pronunciation AI microservice.

This is the subject-oriented successor to the Flask app's ``db.py``. The
authoritative domain modules (``services.py``, ``assessment.py``,
``mastery.py``) are reused VERBATIM, so this module keeps their internal
contract: every domain-facing function still takes an integer ``user_id``.

The difference is what that integer means. There are no named users here. A
``subject`` is an anonymous pronunciation profile keyed by an opaque
Django-generated UUID (``subject_id``) that carries no personal information.
``subjects.id`` is a private, internal integer surrogate; the API layer
resolves a ``subject_id`` UUID to that surrogate and passes it to the domain
code as ``user_id``. Nothing outside this module ever sees the surrogate.

New responsibilities vs. the Flask schema:
  * subjects keyed by opaque UUID (idempotent create).
  * cascade deletion of a subject's attempts, events, mastery, assignments.
  * idempotency keys so a Django retry can never duplicate mastery evidence
    or an assignment.
  * cursor-paginated attempt history.

Kept dependency-free (stdlib ``sqlite3`` only) so scripts and tests can import
it without FastAPI.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.core.paths import PROJECT_ROOT

# The default DB lives at the project root (not next to this module), so it is
# stable regardless of where persistence code is packaged.
DB_PATH = Path(
    os.environ.get("PRONUNCIATION_DB_PATH", str(PROJECT_ROOT / "pronunciation_ai.db"))
).expanduser()

_connection: Optional[sqlite3.Connection] = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS exercise_bank (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL UNIQUE,
    reference_ipa TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    level_proxy REAL NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'retrieval',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sentence_phonemes (
    sentence_id INTEGER NOT NULL REFERENCES exercise_bank(id) ON DELETE CASCADE,
    phoneme TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (sentence_id, phoneme)
);

CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_pk INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    exercise_id INTEGER REFERENCES exercise_bank(id),
    text TEXT NOT NULL,
    reference_ipa TEXT NOT NULL,
    predicted_ipa TEXT NOT NULL,
    phoneme_error_rate REAL NOT NULL,
    weighted_error REAL NOT NULL,
    raw_weighted_per REAL,
    quality_weight REAL,
    scorable INTEGER NOT NULL DEFAULT 1,
    rejected_reason TEXT,
    scoring_engine TEXT,
    scoring_trusted INTEGER NOT NULL DEFAULT 0,
    mastery_updated INTEGER NOT NULL DEFAULT 0,
    insertion_count INTEGER NOT NULL DEFAULT 0,
    reference_unit_count INTEGER NOT NULL DEFAULT 0,
    g2p_mode TEXT,
    reference_g2p_trusted INTEGER NOT NULL DEFAULT 0,
    reference_g2p_reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS attempt_phoneme_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id INTEGER NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    expected_phoneme TEXT,
    spoken_phoneme TEXT,
    operation TEXT NOT NULL,
    distance REAL NOT NULL,
    articulatory_distance REAL,
    alignment_cost REAL,
    scoring_engine TEXT
);

CREATE TABLE IF NOT EXISTS phoneme_skill_state (
    subject_pk INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    phoneme TEXT NOT NULL,
    alpha REAL NOT NULL DEFAULT 1.0,
    beta REAL NOT NULL DEFAULT 1.0,
    attempts_count INTEGER NOT NULL DEFAULT 0,
    occurrence_count INTEGER NOT NULL DEFAULT 0,
    last_practiced_at TEXT,
    PRIMARY KEY (subject_pk, phoneme)
);

CREATE TABLE IF NOT EXISTS practice_assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_pk INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    exercise_id INTEGER NOT NULL REFERENCES exercise_bank(id),
    target_phonemes TEXT NOT NULL,
    assigned_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_attempt_id INTEGER REFERENCES attempts(id)
);

-- Idempotency ledger. A stateful write (analysis or exercise assignment) that
-- carries an Idempotency-Key first reserves (scope, key) here; the UNIQUE
-- primary key makes a duplicate retry fail atomically instead of duplicating
-- mastery evidence or an assignment. The stored response is replayed verbatim.
CREATE TABLE IF NOT EXISTS idempotency_keys (
    scope TEXT NOT NULL,
    key TEXT NOT NULL,
    subject_pk INTEGER REFERENCES subjects(id) ON DELETE CASCADE,
    attempt_id INTEGER REFERENCES attempts(id) ON DELETE CASCADE,
    response_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (scope, key)
);

CREATE INDEX IF NOT EXISTS idx_attempt_phoneme_events_attempt
    ON attempt_phoneme_events(attempt_id);
CREATE INDEX IF NOT EXISTS idx_attempts_subject
    ON attempts(subject_pk, id);
CREATE INDEX IF NOT EXISTS idx_practice_assignments_subject
    ON practice_assignments(subject_pk, assigned_at);
"""


class IdempotencyConflict(RuntimeError):
    """Raised when a stateful write reuses an Idempotency-Key whose stored
    response is not yet available (a genuinely concurrent duplicate)."""


def get_connection() -> sqlite3.Connection:
    """Lazily-opened, process-wide connection. WAL mode keeps a single local
    writer safe at this service's v1 scale (one Modal container, one active
    request at a time)."""
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA journal_mode=WAL")
        _connection.execute("PRAGMA foreign_keys=ON")
    return _connection


def init_db(conn: Optional[sqlite3.Connection] = None) -> None:
    conn = conn or get_connection()
    conn.executescript(SCHEMA)
    conn.commit()


def checkpoint(conn: Optional[sqlite3.Connection] = None) -> None:
    """Fold the WAL back into the main database file. Called before committing
    a Modal Volume so the persisted file is self-contained."""
    conn = conn or get_connection()
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass


def set_database_for_testing(path: "str | Path") -> None:
    """Point the module at a different SQLite file (tests use a temp DB so they
    never touch a real database). Closes any open connection first."""
    global _connection, DB_PATH
    if _connection is not None:
        try:
            _connection.close()
        except Exception:
            pass
    _connection = None
    DB_PATH = Path(path)


# -----------------------------------------------------------------------------
# Subjects — anonymous, opaque-UUID profiles (no personal data, no names).
# -----------------------------------------------------------------------------
def get_or_create_subject(subject_id: str, conn: Optional[sqlite3.Connection] = None) -> sqlite3.Row:
    """Idempotently create the anonymous profile for an opaque subject UUID and
    return its row. Safe to call on every request."""
    conn = conn or get_connection()
    subject_id = str(subject_id).strip()
    existing = conn.execute("SELECT * FROM subjects WHERE subject_id = ?", (subject_id,)).fetchone()
    if existing is not None:
        return existing
    conn.execute("INSERT OR IGNORE INTO subjects (subject_id) VALUES (?)", (subject_id,))
    conn.commit()
    return conn.execute("SELECT * FROM subjects WHERE subject_id = ?", (subject_id,)).fetchone()


def get_subject(subject_id: str, conn: Optional[sqlite3.Connection] = None) -> Optional[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute(
        "SELECT * FROM subjects WHERE subject_id = ?", (str(subject_id).strip(),)
    ).fetchone()


def subject_pk(subject_id: str, conn: Optional[sqlite3.Connection] = None) -> Optional[int]:
    row = get_subject(subject_id, conn)
    return int(row["id"]) if row is not None else None


def delete_subject(subject_id: str, conn: Optional[sqlite3.Connection] = None) -> bool:
    """Cascade-delete a subject and ALL of its pronunciation-domain state:
    attempts, phoneme events, mastery, assignments, and idempotency records.
    Returns True if a subject existed. The shared exercise bank is untouched."""
    conn = conn or get_connection()
    row = get_subject(subject_id, conn)
    if row is None:
        return False
    pk = int(row["id"])
    with conn:  # single transaction; ON DELETE CASCADE handles children
        conn.execute("DELETE FROM subjects WHERE id = ?", (pk,))
    return True


# -----------------------------------------------------------------------------
# Attempts + raw phoneme events (private per subject)
# -----------------------------------------------------------------------------
def _insert_attempt(
    conn: sqlite3.Connection,
    user_id: int,
    text: str,
    reference_ipa: str,
    predicted_ipa: str,
    phoneme_error_rate: float,
    weighted_error: float,
    exercise_id: Optional[int] = None,
    raw_weighted_per: Optional[float] = None,
    quality_weight: Optional[float] = None,
    scorable: bool = True,
    rejected_reason: Optional[str] = None,
    scoring_engine: Optional[str] = None,
    scoring_trusted: bool = False,
    mastery_updated: bool = False,
    insertion_count: int = 0,
    reference_unit_count: int = 0,
    g2p_mode: Optional[str] = None,
    reference_g2p_trusted: bool = False,
    reference_g2p_reason: Optional[str] = None,
) -> int:
    """Insert one attempt row WITHOUT committing (transaction-friendly).
    ``user_id`` is the internal subjects.id surrogate."""
    cur = conn.execute(
        """INSERT INTO attempts
           (subject_pk, exercise_id, text, reference_ipa, predicted_ipa,
            phoneme_error_rate, weighted_error, raw_weighted_per,
            quality_weight, scorable, rejected_reason,
            scoring_engine, scoring_trusted, mastery_updated, insertion_count,
            reference_unit_count, g2p_mode, reference_g2p_trusted, reference_g2p_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id, exercise_id, text, reference_ipa, predicted_ipa,
            phoneme_error_rate, weighted_error, raw_weighted_per,
            quality_weight, 1 if scorable else 0, rejected_reason,
            scoring_engine, 1 if scoring_trusted else 0,
            1 if mastery_updated else 0, insertion_count, reference_unit_count,
            g2p_mode, 1 if reference_g2p_trusted else 0, reference_g2p_reason,
        ),
    )
    return cur.lastrowid


def _insert_events(
    conn: sqlite3.Connection,
    attempt_id: int,
    alignment: List[Dict[str, Any]],
    scoring_engine: Optional[str] = None,
) -> None:
    """Insert alignment events WITHOUT committing (transaction-friendly)."""
    rows = []
    for position, row in enumerate(alignment):
        art = row.get("articulatory_distance", row.get("distance"))
        cost = row.get("alignment_cost")
        compat_distance = art if art is not None else (cost if cost is not None else 0.0)
        rows.append((
            attempt_id,
            position,
            None if row.get("expected") in (None, "-") else row["expected"],
            None if row.get("spoken") in (None, "-") else row["spoken"],
            row.get("result"),
            float(compat_distance),
            None if art is None else float(art),
            None if cost is None else float(cost),
            scoring_engine,
        ))
    conn.executemany(
        """INSERT INTO attempt_phoneme_events
           (attempt_id, position, expected_phoneme, spoken_phoneme, operation,
            distance, articulatory_distance, alignment_cost, scoring_engine)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )


def record_attempt(conn: Optional[sqlite3.Connection] = None, **kwargs) -> int:
    """Insert one attempt and commit. Prefer ``record_recording_atomic`` for the
    full attempt+events+mastery+assignment write. ``user_id`` is subjects.id."""
    conn = conn or get_connection()
    attempt_id = _insert_attempt(conn, **kwargs)
    conn.commit()
    return attempt_id


def record_phoneme_events(
    attempt_id: int,
    alignment: List[Dict[str, Any]],
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    conn = conn or get_connection()
    _insert_events(conn, attempt_id, alignment)
    conn.commit()


def record_recording_atomic(
    user_id: int,
    attempt_kwargs: Dict[str, Any],
    alignment: List[Dict[str, Any]],
    scoring_engine: Optional[str] = None,
    phoneme_states: Optional[Dict[str, Dict[str, Any]]] = None,
    complete_exercise_id: Optional[int] = None,
    idempotency: Optional[Tuple[str, str]] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Write attempt + events + mastery + assignment completion (+ idempotency
    reservation) in ONE atomic SQLite transaction. If any step raises, the whole
    write rolls back, so a partial failure can never leave inconsistent
    evidence and a duplicate Idempotency-Key can never double-count mastery.

    ``idempotency``: optional ``(scope, key)``. Reserved inside the same
    transaction; a duplicate key raises ``sqlite3.IntegrityError`` and rolls
    everything back (the caller replays the stored response instead)."""
    conn = conn or get_connection()
    with conn:  # BEGIN; commits on success, rolls back on any exception
        attempt_id = _insert_attempt(conn, user_id=user_id, **attempt_kwargs)
        _insert_events(conn, attempt_id, alignment, scoring_engine)
        if phoneme_states:
            for phoneme, st in phoneme_states.items():
                _upsert_phoneme_state(
                    conn, user_id, phoneme,
                    st["alpha"], st["beta"], st["attempts_count"],
                    st["occurrence_count"], st["last_practiced_at"],
                )
        if complete_exercise_id is not None:
            _complete_latest_assignment(conn, user_id, complete_exercise_id, attempt_id)
        if idempotency is not None:
            scope, key = idempotency
            conn.execute(
                "INSERT INTO idempotency_keys (scope, key, subject_pk, attempt_id) "
                "VALUES (?, ?, ?, ?)",
                (scope, key, user_id, attempt_id),
            )
    return attempt_id


def get_attempt_phoneme_events(attempt_id: int, conn: Optional[sqlite3.Connection] = None) -> List[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute(
        "SELECT * FROM attempt_phoneme_events WHERE attempt_id = ? ORDER BY position",
        (attempt_id,),
    ).fetchall()


def get_subject_attempts_page(
    user_id: int,
    limit: int = 20,
    before_id: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> List[sqlite3.Row]:
    """Cursor-paginated history, newest first. ``before_id`` is an opaque
    cursor: return attempts with ``id < before_id``. Fetches ``limit + 1`` rows
    so the caller can tell whether another page exists."""
    conn = conn or get_connection()
    if before_id is not None:
        return conn.execute(
            "SELECT * FROM attempts WHERE subject_pk = ? AND id < ? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, before_id, limit + 1),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM attempts WHERE subject_pk = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit + 1),
    ).fetchall()


# Kept for parity with the Flask persistence API / any domain caller.
def get_user_attempts(user_id: int, limit: int = 20, conn: Optional[sqlite3.Connection] = None) -> List[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute(
        "SELECT * FROM attempts WHERE subject_pk = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()


# -----------------------------------------------------------------------------
# Phoneme skill state (mastery cache)
# -----------------------------------------------------------------------------
def get_phoneme_state(user_id: int, phoneme: str, conn: Optional[sqlite3.Connection] = None) -> Optional[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute(
        "SELECT * FROM phoneme_skill_state WHERE subject_pk = ? AND phoneme = ?",
        (user_id, phoneme),
    ).fetchone()


def get_all_phoneme_states(user_id: int, conn: Optional[sqlite3.Connection] = None) -> List[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute(
        "SELECT * FROM phoneme_skill_state WHERE subject_pk = ?", (user_id,)
    ).fetchall()


def upsert_phoneme_state(
    user_id: int,
    phoneme: str,
    alpha: float,
    beta: float,
    attempts_count: int,
    last_practiced_at: str,
    occurrence_count: int = 0,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    conn = conn or get_connection()
    _upsert_phoneme_state(conn, user_id, phoneme, alpha, beta, attempts_count, occurrence_count, last_practiced_at)
    conn.commit()


def _upsert_phoneme_state(
    conn: sqlite3.Connection,
    user_id: int,
    phoneme: str,
    alpha: float,
    beta: float,
    attempts_count: int,
    occurrence_count: int,
    last_practiced_at: str,
) -> None:
    """Upsert one phoneme state WITHOUT committing (transaction-friendly)."""
    conn.execute(
        """INSERT INTO phoneme_skill_state
               (subject_pk, phoneme, alpha, beta, attempts_count, occurrence_count, last_practiced_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(subject_pk, phoneme) DO UPDATE SET
               alpha=excluded.alpha,
               beta=excluded.beta,
               attempts_count=excluded.attempts_count,
               occurrence_count=excluded.occurrence_count,
               last_practiced_at=excluded.last_practiced_at""",
        (user_id, phoneme, alpha, beta, attempts_count, occurrence_count, last_practiced_at),
    )


# -----------------------------------------------------------------------------
# Exercise bank (shared content; not per subject)
# -----------------------------------------------------------------------------
def insert_sentence(
    text: str,
    reference_ipa: str,
    word_count: int,
    level_proxy: float,
    phoneme_counts: Dict[str, int],
    source: str = "retrieval",
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[int]:
    """Insert a sentence + its phoneme tags. Returns None (no-op) if the exact
    sentence text is already in the bank."""
    conn = conn or get_connection()
    existing = conn.execute("SELECT id FROM exercise_bank WHERE text = ?", (text,)).fetchone()
    if existing:
        return None
    cur = conn.execute(
        """INSERT INTO exercise_bank (text, reference_ipa, word_count, level_proxy, source)
           VALUES (?, ?, ?, ?, ?)""",
        (text, reference_ipa, word_count, level_proxy, source),
    )
    sentence_id = cur.lastrowid
    conn.executemany(
        "INSERT INTO sentence_phonemes (sentence_id, phoneme, count) VALUES (?, ?, ?)",
        [(sentence_id, phoneme, count) for phoneme, count in phoneme_counts.items()],
    )
    conn.commit()
    return sentence_id


def get_sentence_by_id(sentence_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute("SELECT * FROM exercise_bank WHERE id = ?", (sentence_id,)).fetchone()


def get_sentence_by_text(text: str, conn: Optional[sqlite3.Connection] = None) -> Optional[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute("SELECT * FROM exercise_bank WHERE text = ?", (text,)).fetchone()


def update_sentence_tags(
    sentence_id: int,
    reference_ipa: str,
    phoneme_counts: Dict[str, int],
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    conn = conn or get_connection()
    conn.execute(
        "UPDATE exercise_bank SET reference_ipa = ? WHERE id = ?",
        (reference_ipa, sentence_id),
    )
    conn.execute("DELETE FROM sentence_phonemes WHERE sentence_id = ?", (sentence_id,))
    conn.executemany(
        "INSERT INTO sentence_phonemes (sentence_id, phoneme, count) VALUES (?, ?, ?)",
        [(sentence_id, phoneme, count) for phoneme, count in phoneme_counts.items()],
    )
    conn.commit()


def get_sentence_phonemes(sentence_id: int, conn: Optional[sqlite3.Connection] = None) -> Dict[str, int]:
    conn = conn or get_connection()
    rows = conn.execute(
        "SELECT phoneme, count FROM sentence_phonemes WHERE sentence_id = ?", (sentence_id,)
    ).fetchall()
    return {row["phoneme"]: row["count"] for row in rows}


def get_sentences_covering_any(phonemes: Iterable[str], conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    conn = conn or get_connection()
    phonemes = list(phonemes)
    if not phonemes:
        return []
    placeholders = ",".join("?" for _ in phonemes)
    sentence_ids = [
        row["sentence_id"]
        for row in conn.execute(
            f"SELECT DISTINCT sentence_id FROM sentence_phonemes WHERE phoneme IN ({placeholders})",
            phonemes,
        ).fetchall()
    ]
    return [_load_sentence_with_phonemes(sid, conn) for sid in sentence_ids]


def get_all_sentences(conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    conn = conn or get_connection()
    ids = [row["id"] for row in conn.execute("SELECT id FROM exercise_bank").fetchall()]
    return [_load_sentence_with_phonemes(sid, conn) for sid in ids]


def count_exercise_bank(conn: Optional[sqlite3.Connection] = None) -> int:
    conn = conn or get_connection()
    row = conn.execute("SELECT COUNT(*) AS n FROM exercise_bank").fetchone()
    return int(row["n"]) if row else 0


def get_all_bank_phonemes(conn: Optional[sqlite3.Connection] = None) -> List[str]:
    conn = conn or get_connection()
    return [row["phoneme"] for row in conn.execute(
        "SELECT DISTINCT phoneme FROM sentence_phonemes"
    ).fetchall()]


def _load_sentence_with_phonemes(sentence_id: int, conn: sqlite3.Connection) -> Dict[str, Any]:
    sentence = get_sentence_by_id(sentence_id, conn)
    phoneme_counts = get_sentence_phonemes(sentence_id, conn)
    return {
        "id": sentence["id"],
        "text": sentence["text"],
        "reference_ipa": sentence["reference_ipa"],
        "word_count": sentence["word_count"],
        "level_proxy": sentence["level_proxy"],
        "source": sentence["source"],
        "phoneme_counts": phoneme_counts,
    }


# -----------------------------------------------------------------------------
# Practice assignments (what was served, and how it scored)
# -----------------------------------------------------------------------------
def record_practice_assignment(
    user_id: int,
    exercise_id: int,
    target_phonemes: List[str],
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    conn = conn or get_connection()
    cur = conn.execute(
        "INSERT INTO practice_assignments (subject_pk, exercise_id, target_phonemes) VALUES (?, ?, ?)",
        (user_id, exercise_id, json.dumps(target_phonemes)),
    )
    conn.commit()
    return cur.lastrowid


def complete_latest_assignment(
    user_id: int,
    exercise_id: int,
    attempt_id: int,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    conn = conn or get_connection()
    _complete_latest_assignment(conn, user_id, exercise_id, attempt_id)
    conn.commit()


def _complete_latest_assignment(
    conn: sqlite3.Connection, user_id: int, exercise_id: int, attempt_id: int
) -> None:
    row = conn.execute(
        """SELECT id FROM practice_assignments
           WHERE subject_pk = ? AND exercise_id = ? AND completed_attempt_id IS NULL
           ORDER BY assigned_at DESC LIMIT 1""",
        (user_id, exercise_id),
    ).fetchone()
    if row is None:
        return
    conn.execute(
        "UPDATE practice_assignments SET completed_attempt_id = ? WHERE id = ?",
        (attempt_id, row["id"]),
    )


def get_recently_served_sentence_ids(user_id: int, limit: int = 15, conn: Optional[sqlite3.Connection] = None) -> set:
    conn = conn or get_connection()
    rows = conn.execute(
        """SELECT exercise_id FROM practice_assignments
           WHERE subject_pk = ? ORDER BY assigned_at DESC LIMIT ?""",
        (user_id, limit),
    ).fetchall()
    return {row["exercise_id"] for row in rows}


# -----------------------------------------------------------------------------
# Idempotency ledger
# -----------------------------------------------------------------------------
def get_idempotent_response(scope: str, key: str, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
    """Return the stored response for a previously-seen (scope, key), or None.
    Raises ``IdempotencyConflict`` if the key was reserved but its response is
    not yet stored (a genuinely in-flight duplicate)."""
    conn = conn or get_connection()
    row = conn.execute(
        "SELECT response_json FROM idempotency_keys WHERE scope = ? AND key = ?",
        (scope, key),
    ).fetchone()
    if row is None:
        return None
    if row["response_json"] is None:
        raise IdempotencyConflict(f"Idempotency-Key '{key}' is still being processed.")
    return json.loads(row["response_json"])


def reserve_idempotency_key(scope: str, key: str, subject_pk_value: Optional[int] = None,
                            conn: Optional[sqlite3.Connection] = None) -> bool:
    """Reserve (scope, key) for a non-attempt stateful write. Returns True when
    freshly reserved, False if it already exists (duplicate)."""
    conn = conn or get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO idempotency_keys (scope, key, subject_pk) VALUES (?, ?, ?)",
                (scope, key, subject_pk_value),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def save_idempotent_response(scope: str, key: str, response: Dict[str, Any],
                             conn: Optional[sqlite3.Connection] = None) -> None:
    conn = conn or get_connection()
    with conn:
        conn.execute(
            "UPDATE idempotency_keys SET response_json = ? WHERE scope = ? AND key = ?",
            (json.dumps(response), scope, key),
        )


def release_idempotency_key(scope: str, key: str, conn: Optional[sqlite3.Connection] = None) -> None:
    """Drop a reservation whose work failed, so a later retry can start fresh
    instead of being stuck as a permanent in-flight duplicate."""
    conn = conn or get_connection()
    with conn:
        conn.execute(
            "DELETE FROM idempotency_keys WHERE scope = ? AND key = ? AND response_json IS NULL",
            (scope, key),
        )


# -----------------------------------------------------------------------------
# Evidence / confusion aggregation for assessment + confusion-aware exercises
# (identical semantics to the Flask app; `subject_pk` is the internal id)
# -----------------------------------------------------------------------------
def _trusted_filter() -> str:
    return "a.mastery_updated = 1"


def get_phoneme_context_stats(
    user_id: int, conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Dict[str, Any]]:
    conn = conn or get_connection()
    rows = conn.execute(
        f"""SELECT e.expected_phoneme AS phoneme,
                   a.id AS attempt_id,
                   a.text AS prompt_text,
                   COALESCE(a.quality_weight, 1.0) AS quality_weight
            FROM attempt_phoneme_events e
            JOIN attempts a ON e.attempt_id = a.id
            WHERE a.subject_pk = ?
              AND e.expected_phoneme IS NOT NULL
              AND {_trusted_filter()}""",
        (user_id,),
    ).fetchall()
    accumulated: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        state = accumulated.setdefault(
            row["phoneme"],
            {"attempt_ids": set(), "prompt_texts": set(), "attempt_weights": {}, "occurrences": 0},
        )
        state["attempt_ids"].add(row["attempt_id"])
        state["prompt_texts"].add(row["prompt_text"])
        state["attempt_weights"][row["attempt_id"]] = max(
            0.0, min(1.0, float(row["quality_weight"]))
        )
        state["occurrences"] += 1
    return {
        phoneme: {
            "recordings": len(state["attempt_ids"]),
            "effective_recordings": round(sum(state["attempt_weights"].values()), 6),
            "distinct_prompts": len(state["prompt_texts"]),
            "occurrences": state["occurrences"],
        }
        for phoneme, state in accumulated.items()
    }


def get_confusion_pairs(
    user_id: int, limit: Optional[int] = None, conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    conn = conn or get_connection()
    query = f"""SELECT e.expected_phoneme AS expected,
                       e.spoken_phoneme AS spoken,
                       COUNT(*) AS count
                FROM attempt_phoneme_events e
                JOIN attempts a ON e.attempt_id = a.id
                WHERE a.subject_pk = ?
                  AND e.operation LIKE '%substitution%'
                  AND e.expected_phoneme IS NOT NULL
                  AND e.spoken_phoneme IS NOT NULL
                  AND {_trusted_filter()}
                GROUP BY e.expected_phoneme, e.spoken_phoneme
                ORDER BY count DESC"""
    params: tuple = (user_id,)
    if limit is not None:
        query += " LIMIT ?"
        params = (user_id, limit)
    rows = conn.execute(query, params).fetchall()
    return [
        {"expected": row["expected"], "spoken": row["spoken"], "count": row["count"]}
        for row in rows
    ]


def get_trusted_recording_count(user_id: int, conn: Optional[sqlite3.Connection] = None) -> int:
    conn = conn or get_connection()
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM attempts a WHERE a.subject_pk = ? AND {_trusted_filter()}",
        (user_id,),
    ).fetchone()
    return int(row["n"]) if row else 0


def get_effective_recording_count(user_id: int, conn: Optional[sqlite3.Connection] = None) -> float:
    conn = conn or get_connection()
    row = conn.execute(
        f"""SELECT COALESCE(SUM(
                    CASE
                        WHEN a.quality_weight IS NULL THEN 1.0
                        WHEN a.quality_weight < 0 THEN 0.0
                        WHEN a.quality_weight > 1 THEN 1.0
                        ELSE a.quality_weight
                    END
                ), 0.0) AS n
            FROM attempts a
            WHERE a.subject_pk = ? AND {_trusted_filter()}""",
        (user_id,),
    ).fetchone()
    return float(row["n"]) if row else 0.0


def get_utterance_epenthesis_state(
    user_id: int, conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Any]:
    conn = conn or get_connection()
    rows = conn.execute(
        f"""SELECT a.id, a.insertion_count, a.reference_unit_count,
                   COALESCE(a.quality_weight, 1.0) AS quality_weight,
                   (SELECT COUNT(*) FROM attempt_phoneme_events e
                    WHERE e.attempt_id = a.id AND e.expected_phoneme IS NOT NULL) AS event_ref_units
            FROM attempts a
            WHERE a.subject_pk = ? AND {_trusted_filter()}""",
        (user_id,),
    ).fetchall()

    alpha = beta = 1.0
    effective_evidence = 0.0
    total_insertions = 0
    included = 0
    for row in rows:
        reference_units = int(row["reference_unit_count"] or row["event_ref_units"] or 0)
        if reference_units <= 0:
            continue
        insertions = max(0, int(row["insertion_count"] or 0))
        weight = max(0.0, min(1.0, float(row["quality_weight"])))
        observation = max(0.0, 1.0 - (insertions / reference_units))
        alpha += weight * observation
        beta += weight * (1.0 - observation)
        effective_evidence += weight
        total_insertions += insertions
        included += 1

    return {
        "alpha": alpha,
        "beta": beta,
        "recordings": included,
        "effective_recordings": round(effective_evidence, 6),
        "insertion_count": total_insertions,
    }
