# Pronunciation AI Microservice — Architecture & Operations Report

## 1. What this is

`pronunciation_ai_service/` is a self-contained FastAPI microservice that
exposes the existing pronunciation-analysis engine (G2P → Wav2Vec2
transcription → audio-quality gating → phoneme alignment → provisional scoring →
Bayesian mastery → evidence-aware assessment → adaptive exercises) as a clean,
authenticated HTTP API.

It was extracted from the Flask application **without importing it**. The
authoritative AI modules were ported (`scoring.py`,
`mastery.py`, `assessment.py`, `tokenization.py`,
`phoneme_vectors_professional.py`, `audio_quality.py`, `g2p_service.py`,
`content.py`, `services.py`, `cleanvoice_service.py`, plus the G2P pipeline and
model config). IPA remains the canonical internal phoneme representation.
Stress-free ARPAbet is an HTTP boundary format used only in responses from the
service and in ARPAbet-keyed API inputs.

## 2. Audit of the source application

| Concern | Finding | How it is preserved here |
|---|---|---|
| Scoring formulas | DP alignment + PanPhon articulatory distance, provisional weighted PER | Preserved in IPA; response rows are converted to ARPAbet only after scoring |
| Trust gate | Mastery updates only when audio scorable **and** PanPhon-trusted **and** reference G2P trusted | Ported exactly in `api/recording.py` |
| Canonical inventory | Single IPA canonicalizer in `phoneme_vectors_professional` | Unchanged internally; `test_single_canonicalizer` passes |
| Mastery model | Per-phoneme Beta posterior, one update/recording, half-life decay | Unchanged |
| Assessment | Evidence-aware level + Monte-Carlo credible interval; honest status | Unchanged |
| Private data | 5 users, 56 attempts in `app.db` | **Not imported.** Fresh DB, subjects only |
| Audio privacy | Uploads deleted after processing | Ported: outer `finally` cleanup on every path |
| Scientific labelling | "provisional, not GOP, not CEFR" | Preserved in payloads + `/capabilities` |

## 3. Architecture boundary

```text
text  -> POS Tagger -> heteronym lexicon / NeMo IpaG2p -> reference IPA
audio -> Wav2Vec2 CTC                                  -> predicted IPA
                              |
                              v
                  IPA tokenization, PanPhon scoring,
                  mastery, exercises, and SQLite
                              |
                              v
                 api/arpabet.py response conversion
                              |
                              v
                    ARPAbet -> Django / frontend
```

NeMo and spaCy POS tagging are the preferred reference path. If either optional
runtime component cannot load, `g2p_service.py` retains the context-aware IPA
dictionary fallback and reports that fallback in `g2p_mode`.

```
┌──────────────┐   Bearer service key    ┌─────────────────────────────┐
│ Django core  │ ──────────────────────▶ │  Pronunciation AI Service   │
│ (auth, users)│   subject_id (opaque    │  FastAPI + SQLite            │
│              │    UUID), audio, text   │  domain engine (verbatim)    │
└──────────────┘ ◀────────────────────── └─────────────────────────────┘
      owns PII          {data, meta}            owns pronunciation state,
   real identities     typed envelope           keyed ONLY by opaque UUID
```

- **Django core** owns authentication and all personal records. It mints an
  opaque `subject_id` (UUID, no PII) per learner and calls this service
  server-to-server with the shared `SERVICE_API_KEY`.
- **This service** owns pronunciation-domain state (attempts, phoneme events,
  mastery, assignments, exercise bank), keyed only by that opaque UUID. It never
  learns a name, email, or anything identifying.

## 4. API contract

All `/api/v1` routes require `Authorization: Bearer <SERVICE_API_KEY>`.
Success → `{"data": ..., "meta": {service, api_version, request_id}}`.
Error → `{"error": {code, message, details}, "meta": {...}}`.
Stateful writes require an `Idempotency-Key` header.

| Method & path | Purpose |
|---|---|
| `GET /health/live` | Public liveness |
| `GET /health/ready` | Public readiness (model, G2P, DB, bank) — 200/503 |
| `GET /api/v1/capabilities` | Model/feature/limits/limitations detail |
| `POST /api/v1/g2p` | Text → stress-free ARPAbet + trust/heteronym/OOV/guide |
| `PUT /api/v1/subjects/{id}` | Idempotent anonymous profile create |
| `DELETE /api/v1/subjects/{id}` | Cascade delete all its state |
| `POST /api/v1/pronunciation/analyses` | Stateless analysis (no state) |
| `POST /api/v1/subjects/{id}/analyses` | Analyze + persist + update mastery |
| `GET /api/v1/subjects/{id}/assessment` | Evidence-aware level + interval |
| `GET /api/v1/subjects/{id}/gaps` | Ranked weak/under-observed phonemes |
| `GET /api/v1/subjects/{id}/attempts` | Cursor-paginated history |
| `POST /api/v1/exercises/generate` | Stateless exercise from metrics |
| `POST /api/v1/subjects/{id}/exercises/next` | Adaptive assignment (recorded) |

### Endpoint examples

```bash
KEY=your-service-api-key
BASE=http://127.0.0.1:8000

# G2P
curl -s -X POST $BASE/api/v1/g2p -H "Authorization: Bearer $KEY" \
     -H 'Content-Type: application/json' -d '{"text":"school"}'

# Create a subject
curl -s -X PUT $BASE/api/v1/subjects/6f1c...uuid -H "Authorization: Bearer $KEY"

# Analyze + persist (multipart)
curl -s -X POST $BASE/api/v1/subjects/6f1c...uuid/analyses \
     -H "Authorization: Bearer $KEY" -H "Idempotency-Key: 9a2f-..." \
     -F text=school -F audio=@recording.webm

# Next adaptive exercise
curl -s -X POST $BASE/api/v1/subjects/6f1c...uuid/exercises/next \
     -H "Authorization: Bearer $KEY" -H "Idempotency-Key: e13b-..."
```

### Django calling example

```python
import uuid, httpx

AI = httpx.Client(base_url=settings.PRONUNCIATION_AI_URL,
                  headers={"Authorization": f"Bearer {settings.PRONUNCIATION_AI_KEY}"})

def ensure_subject(profile):
    # profile.ai_subject_id is a locally stored opaque UUID with no PII.
    if not profile.ai_subject_id:
        profile.ai_subject_id = uuid.uuid4().hex
        profile.save(update_fields=["ai_subject_id"])
    AI.put(f"/api/v1/subjects/{profile.ai_subject_id}")
    return profile.ai_subject_id

def analyze(profile, text, audio_bytes, filename, mimetype):
    sid = ensure_subject(profile)
    r = AI.post(
        f"/api/v1/subjects/{sid}/analyses",
        data={"text": text},
        files={"audio": (filename, audio_bytes, mimetype)},
        headers={"Idempotency-Key": uuid.uuid4().hex},  # retry-safe
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["data"]
```

## 5. Data model

Fresh SQLite, seeded with the public exercise corpus on first boot. No users;
`subjects` are opaque UUIDs.

- `subjects(id, subject_id UNIQUE, created_at)` — `id` is a private internal
  surrogate; `subject_id` is the Django UUID.
- `attempts(subject_pk → subjects, exercise_id, text, reference_ipa,
  predicted_ipa, metrics, scorable, scoring_engine, scoring_trusted,
  mastery_updated, g2p_mode, reference_g2p_trusted, ...)`
- `attempt_phoneme_events(attempt_id → attempts, expected/spoken, operation,
  articulatory_distance, alignment_cost, scoring_engine)`
- `phoneme_skill_state(subject_pk, phoneme, alpha, beta, counts, last_practiced_at)`
- `practice_assignments(subject_pk, exercise_id, target_phonemes, completed_attempt_id)`
- `exercise_bank(id, text UNIQUE, reference_ipa, ...)` + `sentence_phonemes`
- `idempotency_keys(scope, key, subject_pk, attempt_id, response_json)` — the
  retry-dedup ledger.

All child tables use `ON DELETE CASCADE`; deleting a subject removes every trace
of it while leaving the shared exercise bank intact.

## 6. Security & privacy

- **Auth**: single shared `SERVICE_API_KEY`, compared constant-time. Missing key
  → 503 (fail closed). No end-user auth is this service's concern.
- **No PII**: only opaque UUIDs are stored. No names, emails, or free-form
  identity fields exist in the schema.
- **Audio is temporary**: the original upload, converted WAV, Cleanvoice output,
  and reduced WAV are deleted in an outer `finally` on every success and failure.
  Processed audio is never returned in a response and never logged. There is no
  uploads route.
- **Streamed uploads**: recordings stream to a spooled temp file with a 20 MB
  cap, never fully buffered in memory.
- **No leaking internals**: the catch-all handler returns a generic message;
  Cleanvoice SDK errors (which can contain signed URLs) are wrapped.
- **CORS**: off by default (server-to-server). Opt in via `CORS_ALLOW_ORIGINS`.

## 7. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SERVICE_API_KEY` | *(unset → 503)* | Shared bearer secret (**required**) |
| `PRONUNCIATION_DB_PATH` | `./pronunciation_ai.db` | SQLite path |
| `PRONUNCIATION_UPLOAD_DIR` | `./uploads` | Ephemeral upload dir |
| `MAX_AUDIO_BYTES` | `20971520` | Upload cap (20 MB) |
| `MAX_AUDIO_SECONDS` | `60` | Advertised duration cap |
| `CORS_ALLOW_ORIGINS` | *(none)* | Comma-separated origins (opt-in) |
| `RETAIN_AUDIO` | `0` | Debug only; keep audio (never in prod) |
| `CLEANVOICE_API_KEY` / `CLEANVOICE_ENABLED` / `CLEANVOICE_STRICT` | off | Optional enhancement |
| `GEMINI_API_KEY` (or `LLM_API_KEY`, ...) | off | Optional LLM exercise generation |

## 8. Modal operation

`modal_app.py` publishes the FastAPI app via `@modal.asgi_app()` on a CPU image.

```bash
# One-time: create the secret with the service key (+ optional providers)
modal secret create pronunciation-ai-secrets SERVICE_API_KEY=... [GEMINI_API_KEY=...]

# Develop against a live URL
modal serve modal_app.py

# Deploy (explicit; never automatic)
modal deploy modal_app.py
```

- The 378 MB model weight ships in its own cached image layer (via
  `add_local_dir(..., copy=True)`); source-only changes don't re-upload it.
- The database lives on a Volume at `/data/pronunciation_ai.db`. After each
  stateful transaction the WAL is checkpointed and the Volume committed.
- **Concurrency limit (required):** Modal Volumes are not intended for
  concurrent modification of a single SQLite file. v1 therefore runs
  `max_containers=1` with `@modal.concurrent(max_inputs=1)` and a 600 s request
  timeout — one request at a time.

## 9. Scientific limitations (unchanged, surfaced honestly)

- Scores derive from **PanPhon articulatory distance**, a descriptive similarity,
  **not** a calibrated Goodness-of-Pronunciation probability, and **not** a CEFR
  level. Every level payload is labelled `provisional` with an explicit note.
- Mastery is a per-phoneme Beta posterior with half-life decay; the overall
  score is a Monte-Carlo posterior **credible interval**, and levels are assigned
  only when that interval sits inside one band (otherwise `uncertain`).
- If PanPhon cannot vectorize every assessable phoneme, the engine reports
  `fallback_features` and **mastery is never updated** — the service degrades
  loudly, never silently.

## 10. Scaling path (SQLite → PostgreSQL → many containers)

v1 is deliberately single-writer. To scale horizontally:

1. Replace the SQLite `db.py` with a PostgreSQL-backed implementation keeping the
   same function surface (subject-keyed reads/writes, atomic recording,
   idempotency ledger). The domain modules need no change — they only see the
   `db` interface.
2. Move the idempotency ledger + atomic recording into Postgres transactions
   (the same `INSERT ... UNIQUE(scope,key)` guard works, and gives true
   concurrent-safe dedup).
3. Remove the Modal single-container / single-input constraints and raise
   `max_containers`.
4. Keep uploads ephemeral and local to each container (already the case).

Until then, PostgreSQL is a prerequisite for enabling multiple containers.
