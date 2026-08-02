# Pronunciation AI Service

A self-contained FastAPI microservice wrapping the pronunciation-analysis engine
(G2P → Wav2Vec2 → audio-quality gate → phoneme alignment → provisional scoring →
Bayesian mastery → evidence-aware assessment → adaptive exercises).

The phonetic engine remains IPA end to end: the POS-aware heteronym resolver
and NeMo produce the reference IPA, Wav2Vec2 predicts IPA, and PanPhon,
alignment, mastery, exercises, and persistence all operate on IPA. Only the
public API boundary converts phoneme-bearing response fields to uppercase,
stress-free **ARPAbet** (for example, `S K UW L | IH Z`) for Django/frontend
clients. `|` marks word boundaries and `AX` represents schwa.

The public boundary validates every emitted token against the declared
stress-free ARPAbet inventory. Formatted reference IPA uses explicit `|` word
boundaries, while raw CTC output treats whitespace as acoustic word boundaries;
the adapter no longer guesses which format it received. Unsupported reference
or API tokens fail explicitly, while CTC-only recovery is non-fatal and logged.

IPA→ARPAbet→IPA is intentionally canonicalizing rather than lossless. It may
normalize vowel length, allophones, and dialect variants to the service's
American-English-oriented internal inventory.

```text
text  -> POS Tagger -> heteronym lexicon / NeMo -> reference IPA
audio -> Wav2Vec2                              -> predicted IPA
                         IPA alignment + PanPhon + persistence
                                           |
                                           v
                                API IPA -> ARPAbet adapter
                                           |
                                           v
                                    Django / frontend
```

It is called **server-to-server** by a Django core that owns authentication and
user identity. This service stores pronunciation state only under an opaque
`subject_id` (a UUID with no personal data). It does not import or modify the
original Flask app.

See [`MICROSERVICE_REPORT.md`](MICROSERVICE_REPORT.md) for the full architecture,
data model, security/privacy, Modal operation, and scaling notes.

## Quick start (local)

```bash
cd pronunciation_ai_service
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp .env.example .env
#   set SERVICE_API_KEY to a long random secret

# Supply the trained model weight (distributed separately, gitignored):
#   model/my_wav2vec2_phoneme_model/model.safetensors

uvicorn api.main:app --reload
```

- Interactive docs: <http://127.0.0.1:8000/docs>
- OpenAPI: <http://127.0.0.1:8000/openapi.json>
- Liveness: `GET /health/live` · Readiness: `GET /health/ready`

The exercise bank is seeded automatically on first startup from
`data/seed_sentences.txt`. The database starts empty of subjects.

## Auth

Every `/api/v1` route requires `Authorization: Bearer <SERVICE_API_KEY>`.
Stateful writes (`/subjects/{id}/analyses`, `/subjects/{id}/exercises/next`)
also require an `Idempotency-Key` header so retries never duplicate evidence.

## Pronunciation error ranges

Both analysis endpoints include an additive `pronunciation_errors` array. Each
item links a non-correct phoneme alignment row to the relevant spelling in the
trimmed response `text`:

```json
{
  "alignment_index": 1,
  "operation": "substitution",
  "result": "major_substitution",
  "expected": "K",
  "spoken": "T",
  "word_index": 1,
  "reference_span": {
    "start": 1,
    "end": 2,
    "text": "ch",
    "kind": "grapheme"
  }
}
```

`start` and `end` are inclusive browser-compatible UTF-16 code-unit indexes, so
JavaScript clients can read a range with `text.slice(start, end + 1)`.
`word_fallback` spans cover the whole word when English spelling cannot be
mapped confidently; insertions use an empty `boundary` marker with equal
indexes and should not be sliced as a letter range.

## Tests

```bash
python -m pytest            # mocked acoustic layer; no model weight required
```

The suite covers the ported domain logic (scoring, mastery, assessment,
tokenization, G2P, content, audio-quality gate) plus FastAPI contract tests
(auth, envelopes, request ids, media types, OpenAPI), idempotent retries,
subject isolation/deletion, the mastery trust gate, exercise assignment, history
pagination, and temporary-audio cleanup on every failure stage.

An optional real-model smoke test uses the bundled WAV fixture; it needs the
model weight present and only mocks nothing.

## Deploy (Modal)

```bash
modal secret create pronunciation-ai-secrets SERVICE_API_KEY=...
modal serve  modal_app.py     # live dev URL
modal deploy modal_app.py     # production (explicit; never automatic)
```

v1 runs single-writer (`max_containers=1`, one active input) because the SQLite
database lives on a Modal Volume. Move to PostgreSQL before scaling to multiple
containers — see the report.

## Layout

```
api/                 FastAPI layer (config, security, envelopes, errors,
                     schemas, analysis pipeline, routers, bootstrap)
db.py                subject-based SQLite persistence + idempotency ledger
*.py (root)          authoritative domain modules, copied verbatim from the
                     Flask app (scoring, mastery, assessment, g2p_service, ...)
g2p_pipeline_split_v2/, data/, voice-filtering/, model/   bundled assets
scripts/build_exercise_bank.py    offline bank builder
modal_app.py         Modal ASGI deployment
tests/               ported domain tests + FastAPI contract/idempotency tests
```
