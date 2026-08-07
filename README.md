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

The existing Windows Conda environment is the supported local runtime:

```powershell
conda activate nemo_g2p
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8002
```

Confirm that the running process selected NeMo at
`GET http://127.0.0.1:8002/health/ready`; `checks.g2p_mode` must be
`context_aware_nemo_ipa_g2p`.

To build another environment from scratch:

```bash
cd pronunciation_ai_service
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-nemo.txt
python -m spacy download en_core_web_sm

cp .env.example .env
#   set SERVICE_API_KEY to a long random secret

# Supply the trained model weight (distributed separately, gitignored):
#   model/my_wav2vec2_phoneme_model/model.safetensors

uvicorn api.main:app --reload
```

On Windows, use `py -3.11 -m venv .venv` and
`.\.venv\Scripts\Activate.ps1`. Python 3.11 is intentional: DeepFilterLib
publishes ready-made Windows/Linux wheels through CPython 3.11; newer Python
versions otherwise require a Rust/Cargo build toolchain.

NeMo is required for normal local and deployed execution. Startup fails if it
cannot be imported, preventing an unnoticed change in reference-G2P behavior.
`G2P_REQUIRE_NEMO=0` explicitly enables the dictionary-only fallback for the
deterministic test suite or deliberate lightweight development.

- Interactive docs: <http://127.0.0.1:8000/docs>
- OpenAPI: <http://127.0.0.1:8000/openapi.json>
- Liveness: `GET /health/live` · Readiness: `GET /health/ready`

The exercise bank is seeded automatically on first startup from
`data/seed_sentences.txt`. The database starts empty of subjects.

## Auth

Every `/api/v1` route requires `Authorization: Bearer <SERVICE_API_KEY>`.
Stateful writes (`/subjects/{id}/analyses`, `/subjects/{id}/exercises/next`)
also require an `Idempotency-Key` header so retries never duplicate evidence.

## Clean audio endpoint

`POST /api/v1/audio/clean` accepts a multipart `audio` upload and streams a
pronunciation-safe cleaned WAV back to the caller. The primary path is local
and it is not connected to LiveKit or realtime conversations. An optional
Cleanvoice fallback can be enabled for internal model/processing failures.

The pipeline is:

```text
immutable upload
  -> FFmpeg mono/48 kHz/PCM-16 source normalization
  -> DeepFilterNet background-noise reduction
  -> cleaned mono/48 kHz/PCM-16 WAV (playback/download)
  -> FFmpeg mono/16 kHz/PCM-16 WAV (STT + alignment)
  -> Silero VAD + soundfile/NumPy quality metadata

eligible local processing failure + configured CLEANVOICE_API_KEY
  -> Cleanvoice noise reduction + normalization (no speech/timing cuts)
  -> exact 48 kHz + 16 kHz WAV contract and local speech/quality metadata
```

The original is never overwritten or replaced by a cleaned file. The analysis
routes use the pre-denoise signal for quality/scoring evidence and the cleaned
16 kHz file for STT/alignment. By default all service-side copies are private
temporary files deleted after the request; the calling core remains the owner
of its persisted original. Set `AUDIO_CLEANING_KEEP_INTERMEDIATE_FILES=1` only
for local debugging if local retention is required.

Cleanvoice is never called after validation/decode errors such as an empty,
corrupt, or unsupported upload. When configured, the normalized recording is
uploaded only after an eligible local DeepFilterNet/Silero processing failure.
If both paths fail, the API returns the safe
`audio_cleaning_all_providers_failed` error without exposing provider URLs or
credentials.

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/clean?sample_rate=48000" \
  -H "Authorization: Bearer $SERVICE_API_KEY" \
  -F "audio=@recording.wav" \
  --output recording_cleaned.wav
```

The response headers report which path ran:

- `X-Audio-Processing-Pipeline`
- `X-Noise-Reduction-Applied`
- `X-Original-Preserved`
- `X-Audio-Processing-Status`
- `X-Audio-Cleaning-Backend` (`deepfilternet` or `cleanvoice`)
- `X-Audio-Fallback-Used`
- `X-Audio-Speech-Seconds`
- `X-Audio-Scoring-Allowed`
- `X-Audio-Rejection-Reasons`

Use `sample_rate=48000` (default) for playback or `sample_rate=16000` for the
STT/alignment representation. Both are generated during one processing run.

### Installation and configuration

Install FFmpeg as an external system dependency:

```powershell
# Windows (one option)
winget install Gyan.FFmpeg
ffmpeg -version
```

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg
ffmpeg -version
```

Then install the Python dependencies and enable the feature in `.env`:

```bash
pip install -r requirements.txt
```

```dotenv
AUDIO_CLEANING_ENABLED=1
AUDIO_CLEANING_USE_GPU=0
AUDIO_CLEANING_MIN_DURATION_SECONDS=0.5
AUDIO_CLEANING_MIN_SPEECH_SECONDS=0.3
AUDIO_CLEANING_MAX_CLIPPING_RATIO=0.01
AUDIO_CLEANING_MIN_SPEECH_RATIO=
AUDIO_CLEANING_KEEP_INTERMEDIATE_FILES=0
AUDIO_CLEANING_TIMEOUT_SECONDS=120
AUDIO_CLEANING_OUTPUT_DIR=./uploads
FFMPEG_BINARY=ffmpeg

# Optional: external fallback, automatically enabled when the key is present.
CLEANVOICE_API_KEY=your-cleanvoice-api-key
CLEANVOICE_FALLBACK_ENABLED=1
CLEANVOICE_HTTP_TIMEOUT=120
CLEANVOICE_STUDIO_SOUND=0
```

DeepFilterNet and Silero VAD load lazily on the first cleaning request and are
reused afterward. They are not loaded during startup, database initialization,
health checks, or unrelated requests. `AUDIO_CLEANING_USE_GPU=1` requests CUDA;
the service automatically uses CPU when CUDA is unavailable. Silero VAD stays
on CPU because it is lightweight and this avoids GPU contention.

Cleanvoice is optional and never replaces the local-first path. For
pronunciation safety, the fallback enables only noise removal and loudness
normalization; filler, silence, mouth-sound, breath, hesitation, and stutter
editing remain disabled. `CLEANVOICE_ENABLED` is accepted as a legacy alias,
but `CLEANVOICE_FALLBACK_ENABLED` is the preferred setting.

Validation defaults reject durations below 0.5 seconds, detected speech below
0.3 seconds, and clipping ratios above 0.01. A low speech ratio is recorded but
is not rejected unless `AUDIO_CLEANING_MIN_SPEECH_RATIO` is explicitly set.
Rejection reasons are stable codes such as `recording_too_short`,
`no_speech_detected`, `insufficient_speech`, and `severe_clipping`.

Manual processing is available through a FastAPI-project CLI (this repository
is not Django, so no Django management command is introduced):

```bash
python -m scripts.clean_exercise_audio recording.wav --output-dir output
python -m scripts.clean_exercise_audio recording.wav --output-dir output --force
```

The command prints JSON containing both output locations, duration, speech
duration/ratio, clipping ratio, the scoring decision, and rejection reasons.
Completed runs are idempotently reused unless `--force` is supplied.

DeepFilterNet is intended for fan noise, traffic, hum, and general background
noise. It cannot reliably identify and remove a second human speaker whose
speech overlaps the learner. This first version intentionally does not perform
speaker separation or diarization and does not use pitch/formant correction,
UVR, SepFormer, or chained denoisers. Cleanvoice Studio Sound is off by default.

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
RUN_AUDIO_MODEL_TESTS=1 python -m pytest tests/test_audio_cleaning_service.py
```

The suite covers the ported domain logic (scoring, mastery, assessment,
tokenization, G2P, content, audio-quality gate) plus FastAPI contract tests
(auth, envelopes, request ids, media types, OpenAPI), idempotent retries,
subject isolation/deletion, the mastery trust gate, exercise assignment, history
pagination, temporary-audio cleanup on every failure stage, FFmpeg failures,
DeepFilterNet/Silero orchestration, Cleanvoice fallback eligibility and safe
options, VAD metadata, JSON-safe scalar conversion, CPU fallback, idempotency,
original preservation, and lazy model loading.

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
g2p_pipeline_split_v2/, data/, model/          bundled assets
src/core/audio/audio_cleaning.py               DeepFilterNet/Silero pipeline
src/core/audio/cleanvoice_service.py            optional external fallback
scripts/build_exercise_bank.py    offline bank builder
scripts/clean_exercise_audio.py   manual audio-cleaning CLI
modal_app.py         Modal ASGI deployment
tests/               ported domain tests + FastAPI contract/idempotency tests
```
