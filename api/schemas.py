"""
Pydantic request/response models.

Requests are validated strictly (bad input is a 422 before any work happens).
Response *data* models describe the envelope's ``data`` payload for the
generated OpenAPI; the richer analysis/assessment payloads allow extra fields
so the authoritative domain output is never silently truncated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---- meta / envelope -------------------------------------------------------
class Meta(BaseModel):
    service: str
    api_version: str
    request_id: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ErrorDetail
    meta: Meta


def _envelope(data_model):
    """Build a concrete ``{data, meta}`` envelope model for a data payload."""
    ns = {
        "__annotations__": {"data": data_model, "meta": Meta},
        "model_config": ConfigDict(extra="allow"),
    }
    return type(f"{getattr(data_model, '__name__', 'Data')}Envelope", (BaseModel,), ns)


# ---- G2P -------------------------------------------------------------------
class G2PRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="English text to convert to ARPAbet.")


class G2PData(BaseModel):
    model_config = ConfigDict(extra="allow")
    text: str
    arpabet: str
    guide: List[Dict[str, Any]]
    g2p_mode: str
    heteronym_resolution_active: bool
    reference_g2p_trusted: bool
    unresolved_heteronyms: List[str]
    unsupported_heteronyms: List[str]
    oov_words: List[str]


# ---- subjects --------------------------------------------------------------
class SubjectData(BaseModel):
    subject_id: str
    created_at: str
    created: bool = Field(..., description="True if this call created the profile, False if it already existed.")


class DeletedSubjectData(BaseModel):
    subject_id: str
    deleted: bool


# ---- analysis --------------------------------------------------------------
class ReferenceSpan(BaseModel):
    start: int = Field(..., ge=0, description="Inclusive UTF-16 code-unit offset into AnalysisData.text.")
    end: int = Field(..., ge=0, description="Inclusive UTF-16 code-unit offset into AnalysisData.text.")
    text: str
    kind: Literal["grapheme", "word_fallback", "boundary"]


class PronunciationError(BaseModel):
    alignment_index: int = Field(..., ge=0, description="Zero-based index into the alignment array.")
    operation: Literal["substitution", "deletion", "insertion"]
    result: str
    expected: Optional[str]
    spoken: Optional[str]
    word_index: Optional[int] = Field(None, ge=1, description="One-based reference word index when available.")
    reference_span: ReferenceSpan


class AnalysisData(BaseModel):
    """Superset of the stateless and stateful analysis payloads. Extra fields
    are allowed so the full domain output (guides, provenance, diagnostics)
    passes through untouched."""
    model_config = ConfigDict(extra="allow")
    scorable: bool
    quality_warning: bool
    text: str
    reference_arpabet: str
    predicted_arpabet: str
    scoring_engine: str
    scoring_trusted: bool
    metrics: Dict[str, Any]
    alignment: List[Dict[str, Any]]
    pronunciation_errors: List[PronunciationError] = Field(default_factory=list)
    audio_quality: Dict[str, Any]


# ---- assessment / gaps / history ------------------------------------------
class AssessmentData(BaseModel):
    model_config = ConfigDict(extra="allow")
    assessment: Optional[Dict[str, Any]]


class GapPhoneme(BaseModel):
    model_config = ConfigDict(extra="allow")
    phoneme: str
    mastery: float
    lower_confidence_bound: float
    independent_attempts: int
    occurrence_count: int
    last_practiced_at: Optional[str]
    example: str


class GapsData(BaseModel):
    phonemes: List[GapPhoneme]


class AttemptSummary(BaseModel):
    id: int
    text: str
    phoneme_error_rate: float
    scorable: bool
    mastery_updated: bool
    scoring_engine: Optional[str]
    audio_processing: Dict[str, Any] = Field(default_factory=dict)
    created_at: str


class AttemptsPageData(BaseModel):
    attempts: List[AttemptSummary]
    next_cursor: Optional[str] = Field(
        None, description="Pass as ?cursor= to fetch the next (older) page; null when exhausted."
    )
    has_more: bool


# ---- exercises -------------------------------------------------------------
class ExerciseGenerateRequest(BaseModel):
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-phoneme mastery scores in [0,1] keyed by ARPAbet symbol.",
    )


class ExerciseData(BaseModel):
    model_config = ConfigDict(extra="allow")
    assessment: Dict[str, Any]
    exercise: Optional[Dict[str, Any]]
    target_phonemes: List[str] = Field(default_factory=list)


class NextExerciseData(BaseModel):
    """The adaptive /exercises/next payload (a flat assignment, not the
    stateless {assessment, exercise} shape)."""
    model_config = ConfigDict(extra="allow")
    sentence_id: int
    text: str
    reference_arpabet: str
    mode: str
    exercise_type: str
    target_phonemes: List[str] = Field(default_factory=list)
    assessment: Dict[str, Any]


# Concrete envelope models for OpenAPI.
G2PEnvelope = _envelope(G2PData)
SubjectEnvelope = _envelope(SubjectData)
DeletedSubjectEnvelope = _envelope(DeletedSubjectData)
AnalysisEnvelope = _envelope(AnalysisData)
AssessmentEnvelope = _envelope(AssessmentData)
GapsEnvelope = _envelope(GapsData)
AttemptsPageEnvelope = _envelope(AttemptsPageData)
ExerciseEnvelope = _envelope(ExerciseData)
NextExerciseEnvelope = _envelope(NextExerciseData)
