"""
Wav2Vec2 acoustic phoneme transcription, isolated behind a small surface so the
CI suite can exercise the decoder without importing torch/transformers.

The deployed model was trained with an explicit, token-level ARPAbet label
inventory.  Its Hugging Face repository intentionally contains only the model
config and weights, so loading an ``AutoProcessor`` would be both incorrect and
impossible.  This module recreates the training feature extractor, performs CTC
collapse with the exact training label order, and converts the result to the
service's internal IPA representation.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)

SERVICE_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = SERVICE_ROOT / "model" / "my_wav2vec2_phoneme_model"
MODEL_ID = "waelhasan/wav2vec2-l2-arctic-phoneme-model"
MODEL_REVISION = "6aaebdaf68bfda0aaa4b9a03b0e7eb1531d58e30"

# This order is part of the trained checkpoint contract.  It comes from
# Wav2Vec2CTCStrategy.vocab in the training notebook; changing it without
# retraining silently changes every predicted phoneme.
ARPABET_VOCAB: Tuple[str, ...] = (
    "[PAD]",
    "[UNK]",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
)
CTC_BLANK_ID = 0
CTC_UNKNOWN_ID = 1

_feature_extractor: Optional[Any] = None
_model: Optional[Any] = None
_device: Optional[str] = None


def model_config_present() -> bool:
    return (MODEL_PATH / "config.json").exists()


def model_weight_present() -> bool:
    return (MODEL_PATH / "model.safetensors").exists() or (
        MODEL_PATH / "pytorch_model.bin"
    ).exists()


def model_loaded() -> bool:
    return _model is not None and _feature_extractor is not None


def device() -> str:
    global _device
    if _device is None:
        try:
            import torch

            _device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            _device = "cpu"
    return _device


def _validate_model_contract(model: Any) -> None:
    """Fail closed if the checkpoint and hard-coded label order diverge."""
    from .errors import ModelUnavailableError

    vocab_size = int(getattr(model.config, "vocab_size", -1))
    pad_token_id = int(getattr(model.config, "pad_token_id", -1))
    if vocab_size != len(ARPABET_VOCAB) or pad_token_id != CTC_BLANK_ID:
        raise ModelUnavailableError(
            "The acoustic checkpoint does not match its ARPAbet decoder: "
            f"vocab_size={vocab_size}, pad_token_id={pad_token_id}, "
            f"expected vocab_size={len(ARPABET_VOCAB)} and "
            f"pad_token_id={CTC_BLANK_ID}."
        )


def load_model() -> None:
    """Load the pinned ARPAbet Wav2Vec2 checkpoint from the local model folder."""
    global _feature_extractor, _model
    if model_loaded():
        return
    if not model_config_present() or not model_weight_present():
        from .errors import ModelUnavailableError

        raise ModelUnavailableError(
            f"The acoustic model is not available in: {MODEL_PATH}. Download "
            f"{MODEL_ID} at revision {MODEL_REVISION} into that directory."
        )

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(MODEL_PATH))
    model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_PATH))
    _validate_model_contract(model)
    model.to(device())
    model.eval()
    _feature_extractor = feature_extractor
    _model = model


def decode_ctc_ids_to_arpabet(predicted_ids: Sequence[int]) -> Tuple[str, ...]:
    """Collapse one CTC id sequence using the checkpoint's training labels."""
    tokens = []
    for raw_id, _ in itertools.groupby(int(value) for value in predicted_ids):
        if raw_id == CTC_BLANK_ID:
            continue
        if not 0 <= raw_id < len(ARPABET_VOCAB):
            raise ValueError(f"Acoustic model emitted an out-of-range token id: {raw_id}")
        if raw_id == CTC_UNKNOWN_ID:
            logger.warning("Dropping [UNK] emitted by the acoustic CTC model")
            continue
        tokens.append(ARPABET_VOCAB[raw_id])
    return tuple(tokens)


def arpabet_tokens_to_raw_ctc_ipa(tokens: Iterable[str]) -> str:
    """Convert predicted ARPAbet tokens to compact, internal raw-CTC IPA.

    The model was not trained with word-boundary labels.  Compact IPA is
    therefore intentional: whitespace in the service's raw-CTC format means a
    real acoustic word boundary and must not be inserted between phonemes.
    """
    from src.core.g2p.phoneme_alphabet import arpabet_phoneme_to_ipa
    from src.core.g2p.tokenization import normalize_ipa

    return normalize_ipa("".join(arpabet_phoneme_to_ipa(token) for token in tokens))


def _predict_arpabet_tokens(model_input_path: Path) -> Tuple[str, ...]:
    import librosa
    import torch

    load_model()
    audio, _ = librosa.load(str(model_input_path), sr=16000, mono=True)
    inputs = _feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        logits = _model(inputs.input_values.to(device())).logits
    predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()
    return decode_ctc_ids_to_arpabet(predicted_ids)


def transcribe_arpabet(model_input_path: Path) -> str:
    """Transcribe a prepared 16 kHz mono WAV into spaced, stress-free ARPAbet."""
    return " ".join(_predict_arpabet_tokens(model_input_path))


def transcribe(model_input_path: Path) -> str:
    """Transcribe a prepared 16 kHz mono WAV into compact internal IPA."""
    return arpabet_tokens_to_raw_ctc_ipa(_predict_arpabet_tokens(model_input_path))


def probe_loadable() -> Any:
    """Deep readiness check. Returns True or an error string and never raises."""
    try:
        from transformers import Wav2Vec2FeatureExtractor  # noqa: F401

        load_model()
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"error: {exc!r}"
