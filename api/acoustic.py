"""
Wav2Vec2 acoustic phoneme transcription, isolated behind a tiny surface so the
CI test suite can mock it without importing torch/transformers.

This is the only place the heavy model stack is touched. ``transcribe()`` takes
a prepared 16 kHz mono WAV path and returns normalized IPA. Loading is lazy and
cached process-wide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

SERVICE_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = SERVICE_ROOT / "model" / "my_wav2vec2_phoneme_model"

_processor: Optional[Any] = None
_model: Optional[Any] = None
_device: Optional[str] = None


def model_config_present() -> bool:
    return (MODEL_PATH / "config.json").exists()


def model_weight_present() -> bool:
    return (MODEL_PATH / "model.safetensors").exists() or (MODEL_PATH / "pytorch_model.bin").exists()


def model_loaded() -> bool:
    return _model is not None and _processor is not None


def device() -> str:
    global _device
    if _device is None:
        try:
            import torch

            _device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            _device = "cpu"
    return _device


def load_model() -> None:
    """Load the trained Wav2Vec2 phoneme model from the local model folder.
    Raises ``ModelUnavailableError`` (503) if the config or the weight has not
    been supplied, so a deployment missing the separately-distributed model
    fails as "not ready" rather than as an opaque 500."""
    global _processor, _model
    if model_loaded():
        return
    if not model_config_present() or not model_weight_present():
        from .errors import ModelUnavailableError

        raise ModelUnavailableError(
            f"The acoustic model is not available in: {MODEL_PATH}. "
            "Supply the trained model config and weight there "
            "(the weight is distributed separately)."
        )
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    _processor = Wav2Vec2Processor.from_pretrained(str(MODEL_PATH))
    _model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_PATH))
    _model.to(device())
    _model.eval()


def transcribe(model_input_path: Path) -> str:
    """Transcribe a prepared 16 kHz mono WAV into normalized IPA."""
    import librosa
    import torch

    from src.core.g2p.tokenization import normalize_ipa

    load_model()
    audio, _ = librosa.load(str(model_input_path), sr=16000, mono=True)
    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = _model(inputs.input_values.to(device())).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return normalize_ipa(_processor.batch_decode(predicted_ids)[0])


def probe_loadable() -> Any:
    """Deep readiness check: actually load the processor + model. Returns True or
    an error string (never raises)."""
    try:
        from transformers import Wav2Vec2Processor  # noqa: F401

        load_model()
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"error: {exc!r}"
