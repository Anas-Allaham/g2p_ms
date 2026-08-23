"""Conversion between the service's internal IPA and public ARPAbet.

IPA is canonical inside the G2P, acoustic, persistence, alignment, mastery,
and scoring layers. Django and frontend clients use ARPAbet, so the API
boundary converts through this module.

The service intentionally uses stress-free uppercase ARPAbet because the
acoustic model does not predict lexical stress.  ``AX`` remains distinct from
``AH`` in references and public guides.  The active 39-phone model emits
``AH`` for both, so scoring handles that checkpoint limitation explicitly.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from enum import Enum
from typing import List, Mapping


logger = logging.getLogger(__name__)


IPA_TO_ARPABET: Mapping[str, str] = {
    "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "ɡ": "G", "g": "G",
    "f": "F", "v": "V", "θ": "TH", "ð": "DH", "s": "S", "z": "Z",
    "ʃ": "SH", "ʒ": "ZH", "h": "HH", "tʃ": "CH", "t͡ʃ": "CH",
    "dʒ": "JH", "d͡ʒ": "JH", "m": "M", "n": "N", "ŋ": "NG", "l": "L",
    "ɫ": "L", "ɹ": "R", "r": "R", "w": "W", "j": "Y",
    "i": "IY", "iː": "IY", "ɪ": "IH", "ɛ": "EH", "æ": "AE",
    "ɑ": "AA", "ɑː": "AA", "ɔ": "AO", "ɔː": "AO", "ʊ": "UH",
    "u": "UW", "uː": "UW", "ʌ": "AH", "ə": "AX", "ɝ": "ER",
    "ɚ": "ER", "ɜː": "ER", "ə˞": "ER",
    "aɪ": "AY", "aʊ": "AW", "eɪ": "EY", "oʊ": "OW", "ɔɪ": "OY",
}

# These tokens belong to the acoustic/scoring inventory, but their ARPAbet
# equivalents are recovery approximations rather than exact general-IPA
# conversions. They are enabled only for acoustic/internal paths.
CTC_COMPONENT_FALLBACKS: Mapping[str, str] = {
    "a": "AA", "e": "EH", "o": "OW",
}
CTC_CONTROL_TOKENS = frozenset({
    "[PAD]", "[UNK]", "<pad>", "<unk>", "<s>", "</s>",
})
# Older NeMo data may contain sentence punctuation as a standalone reference
# chunk (for example ``| .``). It is formatting noise, not a phoneme.
REFERENCE_PUNCTUATION_TOKENS = frozenset({".", ",", "!", "?", ";", "…"})

ARPABET_TO_IPA: Mapping[str, str] = {
    "P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "ɡ",
    "F": "f", "V": "v", "TH": "θ", "DH": "ð", "S": "s", "Z": "z",
    "SH": "ʃ", "ZH": "ʒ", "HH": "h", "CH": "tʃ", "JH": "dʒ",
    "M": "m", "N": "n", "NG": "ŋ", "L": "l", "R": "ɹ", "W": "w",
    "Y": "j", "IY": "i", "IH": "ɪ", "EH": "ɛ", "AE": "æ",
    "AA": "ɑ", "AO": "ɔ", "UH": "ʊ", "UW": "u", "AH": "ʌ",
    "AX": "ə", "ER": "ɝ", "AY": "aɪ", "AW": "aʊ", "EY": "eɪ",
    "OW": "oʊ", "OY": "ɔɪ",
}

ARPABET_ALIASES: Mapping[str, str] = {
    "AXR": "ER",
    "UX": "UW",
}

ARPABET_VOWELS = frozenset({
    "IY", "IH", "EH", "AE", "AA", "AO", "UH", "UW", "AH", "AX", "ER",
    "AY", "AW", "EY", "OW", "OY",
})

ARPABET_CONSONANTS = frozenset({
    "P", "B", "T", "D", "K", "G", "F", "V", "TH", "DH", "S", "Z",
    "SH", "ZH", "HH", "CH", "JH", "M", "N", "NG", "L", "R", "W", "Y",
})

CANONICAL_ARPABET = ARPABET_VOWELS | ARPABET_CONSONANTS


class IPAInputFormat(str, Enum):
    """The two IPA shapes produced inside this service."""

    FORMATTED_REFERENCE = "formatted_reference"
    RAW_CTC = "raw_ctc"


class UnsupportedPhonemeError(ValueError):
    """Raised when a phoneme cannot be represented publicly as ARPAbet."""

    def __init__(self, phoneme: str) -> None:
        self.phoneme = phoneme
        super().__init__(f"Unsupported IPA/ARPAbet phoneme: {phoneme!r}")

_IPA_ALIASES: Mapping[str, str] = {
    "ɜ:": "ɜː", "ɑ:": "ɑː", "ɔ:": "ɔː", "i:": "iː", "u:": "uː",
}
_IPA_COMBINING_MARKS = {"ː", "̃", "̩", "̯", "ʰ", "˞"}
_IPA_SYMBOLS = tuple(sorted(
    set(IPA_TO_ARPABET) | set(CTC_COMPONENT_FALLBACKS),
    key=len,
    reverse=True,
))


def canonicalize_arpabet_phoneme(
    phoneme: str,
    *,
    strict: bool = True,
    allow_ctc_fallbacks: bool = False,
) -> str:
    """Return one validated, stress-free uppercase ARPAbet token.

    IPA input is also accepted so boundary adapters can normalize one symbol.
    """
    token = unicodedata.normalize("NFC", str(phoneme)).strip()
    token = token.replace("/", "").replace("[", "").replace("]", "")
    token = token.replace("ˈ", "").replace("ˌ", "")
    if not token or token == "|":
        if strict:
            raise UnsupportedPhonemeError(str(phoneme))
        return ""

    if token in IPA_TO_ARPABET:
        return IPA_TO_ARPABET[token]
    token = _IPA_ALIASES.get(token, token)
    if token in IPA_TO_ARPABET:
        return IPA_TO_ARPABET[token]

    # Match the scoring canonicalizer: apply specific aliases first, then
    # remove a residual length mark. This covers NeMo forms such as /ɝː/.
    if "ː" in token and len(token) > 1:
        without_length = token.replace("ː", "")
        if without_length in IPA_TO_ARPABET:
            return IPA_TO_ARPABET[without_length]

    if allow_ctc_fallbacks and token in CTC_COMPONENT_FALLBACKS:
        return CTC_COMPONENT_FALLBACKS[token]

    upper = re.sub(r"[012]$", "", token.upper())
    upper = ARPABET_ALIASES.get(upper, upper)
    if upper in CANONICAL_ARPABET:
        return upper

    if strict:
        raise UnsupportedPhonemeError(str(phoneme))

    logger.warning(
        "Dropping unsupported phoneme during non-strict CTC conversion: %r",
        phoneme,
    )
    return ""


def split_ipa_word(ipa_word: str) -> List[str]:
    """Split a compact IPA word into the model's phoneme units."""
    word = unicodedata.normalize("NFC", str(ipa_word))
    word = word.replace("ˈ", "").replace("ˌ", "").replace("_", "")
    word = word.replace("/", "").replace("[", "").replace("]", "")
    for source, target in _IPA_ALIASES.items():
        word = word.replace(source, target)

    tokens: List[str] = []
    index = 0
    while index < len(word):
        if word[index].isspace() or word[index] == "|":
            index += 1
            continue
        match = next((symbol for symbol in _IPA_SYMBOLS if word.startswith(symbol, index)), None)
        if match is not None:
            tokens.append(match)
            index += len(match)
            continue
        char = word[index]
        if char in _IPA_COMBINING_MARKS and tokens:
            tokens[-1] += char
        else:
            tokens.append(char)
        index += 1
    return tokens


def ipa_word_to_arpabet_tokens(
    ipa_word: str,
    *,
    strict: bool = True,
    allow_ctc_fallbacks: bool = False,
) -> List[str]:
    """Convert one IPA word to canonical stress-free ARPAbet tokens."""
    raw_word = unicodedata.normalize("NFC", str(ipa_word)).strip()
    if raw_word in CTC_CONTROL_TOKENS:
        if strict:
            raise UnsupportedPhonemeError(raw_word)
        logger.warning(
            "Dropping tokenizer control token during non-strict CTC conversion: %r",
            raw_word,
        )
        return []

    converted = []
    for token in split_ipa_word(ipa_word):
        arpabet = canonicalize_arpabet_phoneme(
            token,
            strict=strict,
            allow_ctc_fallbacks=allow_ctc_fallbacks,
        )
        if arpabet:
            converted.append(arpabet)
    return converted


def ipa_to_arpabet(
    ipa: str,
    *,
    input_format: IPAInputFormat,
    strict: bool = True,
) -> str:
    """Convert one explicitly identified IPA shape to spaced ARPAbet.

    Formatted references use ``|`` between words and spaces between phonemes.
    Raw CTC output uses whitespace as word boundaries and compact phonemes
    inside each word. Explicit ``|`` boundaries are accepted in either mode.
    """
    text = unicodedata.normalize("NFC", str(ipa)).strip()
    if not text:
        return ""

    if input_format is IPAInputFormat.RAW_CTC:
        for control_token in CTC_CONTROL_TOKENS:
            if control_token not in text:
                continue
            if strict:
                raise UnsupportedPhonemeError(control_token)
            logger.warning(
                "Dropping tokenizer control token during non-strict CTC conversion: %r",
                control_token,
            )
            text = text.replace(control_token, "")
        text = text.strip()
        if not text:
            return ""

    if "|" in text:
        word_chunks = [chunk.strip() for chunk in text.split("|")]
    elif input_format is IPAInputFormat.FORMATTED_REFERENCE:
        word_chunks = [text]
    elif input_format is IPAInputFormat.RAW_CTC:
        word_chunks = text.split()
    else:
        raise ValueError(f"Unsupported IPA input format: {input_format!r}")

    words: List[str] = []
    allow_ctc_fallbacks = input_format is IPAInputFormat.RAW_CTC
    for chunk in word_chunks:
        tokens: List[str] = []
        for part in chunk.split():
            if (
                input_format is IPAInputFormat.FORMATTED_REFERENCE
                and part in REFERENCE_PUNCTUATION_TOKENS
            ):
                continue
            tokens.extend(ipa_word_to_arpabet_tokens(
                part,
                strict=strict,
                allow_ctc_fallbacks=allow_ctc_fallbacks,
            ))
        if tokens:
            words.append(" ".join(tokens))
    return " | ".join(words)


def arpabet_phoneme_to_ipa(phoneme: str) -> str:
    """Convert one canonical ARPAbet token to the IPA used by PanPhon."""
    token = canonicalize_arpabet_phoneme(phoneme)
    try:
        return ARPABET_TO_IPA[token]
    except KeyError as exc:
        raise ValueError(f"Unsupported ARPAbet phoneme: {phoneme!r}") from exc


def arpabet_to_ipa(arpabet: str) -> str:
    """Convert a spaced ARPAbet sequence to formatted IPA."""
    words: List[str] = []
    for word in str(arpabet).replace("|", " | ").split("|"):
        segments = [arpabet_phoneme_to_ipa(token) for token in word.split()]
        if segments:
            words.append("".join(segments))
    return " | ".join(words)
