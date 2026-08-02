"""Frontend-oriented pronunciation errors with reference-text spans.

The scoring domain intentionally works in IPA.  This module sits after the
public ARPAbet conversion and projects non-correct phoneme alignment rows back
onto the spelling that the learner was asked to read.

English spelling is not phonemic, so the mapper is deliberately conservative:
recognized grapheme/phoneme rules get a narrow ``grapheme`` span, while an
unrecognized phone falls back to its containing word.  That is preferable to
giving the UI a precise-looking but misleading letter range.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_WORD_RE = re.compile(r"[A-Za-z']+")
_MAX_GRAPHEME_LENGTH = 5
_MAX_ALIGNMENT_CELLS = 4096


# A grapheme may represent no phone (silent spelling), one phone, or several
# phones (for example ``x`` -> K S).  ARPAbet is stress-free at the API
# boundary, so the rule inventory is small and deterministic.
_GRAPHEME_RULES: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "'": ((),),
    "a": (("AE",), ("EY",), ("AH",), ("AA",), ("AO",)),
    "b": (("B",), ()),
    "c": (("K",), ("S",)),
    "d": (("D",),),
    "e": (("EH",), ("IY",), ("IH",), ()),
    "f": (("F",),),
    "g": (("G",), ("JH",), ()),
    "h": (("HH",), ()),
    "i": (("IH",), ("AY",), ("IY",)),
    "j": (("JH",),),
    "k": (("K",), ()),
    "l": (("L",), ()),
    "m": (("M",),),
    "n": (("N",), ("NG",)),
    "o": (("AA",), ("AO",), ("OW",), ("AH",), ("UW",)),
    "p": (("P",), ()),
    "q": (("K",),),
    "r": (("R",), ()),
    "s": (("S",), ("Z",), ("ZH",)),
    "t": (("T",), ()),
    "u": (("AH",), ("UW",), ("UH",), ("Y", "UW")),
    "v": (("V",),),
    "w": (("W",), ()),
    "x": (("K", "S"), ("G", "Z"), ("Z",)),
    "y": (("Y",), ("IY",), ("AY",), ("IH",)),
    "z": (("Z",),),
    # Doubled consonants.
    "bb": (("B",),),
    "dd": (("D",),),
    "ff": (("F",),),
    "gg": (("G",), ("JH",)),
    "ll": (("L",),),
    "mm": (("M",),),
    "nn": (("N",),),
    "pp": (("P",),),
    "rr": (("R",),),
    "ss": (("S",), ("Z",)),
    "tt": (("T",),),
    "zz": (("Z",),),
    # Consonant digraphs/trigraphs.
    "th": (("TH",), ("DH",)),
    "sh": (("SH",),),
    "ch": (("CH",), ("K",), ("SH",)),
    "tch": (("CH",),),
    "ph": (("F",),),
    "ng": (("NG",),),
    "ck": (("K",),),
    "wh": (("W",), ("HH",)),
    "wr": (("R",),),
    "gn": (("N",),),
    "mb": (("M",),),
    "rh": (("R",),),
    "ps": (("S",),),
    "dg": (("JH",),),
    "dge": (("JH",),),
    "qu": (("K", "W"), ("K",)),
    "gu": (("G",),),
    "gh": ((), ("F",)),
    # Vowel teams and r-coloured vowels.
    "ai": (("EY",), ("EH",)),
    "ay": (("EY",),),
    "ee": (("IY",),),
    "ea": (("IY",), ("EH",)),
    "ie": (("IY",), ("AY",)),
    "ei": (("IY",), ("EY",)),
    "ey": (("IY",), ("EY",)),
    "oa": (("OW",),),
    "oe": (("OW",),),
    "ow": (("OW",), ("AW",)),
    "oo": (("UW",), ("UH",)),
    "ou": (("AW",), ("AH",), ("UW",), ("UH",)),
    "oy": (("OY",),),
    "oi": (("OY",),),
    "au": (("AO",),),
    "aw": (("AO",),),
    "er": (("ER",),),
    "ir": (("ER",),),
    "ur": (("ER",),),
    "ew": (("Y", "UW"), ("UW",)),
    "ue": (("UW",),),
    "ui": (("UW",),),
    "igh": (("AY",),),
    "eigh": (("EY",),),
    "augh": (("AO",), ("AE",)),
    "ough": (("AO",), ("AW",), ("OW",), ("UW",), ("AH",)),
    # Productive English endings.
    "tion": (("SH", "AH", "N"),),
    "sion": (("ZH", "AH", "N"), ("SH", "AH", "N")),
    "cian": (("SH", "AH", "N"),),
    "ture": (("CH", "ER"),),
    "ure": (("ER",),),
}


@dataclass(frozen=True)
class _TextWord:
    text: str
    normalized: str
    start: int
    end: int
    word_index: int


@dataclass(frozen=True)
class _Step:
    grapheme_start: int
    grapheme_end: int
    phone_start: int
    phone_end: int
    known: bool


@dataclass(frozen=True)
class _PhoneSpan:
    start: int
    end: int
    word_index: Optional[int]
    kind: str


def _text_words(text: str) -> List[_TextWord]:
    words: List[_TextWord] = []
    for match in _WORD_RE.finditer(text):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip("'"))
        trailing = len(raw) - len(raw.rstrip("'"))
        normalized = raw.strip("'").lower()
        if not normalized:
            continue
        start = match.start() + leading
        end = match.end() - trailing
        words.append(_TextWord(text[start:end], normalized, start, end, len(words) + 1))
    return words


def _reference_words(reference_arpabet: str) -> List[List[str]]:
    words = []
    for chunk in str(reference_arpabet or "").split("|"):
        phones = [phone.strip().upper() for phone in chunk.split() if phone.strip()]
        if phones:
            words.append(phones)
    return words


def _phone_mapping_signature(path: Sequence[_Step], phone_count: int) -> Tuple[Any, ...]:
    signature: List[Any] = [None] * phone_count
    for step in path:
        if not step.known or step.phone_start == step.phone_end:
            continue
        for phone_index in range(step.phone_start, step.phone_end):
            signature[phone_index] = (step.grapheme_start, step.grapheme_end)
    return tuple(signature)


def _best_grapheme_path(word: str, phones: Sequence[str]) -> Tuple[List[_Step], bool]:
    """Return a deterministic many-to-many spelling/pronunciation alignment.

    The score first minimizes phones covered by fallback transitions, then
    fallback transitions/letters, then total steps.  Fewer steps makes a known
    digraph such as ``ph`` win over an incidental single-letter path.
    """
    letter_count, phone_count = len(word), len(phones)
    # state -> (score tuple, path, ambiguous phone-to-span mapping).  Longer
    # known graphemes are considered first, while equal-scoring paths with
    # different phone spans make the entire word use the conservative fallback.
    states: Dict[Tuple[int, int], Tuple[Tuple[int, int, int, int], List[_Step], bool]] = {
        (0, 0): ((0, 0, 0, 0), [], False)
    }

    def update(
        target: Tuple[int, int],
        score: Tuple[int, int, int, int],
        path: List[_Step],
        ambiguous: bool,
    ) -> None:
        current = states.get(target)
        if current is None or score < current[0]:
            states[target] = (score, path, ambiguous)
        elif score == current[0]:
            same_mapping = _phone_mapping_signature(current[1], phone_count) == _phone_mapping_signature(
                path, phone_count
            )
            states[target] = (current[0], current[1], current[2] or ambiguous or not same_mapping)

    for letter_index in range(letter_count + 1):
        for phone_index in range(phone_count + 1):
            current = states.get((letter_index, phone_index))
            if current is None:
                continue
            score, path, ambiguous = current

            max_letters = min(_MAX_GRAPHEME_LENGTH, letter_count - letter_index)
            for grapheme_length in range(max_letters, 0, -1):
                grapheme = word[letter_index:letter_index + grapheme_length]
                for pronunciation in _GRAPHEME_RULES.get(grapheme, ()):
                    next_phone = phone_index + len(pronunciation)
                    if next_phone > phone_count:
                        continue
                    if tuple(phones[phone_index:next_phone]) != pronunciation:
                        continue
                    step = _Step(
                        letter_index,
                        letter_index + grapheme_length,
                        phone_index,
                        next_phone,
                        True,
                    )
                    next_score = (score[0], score[1], score[2], score[3] + 1)
                    update(
                        (letter_index + grapheme_length, next_phone),
                        next_score,
                        path + [step],
                        ambiguous,
                    )

            # Fallback transitions guarantee a complete path while marking any
            # affected phone as unsafe for a narrow range.
            if letter_index < letter_count and phone_index < phone_count:
                step = _Step(letter_index, letter_index + 1, phone_index, phone_index + 1, False)
                next_score = (score[0] + 1, score[1] + 1, score[2] + 1, score[3] + 1)
                update((letter_index + 1, phone_index + 1), next_score, path + [step], ambiguous)
            if letter_index < letter_count:
                step = _Step(letter_index, letter_index + 1, phone_index, phone_index, False)
                next_score = (score[0], score[1] + 1, score[2] + 1, score[3] + 1)
                update((letter_index + 1, phone_index), next_score, path + [step], ambiguous)
            if phone_index < phone_count:
                step = _Step(letter_index, letter_index, phone_index, phone_index + 1, False)
                next_score = (score[0] + 1, score[1] + 1, score[2], score[3] + 1)
                update((letter_index, phone_index + 1), next_score, path + [step], ambiguous)

    final = states[(letter_count, phone_count)]
    return final[1], final[2]


def _map_word_phones(word: _TextWord, phones: Sequence[str]) -> List[_PhoneSpan]:
    fallback = _PhoneSpan(word.start, word.end, word.word_index, "word_fallback")
    if len(word.normalized) * max(1, len(phones)) > _MAX_ALIGNMENT_CELLS:
        return [fallback for _ in phones]

    path, ambiguous = _best_grapheme_path(word.normalized, phones)
    if ambiguous:
        return [fallback for _ in phones]

    mapped: List[Optional[_PhoneSpan]] = [None] * len(phones)
    for step in path:
        if not step.known or step.phone_start == step.phone_end:
            continue
        start = word.start + step.grapheme_start
        end = word.start + step.grapheme_end
        if start == end:
            continue
        for phone_index in range(step.phone_start, step.phone_end):
            mapped[phone_index] = _PhoneSpan(start, end, word.word_index, "grapheme")

    return [span if span is not None else fallback for span in mapped]


def _all_phone_spans(
    text: str,
    reference_arpabet: str,
    fallback_words: Iterable[str] = (),
) -> Tuple[List[_PhoneSpan], List[_TextWord]]:
    text_words = _text_words(text)
    reference_words = _reference_words(reference_arpabet)
    phone_spans: List[_PhoneSpan] = []
    whole_text_fallback = _PhoneSpan(0, len(text), None, "word_fallback")
    forced_fallbacks = {str(word).strip("'").lower() for word in fallback_words}

    for word_offset, phones in enumerate(reference_words):
        if word_offset < len(text_words):
            text_word = text_words[word_offset]
            if text_word.normalized in forced_fallbacks:
                fallback = _PhoneSpan(text_word.start, text_word.end, text_word.word_index, "word_fallback")
                phone_spans.extend(fallback for _ in phones)
            else:
                phone_spans.extend(_map_word_phones(text_word, phones))
        else:
            phone_spans.extend(whole_text_fallback for _ in phones)
    return phone_spans, text_words


def _utf16_offset(text: str, codepoint_offset: int) -> int:
    """Convert a Python code-point offset to a browser UTF-16 code-unit offset."""
    return len(text[:codepoint_offset].encode("utf-16-le")) // 2


def _public_span(text: str, span: _PhoneSpan) -> Dict[str, Any]:
    start = _utf16_offset(text, span.start)
    exclusive_end = _utf16_offset(text, span.end)
    return {
        "start": start,
        # Letter/word ranges are inclusive. Boundary markers are the one
        # intentional exception: start == end identifies a position while the
        # empty text and ``boundary`` kind tell clients not to highlight it.
        "end": start if span.kind == "boundary" else max(start, exclusive_end - 1),
        "text": text[span.start:span.end],
        "kind": span.kind,
    }


def _operation_for(row: Mapping[str, Any]) -> Optional[str]:
    result = str(row.get("result") or "")
    if result == "correct":
        return None
    if result == "insertion" or row.get("expected") in (None, "-"):
        return "insertion"
    if result == "deletion" or row.get("spoken") in (None, "-"):
        return "deletion"
    if result.endswith("_substitution"):
        return "substitution"
    return "substitution"


def _public_phone(value: Any) -> Optional[str]:
    if value in (None, "-"):
        return None
    return str(value)


def build_pronunciation_errors(
    text: str,
    reference_arpabet: str,
    alignment: Sequence[Mapping[str, Any]],
    *,
    fallback_words: Iterable[str] = (),
) -> List[Dict[str, Any]]:
    """Project every non-correct alignment row onto the reference spelling."""
    text = str(text or "")
    phone_spans, text_words = _all_phone_spans(text, reference_arpabet, fallback_words)
    errors: List[Dict[str, Any]] = []
    reference_cursor = 0

    for alignment_index, row in enumerate(alignment or []):
        operation = _operation_for(row)
        has_expected = row.get("expected") not in (None, "-")

        if operation is None:
            if has_expected:
                reference_cursor += 1
            continue

        if operation == "insertion":
            if reference_cursor < len(phone_spans):
                neighbor = phone_spans[reference_cursor]
                boundary = _PhoneSpan(neighbor.start, neighbor.start, neighbor.word_index, "boundary")
            elif phone_spans:
                neighbor = phone_spans[-1]
                boundary = _PhoneSpan(neighbor.end, neighbor.end, neighbor.word_index, "boundary")
            elif text_words:
                neighbor_word = text_words[0]
                boundary = _PhoneSpan(
                    neighbor_word.start,
                    neighbor_word.start,
                    neighbor_word.word_index,
                    "boundary",
                )
            else:
                boundary = _PhoneSpan(0, 0, None, "boundary")
            span = boundary
        elif reference_cursor < len(phone_spans):
            span = phone_spans[reference_cursor]
        else:
            # A malformed/mismatched reference should still honor the response
            # contract for deletions/substitutions with a non-empty fallback.
            span = _PhoneSpan(0, len(text), None, "word_fallback")

        errors.append({
            "alignment_index": alignment_index,
            "operation": operation,
            "result": str(row.get("result") or operation),
            "expected": _public_phone(row.get("expected")),
            "spoken": _public_phone(row.get("spoken")),
            "word_index": span.word_index,
            "reference_span": _public_span(text, span),
        })

        if has_expected:
            reference_cursor += 1

    return errors


def with_pronunciation_errors(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a payload containing the additive feedback field.

    The presence check keeps new idempotent responses byte-for-byte stable.
    Older stored responses can be enriched from fields they already contain.
    """
    if "pronunciation_errors" in payload:
        return dict(payload)
    enriched = dict(payload)
    enriched["pronunciation_errors"] = build_pronunciation_errors(
        text=str(payload.get("text") or ""),
        reference_arpabet=str(payload.get("reference_arpabet") or ""),
        alignment=payload.get("alignment") or [],
        fallback_words=payload.get("oov_words") or [],
    )
    return enriched
