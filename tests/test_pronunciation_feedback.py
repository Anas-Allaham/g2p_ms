"""Projection of phoneme errors onto frontend-safe reference-text ranges."""

from api.pronunciation_feedback import build_pronunciation_errors


def _rows_with_errors(phones, error_indexes):
    return [
        {
            "expected": phone,
            "spoken": "T" if index in error_indexes else phone,
            "result": "major_substitution" if index in error_indexes else "correct",
        }
        for index, phone in enumerate(phones)
    ]


def _utf16_inclusive_slice(text, start, end):
    encoded = text.encode("utf-16-le")
    return encoded[start * 2:(end + 1) * 2].decode("utf-16-le")


def test_common_english_graphemes_get_narrow_ranges():
    cases = [
        ("school", ["S", "K", "UW", "L"], 1, "ch"),
        ("think", ["TH", "IH", "NG", "K"], 0, "th"),
        ("phone", ["F", "OW", "N"], 0, "ph"),
        ("knight", ["N", "AY", "T"], 1, "igh"),
        ("boat", ["B", "OW", "T"], 1, "oa"),
    ]
    for text, phones, error_index, expected_letters in cases:
        errors = build_pronunciation_errors(
            text,
            " ".join(phones),
            _rows_with_errors(phones, {error_index}),
        )
        assert len(errors) == 1
        assert errors[0]["reference_span"]["text"] == expected_letters
        assert errors[0]["reference_span"]["kind"] == "grapheme"


def test_one_grapheme_can_be_shared_by_multiple_phone_errors():
    phones = ["B", "AA", "K", "S"]
    errors = build_pronunciation_errors("box", " ".join(phones), _rows_with_errors(phones, {2, 3}))
    assert [error["reference_span"]["text"] for error in errors] == ["x", "x"]
    assert errors[0]["reference_span"] == errors[1]["reference_span"]


def test_unknown_phone_uses_whole_word_fallback():
    errors = build_pronunciation_errors(
        "xyzzy",
        "TH",
        [{"expected": "TH", "spoken": "S", "result": "unknown_substitution"}],
    )
    assert errors[0]["reference_span"] == {
        "start": 0,
        "end": 4,
        "text": "xyzzy",
        "kind": "word_fallback",
    }


def test_g2p_reported_oov_word_forces_word_fallback():
    errors = build_pronunciation_errors(
        "school",
        "S K UW L",
        _rows_with_errors(["S", "K", "UW", "L"], {1}),
        fallback_words=["school"],
    )
    assert errors[0]["reference_span"] == {
        "start": 0,
        "end": 5,
        "text": "school",
        "kind": "word_fallback",
    }


def test_substitution_deletion_and_insertion_contract():
    rows = [
        {"expected": "S", "spoken": "S", "result": "correct"},
        {"expected": "K", "spoken": "T", "result": "major_substitution"},
        {"expected": "UW", "spoken": "UW", "result": "correct"},
        {"expected": "L", "spoken": "-", "result": "deletion"},
        {"expected": "-", "spoken": "AH", "result": "insertion"},
    ]
    errors = build_pronunciation_errors("school", "S K UW L", rows)

    assert [error["operation"] for error in errors] == ["substitution", "deletion", "insertion"]
    assert [error["alignment_index"] for error in errors] == [1, 3, 4]
    assert errors[0]["reference_span"]["text"] == "ch"
    assert errors[1]["spoken"] is None
    assert errors[1]["reference_span"]["text"] == "l"
    assert errors[2]["expected"] is None
    assert errors[2]["reference_span"] == {
        "start": 6,
        "end": 6,
        "text": "",
        "kind": "boundary",
    }


def test_insertions_use_initial_middle_and_final_boundaries():
    rows = [
        {"expected": "-", "spoken": "AH", "result": "insertion"},
        {"expected": "S", "spoken": "S", "result": "correct"},
        {"expected": "-", "spoken": "AH", "result": "insertion"},
        {"expected": "K", "spoken": "K", "result": "correct"},
        {"expected": "UW", "spoken": "UW", "result": "correct"},
        {"expected": "L", "spoken": "L", "result": "correct"},
        {"expected": "-", "spoken": "AH", "result": "insertion"},
    ]
    errors = build_pronunciation_errors("school", "S K UW L", rows)
    assert [(e["reference_span"]["start"], e["reference_span"]["end"]) for e in errors] == [
        (0, 0),
        (1, 1),
        (6, 6),
    ]


def test_repeated_words_and_punctuation_preserve_source_occurrence():
    phones = ["S", "K", "UW", "L"] * 2
    rows = _rows_with_errors(phones, {5})
    errors = build_pronunciation_errors(
        "School, school!",
        "S K UW L | S K UW L",
        rows,
    )
    assert errors[0]["word_index"] == 2
    assert errors[0]["reference_span"] == {
        "start": 9,
        "end": 10,
        "text": "ch",
        "kind": "grapheme",
    }


def test_offsets_are_utf16_code_units_for_browser_slicing():
    text = "🙂 school"
    phones = ["S", "K", "UW", "L"]
    error = build_pronunciation_errors(
        text,
        "S K UW L",
        _rows_with_errors(phones, {1}),
    )[0]
    span = error["reference_span"]
    assert (span["start"], span["end"]) == (4, 5)
    assert _utf16_inclusive_slice(text, span["start"], span["end"]) == span["text"] == "ch"


def test_all_correct_alignment_has_no_pronunciation_errors():
    phones = ["S", "K", "UW", "L"]
    assert build_pronunciation_errors(
        "school",
        "S K UW L",
        _rows_with_errors(phones, set()),
    ) == []
