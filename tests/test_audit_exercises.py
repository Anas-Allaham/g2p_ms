"""Audit #6 & #7: functional exercise types + confusion retrieval; insertions."""

from src.core.g2p import content
from src.core.g2p.g2p_service import g2p_convert
from src.core.g2p.tokenization import ipa_to_tokens
from src.core.scoring.scoring import align_phonemes, calculate_metrics


# ---- #6: minimal pairs / isolated words for low mastery ----
def test_minimal_pair_lookup_is_order_independent():
    assert content.minimal_pair_words("θ", "s") == content.minimal_pair_words("s", "θ")
    assert content.minimal_pair_words("θ", "s")  # non-empty for a known confusion


def test_low_mastery_builds_minimal_pair_drill():
    ex = content.make_low_mastery_exercise("θ", "s", g2p_convert, ipa_to_tokens)
    assert ex is not None
    assert ex["source"] == "minimal_pairs"
    # It really contrasts θ and s and is short (isolated words, not a sentence).
    assert "think" in ex["text"] and "sink" in ex["text"]
    assert ex["word_count"] <= 8


def test_low_mastery_falls_back_to_isolated_words():
    ex = content.make_low_mastery_exercise("ŋ", None, g2p_convert, ipa_to_tokens)
    assert ex is not None
    assert ex["source"] == "isolated_words"


# ---- #6: exercise_type drives length band ----
def test_word_count_band_selection():
    from src.core.exercises.services import _word_count_range_for
    assert _word_count_range_for("short_phrase") == (
        content.SHORT_PHRASE_MIN_WORDS,
        content.SHORT_PHRASE_MAX_WORDS,
    )
    assert _word_count_range_for("targeted_sentence") == (
        content.TARGETED_SENTENCE_MIN_WORDS,
        content.TARGETED_SENTENCE_MAX_WORDS,
    )
    assert _word_count_range_for("maintenance")[0] == content.MAINTENANCE_MIN_WORDS

    short = {"id": 1, "text": "a", "phoneme_counts": {"s": 2}, "word_count": 3, "level_proxy": 5}
    complete_phrase = {
        "id": 2, "text": "b", "phoneme_counts": {"s": 2}, "word_count": 8, "level_proxy": 10
    }
    picked = content.pick_next_sentence(
        [short, complete_phrase],
        ["s"],
        word_count_range=(content.SHORT_PHRASE_MIN_WORDS, content.SHORT_PHRASE_MAX_WORDS),
    )
    assert picked["id"] == 2  # complete phrase within the richer short-phrase band


def test_generated_exercise_rejects_text_below_complexity_band():
    tagged = {
        "phoneme_counts": {"s": 2},
        "word_count": 5,
        "reference_g2p_trusted": True,
        "has_oov_words": False,
    }
    ok, reasons = content.verify_generated_exercise(
        tagged,
        ["s"],
        set(),
        min_word_count=content.SHORT_PHRASE_MIN_WORDS,
        max_word_count=content.SHORT_PHRASE_MAX_WORDS,
    )
    assert ok is False
    assert "too_short" in reasons


# ---- #6: retrieval is confusion-aware (not only LLM) ----
def test_retrieval_prefers_contrastive_sentence():
    plain = {"id": 1, "text": "a", "phoneme_counts": {"θ": 2, "k": 1}, "word_count": 6, "level_proxy": 8}
    contrastive = {"id": 2, "text": "b", "phoneme_counts": {"θ": 2, "s": 2}, "word_count": 6, "level_proxy": 8}
    picked = content.pick_next_sentence(
        [plain, contrastive], ["θ"], confusion_phonemes={"s"}
    )
    assert picked["id"] == 2  # covers the confused-with sound too


# ---- #7: insertions influence utterance score but not per-phoneme mastery ----
def test_insertions_tracked_at_utterance_level_only():
    # Reference "s", hypothesis "s s s" -> 2 insertions, no expected phoneme.
    rows = align_phonemes(["s"], ["s", "s", "s"])
    m = calculate_metrics(rows)
    assert m["insertion_count"] == 2
    assert m["insertion_penalty"] > 0
    # Utterance score is degraded by the insertions...
    assert m["utterance_score"] < 100.0
    # ...but no insertion row carries an expected phoneme (so mastery can't see it).
    for r in rows:
        if r["result"] == "insertion":
            assert r["expected"] == "-"


def test_insertions_do_not_update_phoneme_mastery():
    from src.core.scoring import mastery
    from datetime import datetime, timezone
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = align_phonemes(["s"], ["s", "t", "t"])  # extra t's are insertions
    stats = mastery.update_mastery_for_recording({}, rows, now)
    # Only the expected phoneme 's' gets evidence; inserted 't' does not appear.
    assert set(stats.keys()) == {"s"}
