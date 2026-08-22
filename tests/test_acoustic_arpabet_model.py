"""Contract tests for the custom ARPAbet decoder used by the acoustic model."""

from __future__ import annotations

import json

from api import acoustic


def test_training_vocab_matches_versioned_label_map():
    vocab = json.loads((acoustic.MODEL_PATH / "vocab.json").read_text(encoding="utf-8"))

    assert len(acoustic.ARPABET_VOCAB) == 41
    assert tuple(token for token, _ in sorted(vocab.items(), key=lambda item: item[1])) == (
        acoustic.ARPABET_VOCAB
    )
    assert vocab["[PAD]"] == acoustic.CTC_BLANK_ID == 0
    assert vocab["[UNK]"] == acoustic.CTC_UNKNOWN_ID == 1


def test_ctc_decoder_collapses_repeats_and_removes_blank():
    # S S blank K K UW UW blank L -> S K UW L
    ids = [30, 30, 0, 21, 21, 35, 35, 0, 22]

    assert acoustic.decode_ctc_ids_to_arpabet(ids) == ("S", "K", "UW", "L")


def test_ctc_decoder_drops_unknown_without_merging_neighbors(caplog):
    assert acoustic.decode_ctc_ids_to_arpabet([30, 1, 30]) == ("S", "S")
    assert "Dropping [UNK]" in caplog.text


def test_arpabet_prediction_becomes_compact_internal_ipa():
    assert acoustic.arpabet_tokens_to_raw_ctc_ipa(("S", "K", "UW", "L")) == "skul"
    assert acoustic.arpabet_tokens_to_raw_ctc_ipa(("CH", "EY", "N", "JH")) == "tʃeɪndʒ"


def test_checkpoint_config_matches_decoder_contract():
    config = json.loads((acoustic.MODEL_PATH / "config.json").read_text(encoding="utf-8"))

    assert config["vocab_size"] == len(acoustic.ARPABET_VOCAB)
    assert config["pad_token_id"] == acoustic.CTC_BLANK_ID
