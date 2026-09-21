"""Prompt construction, truncation, and option rendering (unit)."""

from laya_linux.prompts import (
    QTYPES,
    build_sequence,
    confidence_from_probs,
    render_options,
    temp_bucket,
)


def test_render_options_choice(q="unused"):
    q = {"t": "choice", "ins": "i", "crit": {"a": None, "b": "desc", "c": 0}}
    assert render_options(q) == ["a", "b: desc", "c: 0"]


def test_render_options_score():
    q = {"t": "score", "ins": "i", "crit": ["low", "high"]}
    assert render_options(q) == ["level 0: low", "level 1: high"]


def test_render_options_noul_defaults():
    q = {"t": "noul", "ins": "i", "crit": None}
    assert render_options(q) == [
        "false: no, the statement does not hold",
        "true: yes, the statement holds",
    ]


def test_render_options_noul_criteria():
    q = {"t": "noul", "ins": "i", "crit": {"false": "not it", "true": "definitely"}}
    assert render_options(q) == ["false: not it", "true: definitely"]


def test_render_criterion_structured():
    from laya_linux.prompts import render_criterion

    assert render_criterion({"desc": "x"}) == '{"desc": "x"}'
    assert render_criterion("plain") == "plain"


def test_build_sequence_shape(small_tok):
    q = {"t": "choice", "ins": "Pick one.", "crit": {"x": None, "y": None}}
    ids, markers = build_sequence(small_tok, {"m": "hello tok5"}, q, max_len=128, head_max_len=64)
    assert ids[0] == small_tok.cls_token_id
    assert ids.count(small_tok.mask_token_id) == 2  # one marker per option
    assert ids[-1] == small_tok.sep_token_id
    assert len(markers) == 2
    assert markers[0] < markers[1] < len(ids)


def test_build_sequence_truncates_state(small_tok):
    q = {"t": "noul", "ins": "i", "crit": None}
    long_state = "tok1 " * 500
    ids, markers = build_sequence(small_tok, long_state, q, max_len=64, head_max_len=32)
    assert len(ids) == 64
    assert all(m < 64 for m in markers)


def test_build_sequence_mask_token_never_in_text(small_tok):
    q = {"t": "noul", "ins": "has <|mask|> inside", "crit": None}
    ids, markers = build_sequence(small_tok, "state <|mask|> too", q, max_len=128, head_max_len=64)
    # mask tokens appear only as the two option markers
    assert ids.count(small_tok.mask_token_id) == 2


def test_confidence_extremes():
    assert confidence_from_probs([1.0, 0.0], 2) == 1.0
    assert confidence_from_probs([0.5, 0.5], 2) == 0.0


def test_temp_buckets():
    assert temp_bucket(QTYPES["noul"], 2) == "noul:2"
    assert temp_bucket(QTYPES["choice"], 4) == "choice:3-5"
    assert temp_bucket(QTYPES["choice"], 12) == "choice:11+"
    assert temp_bucket(QTYPES["score"], 10) == "score:6-10"
