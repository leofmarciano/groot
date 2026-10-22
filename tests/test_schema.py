import pytest
from groot1.schema import (
    ChoiceQuestion, ScoreQuestion, NoulQuestion, SystemOneRequest, build_answer,
    confidence_choice, confidence_score, confidence_noul, expected_score, render_content, softmax,
)


def test_choice_confidence_matches_docs():
    assert confidence_choice([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert confidence_choice([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert confidence_choice([0.8, 0.1, 0.1]) == pytest.approx((3 * 0.8 - 1) / 2)


def test_score_confidence_matches_docs():
    assert confidence_score([0.0, 0.57, 0.43]) == pytest.approx(1 - 0.43 / (2 / 3), abs=1e-6)
    assert confidence_score([0.0, 0.5, 0.5]) == pytest.approx(0.25)
    assert confidence_score([0.5, 0.0, 0.5]) == pytest.approx(0.0)
    assert confidence_score([0.0, 1.0, 0.0]) == pytest.approx(1.0)


def test_noul_confidence():
    assert confidence_noul(0.5) == 0.0
    assert confidence_noul(0.99) == pytest.approx(0.98)


def test_expected_score_and_softmax():
    assert expected_score([0.1, 0.1, 0.8]) == pytest.approx(1.7)
    p = softmax([1.0, 2.0, 3.0])
    assert sum(p) == pytest.approx(1.0) and p[2] > p[1] > p[0]


def test_build_answers():
    q = ChoiceQuestion(instructions="tone?", criteria={"angry": "upset", "calm": None})
    a = build_answer(q, [0.9, 0.1])
    assert a.choice == "angry" and a.probabilities["calm"] == pytest.approx(0.1)
    s = ScoreQuestion(criteria=["low", "mid", "high"])
    b = build_answer(s, [0.0, 0.57, 0.43])
    assert b.score == pytest.approx(1.43) and b.legend["2"] == "high"
    n = NoulQuestion(instructions="is spam?")
    c = build_answer(n, [0.2, 0.8])
    assert c.noul == pytest.approx(0.8)


def test_request_validation_limits():
    with pytest.raises(Exception):
        ChoiceQuestion(criteria={})
    with pytest.raises(Exception):
        ScoreQuestion(criteria=[f"l{i}" for i in range(11)])
    with pytest.raises(Exception):
        NoulQuestion()
    req = SystemOneRequest(state="hi", questions={"q": {"type": "noul", "instructions": "ok?"}})
    assert isinstance(req.questions["q"], NoulQuestion)


def test_render_content_json():
    text = render_content({"a": 1, "b": {"c": "x"}, "d": [1, {"e": True}]})
    assert "a: 1" in text and "c: x" in text and "[1]" in text and "e: true" in text
