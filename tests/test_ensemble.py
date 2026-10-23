import json

import pytest

from groot1.schema import SystemOneRequest, Usage
from groot1.serve import GrootEngine


class FakeMember:
    def __init__(self, probs, p_correct, cfg=None):
        self.probs, self.p_correct, self.cfg = probs, p_correct, cfg

    def forward(self, state, questions):
        extra = [{"p_correct": self.p_correct, "truncated": False} for _ in questions]
        return self.probs, extra, Usage(input_tokens=10, output_tokens=len(questions))


def ensemble(members, weights):
    eng = object.__new__(GrootEngine)
    eng.members, eng.weights, eng.name = members, weights, "groot-1"
    return eng


def test_ensemble_averages_member_probabilities():
    req = SystemOneRequest(state="x", questions={
        "c": {"type": "choice", "instructions": "pick", "criteria": {"a": None, "b": None}},
        "n": {"type": "noul", "instructions": "true?"},
    })
    eng = ensemble([FakeMember([[0.9, 0.1], [0.2, 0.8]], 0.8), FakeMember([[0.3, 0.7], [0.6, 0.4]], 0.4)], [0.5, 0.5])
    res = eng.answer(req.state, req.questions, extended=True)
    assert res.answers["c"].probabilities == pytest.approx({"a": 0.6, "b": 0.4})
    assert res.answers["c"].choice == "a"
    assert res.answers["n"].noul == pytest.approx(0.6)
    assert res.x_groot["c"]["p_correct"] == pytest.approx(0.6)
    assert res.model == "groot-1"


def test_ensemble_weights_are_respected():
    req = SystemOneRequest(state="x", questions={"c": {"type": "choice", "instructions": "pick", "criteria": {"a": None, "b": None}}})
    eng = ensemble([FakeMember([[1.0, 0.0]], 1.0), FakeMember([[0.0, 1.0]], 0.0)], [0.25, 0.75])
    assert eng.answer(req.state, req.questions).answers["c"].probabilities == pytest.approx({"a": 0.25, "b": 0.75})


def test_ensemble_spec_loads_members(tmp_path, monkeypatch):
    (tmp_path / "ensemble.json").write_text(json.dumps({"name": "groot-1", "members": ["s", "m"], "weights": [1, 3]}))
    loaded = []
    real_init = GrootEngine.__init__

    def fake_init(self, checkpoint, device=None, model_name=None, overrides=None):
        if (tmp_path / "ensemble.json") == (type(tmp_path)(checkpoint) / "ensemble.json"):
            return real_init(self, checkpoint, "cpu", model_name, overrides)
        loaded.append(type(tmp_path)(checkpoint).name)
        self.members, self.cfg = [], object()

    monkeypatch.setattr(GrootEngine, "__init__", fake_init)
    eng = GrootEngine(tmp_path)
    assert loaded == ["s", "m"]
    assert eng.weights == pytest.approx([0.25, 0.75])
    assert eng.name == "groot-1" and eng.aliases["jev-latest"] == "groot-1"
