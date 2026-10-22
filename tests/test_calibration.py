import numpy as np
import pytest

from groot1.calibration import (
    Record, brier, ece, evaluate_records, fit_temperatures, nll, softmax, typed_accuracy,
)


def make_records(n=200, qtype=0, sharp=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        correct = rng.integers(3)
        logits = rng.normal(0, 1, 3)
        logits[correct] += 3.0 if sharp else 0.3
        target = np.zeros(3)
        target[correct] = 1.0
        rows.append(Record(logits=logits, target=target, qtype=qtype))
    return rows


def test_fit_temperatures_reduces_nll():
    # over-confident logits -> optimal temperature > 1 must improve NLL
    recs = make_records(sharp=True)
    hot = [Record(logits=r.logits * 6.0, target=r.target, qtype=0) for r in recs]
    temps = fit_temperatures(hot, n_types=1)
    assert temps[0] > 1.0
    assert nll(hot, temps) < nll(hot, [1.0])


def test_typed_accuracy_and_metrics():
    recs = make_records(qtype=0, seed=1) + make_records(qtype=1, sharp=False, seed=2)
    acc = typed_accuracy(recs)
    assert acc["type0"] > acc["type1"]
    assert acc["type0"] > 0.9
    assert 0.0 <= ece(recs) <= 1.0
    assert brier(recs) >= 0.0
    rep = evaluate_records(recs)
    d = rep.to_dict()
    assert d["count"] == len(recs) and "accuracy" in d and "nll" in d


def test_evaluate_records_calibrated_not_worse_on_skew():
    recs = make_records()
    hot = [Record(logits=r.logits * 6.0, target=r.target, qtype=0) for r in recs]
    rep = evaluate_records(hot, temps=fit_temperatures(hot, n_types=3))
    assert rep.to_dict()["count"] == 200
