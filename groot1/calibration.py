"""Calibration and decision metrics.

* `fit_temperatures`  — per-question-type temperature scaling on a calibration
                        split (minimises NLL of the soft/hard targets).
* `ece`, `brier`, `nll` — standard proper-scoring and reliability metrics.
* `typed_accuracy`    — argmax agreement, per type and overall.

Temperature scaling keeps the argmax (and therefore accuracy) unchanged and
only reshapes the probabilities, which is what the System One contract needs:
`confidence` is a pure function of `probabilities` (see schema.py).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass
class Record:
    """One scored question: raw logits, target distribution, question type."""

    logits: np.ndarray  # (k,)
    target: np.ndarray  # (k,) probabilities (one-hot for hard labels)
    qtype: int
    correct_conf: float | None = None  # model's correctness estimate (sigmoid of confidence head)


def softmax(x: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = x / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def nll(records: Sequence[Record], temps: Sequence[float] | None = None) -> float:
    total = 0.0
    for r in records:
        t = temps[r.qtype] if temps is not None else 1.0
        p = softmax(r.logits, t)
        total += -float((r.target * np.log(np.clip(p, 1e-12, 1))).sum())
    return total / max(1, len(records))


def brier(records: Sequence[Record], temps: Sequence[float] | None = None) -> float:
    total = 0.0
    for r in records:
        t = temps[r.qtype] if temps is not None else 1.0
        p = softmax(r.logits, t)
        total += float(((p - r.target) ** 2).sum())
    return total / max(1, len(records))


def typed_accuracy(records: Sequence[Record]) -> dict[str, float]:
    hits: dict[int, list[int]] = defaultdict(list)
    for r in records:
        hits[r.qtype].append(int(r.logits.argmax() == r.target.argmax()))
    out = {f"type{t}": float(np.mean(v)) for t, v in hits.items()}
    out["all"] = float(np.mean([h for v in hits.values() for h in v])) if hits else float("nan")
    return out


def ece(records: Sequence[Record], temps: Sequence[float] | None = None, bins: int = 15) -> float:
    """Expected calibration error of the top-1 probability (argmax vs target argmax)."""
    conf, acc = [], []
    for r in records:
        t = temps[r.qtype] if temps is not None else 1.0
        p = softmax(r.logits, t)
        conf.append(float(p.max()))
        acc.append(float(p.argmax() == r.target.argmax()))
    conf_a, acc_a = np.array(conf), np.array(acc)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf_a > lo) & (conf_a <= hi)
        if m.any():
            total += m.mean() * abs(acc_a[m].mean() - conf_a[m].mean())
    return float(total)


def fit_temperatures(records: Sequence[Record], n_types: int = 3, grid: Iterable[float] | None = None) -> list[float]:
    """Per-type temperature minimising NLL on a log-spaced grid, then refined."""
    grid = list(grid) if grid is not None else list(np.exp(np.linspace(np.log(0.25), np.log(8.0), 61)))
    temps = [1.0] * n_types
    for t in range(n_types):
        subset = [r for r in records if r.qtype == t]
        if not subset:
            continue
        best_t, best_v = 1.0, float("inf")
        for cand in grid:
            v = nll(subset, [cand] * n_types)
            if v < best_v:
                best_t, best_v = cand, v
        # local refinement
        for cand in np.linspace(best_t * 0.9, best_t * 1.1, 21):
            v = nll(subset, [float(cand)] * n_types)
            if v < best_v:
                best_t, best_v = float(cand), v
        temps[t] = float(best_t)
    return temps


@dataclass
class Report:
    accuracy: dict[str, float] = field(default_factory=dict)
    nll: float = float("nan")
    brier: float = float("nan")
    ece: float = float("nan")
    temperatures: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    count: int = 0

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "accuracy": self.accuracy,
            "nll": self.nll,
            "brier": self.brier,
            "ece": self.ece,
            "temperatures": self.temperatures,
        }


def evaluate_records(records: Sequence[Record], temps: Sequence[float] | None = None) -> Report:
    return Report(
        accuracy=typed_accuracy(records),
        nll=nll(records, temps),
        brier=brier(records, temps),
        ece=ece(records, temps),
        temperatures=list(temps) if temps is not None else [1.0, 1.0, 1.0],
        count=len(records),
    )
