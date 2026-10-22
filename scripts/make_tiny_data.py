"""Generate a tiny synthetic decision dataset for pipeline smoke tests.

Deterministic toy task: the state describes a colored box; the question asks
which color / how heavy / whether fragile, and targets follow a rule the model
can actually learn (so loss drops in a few steps).

Usage:
  python scripts/make_tiny_data.py --out /tmp/groot_tiny --n 96
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from groot1.data import make_row, write_jsonl  # noqa: E402

COLORS = ["red", "blue", "green", "amber"]


def gen_row(rng: random.Random, i: int, split: str):
    color = rng.choice(COLORS)
    weight = rng.randint(1, 10)
    fragile = rng.random() < 0.5
    state = {
        "text": f"A {color} box weighing {weight}kg, marked {'fragile' if fragile else 'sturdy'}.",
        "metadata": {"weight": weight, "fragile": fragile},
    }
    kind = i % 3
    if kind == 0:
        target = [1.0 if c == color else 0.01 for c in COLORS]
        return make_row("tiny", i, state, "Which color is the box?", "choice", COLORS, target, split)
    if kind == 1:
        band = min(3, weight // 3)  # 0..3 ordinal target
        target = [1.0 if j == band else 0.05 for j in range(4)]
        return make_row("tiny", i, state, "Rate the weight", "score",
                        [f"level {j}" for j in range(4)], target, split)
    target = [0.95, 0.05] if fragile else [0.05, 0.95]  # [false, true]
    return make_row("tiny", i, state, "The box is fragile", "noul", ["false", "true"], target, split)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=96)
    args = ap.parse_args()
    rng = random.Random(7)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", args.n), ("val", max(16, args.n // 4))):
        rows = [gen_row(rng, i, "train") for i in range(n)]
        path = out / f"{split}.jsonl"
        write_jsonl(rows, path)
        print(f"wrote {n} rows -> {path}")


if __name__ == "__main__":
    main()
