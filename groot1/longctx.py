"""Long-context data for Groot 1: distractor augmentation and needle tests.

    python -m groot1.longctx augment --rows base.jsonl --distractors agnews.jsonl race.jsonl --out long.jsonl \
        --n 20000 --min-tokens 4000 --max-tokens 32000
    python -m groot1.longctx needles --distractors ... --out needle_train.jsonl --split train --per-length 400 --lengths 2000 4000 8000 16000 32000
    python -m groot1.longctx needles --distractors ... --out needle_eval.jsonl  --split eval  --per-length 100 --lengths 1000 4000 16000 32000 64000

`augment` hides a row's original state at a random position among unrelated
documents, so the gold label is unchanged while the state grows to the target
length. `needles` builds facts with exact answers inside distractor text; the
train and eval splits use disjoint templates and entity names so the eval
measures retrieval, not template memorisation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from typing import Iterable

from .data import NOUL_OPTIONS, choice_options, make_row, read_jsonl, write_jsonl
from .schema import render_content

CHARS_PER_TOKEN = 3.2


def load_distractors(paths: Iterable[str], limit: int = 200_000) -> list[str]:
    """Plain-text passages taken from the states of unrelated decision rows."""
    pool: list[str] = []
    for p in paths:
        for r in read_jsonl(p):
            if len(pool) >= limit:
                break
            text = render_content(r["state"]).strip()
            if len(text) > 200:
                pool.append(text)
    if not pool:
        raise ValueError("no distractor passages")
    return pool


def fill_to(rng: random.Random, pool: list[str], chars: int) -> list[str]:
    docs, n = [], 0
    while n < chars:
        d = rng.choice(pool)
        docs.append(d)
        n += len(d) + 2
    return docs


def log_uniform(rng: random.Random, lo: int, hi: int) -> int:
    return int(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def augment(rows: list[dict], pool: list[str], n: int, lo: int, hi: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        r = rng.choice(rows)
        target_chars = int(log_uniform(rng, lo, hi) * CHARS_PER_TOKEN)
        original = render_content(r["state"])
        docs = fill_to(rng, pool, max(0, target_chars - len(original)))
        docs.insert(rng.randint(0, len(docs)), original)
        out.append(make_row(f"long_{r['source']}", f"{r['id']}/L{i}", {"documents": docs}, r["question"], r["type"], r["options"],
                            r["target"], r.get("split", "train"), r.get("lang", "en")))
    return out


# ------------------------------------------------------------------ needles
COLORS = ["red", "blue", "green", "amber", "violet", "black", "white", "orange"]
TRAIN_TEMPLATES = [
    ("Inventory note: the access code for vault {e} is {v}.", "Which access code does vault {e} use?",
     "the access code for vault {e} is {v}."),
    ("Ops log: service {e} was assigned the color tag {v} during the migration.", "Which color tag was service {e} assigned?",
     "service {e} was assigned the color tag {v}."),
    ("Memo: contract {e} must be filed under the {v} folder.", "Under which folder must contract {e} be filed?",
     "contract {e} must be filed under the {v} folder."),
]
EVAL_TEMPLATES = [
    ("Audit trail entry: the badge issued to employee {e} is {v}.", "What badge was issued to employee {e}?",
     "the badge issued to employee {e} is {v}."),
    ("Shipping manifest: crate {e} is sealed with a {v} seal.", "What seal is used on crate {e}?",
     "crate {e} is sealed with a {v} seal."),
]
TRAIN_NAMES = [f"K-{i:03d}" for i in range(500)]
EVAL_NAMES = [f"Z{chr(65 + i % 26)}{i:02d}" for i in range(300)]


def needles(pool: list[str], split: str, per_length: int, lengths: list[int], seed: int = 0) -> list[dict]:
    rng = random.Random(seed + (0 if split == "train" else 1000))
    templates = TRAIN_TEMPLATES if split == "train" else EVAL_TEMPLATES
    names = TRAIN_NAMES if split == "train" else EVAL_NAMES
    out = []
    for L in lengths:
        for j in range(per_length):
            fact_t, q_t, claim_t = rng.choice(templates)
            e, v = rng.choice(names), rng.choice(COLORS)
            e2 = rng.choice([x for x in names if x != e])  # decoy: same template, other entity, other value
            v2 = rng.choice([c for c in COLORS if c != v])
            docs = fill_to(rng, pool, int(L * CHARS_PER_TOKEN))
            depth = rng.random()
            docs.insert(int(depth * len(docs)), fact_t.format(e=e, v=v))
            docs.insert(rng.randint(0, len(docs)), fact_t.format(e=e2, v=v2))
            state = {"documents": docs}
            opts = COLORS[:]
            rng.shuffle(opts)
            opts = opts[:6] if v in opts[:6] else opts[:5] + [v]
            rid = f"needle_{split}_{L}_{j}"
            c = make_row("needle", rid + "/c", state, q_t.format(e=e), "choice", choice_options(opts),
                         [1.0 if o == v else 0.0 for o in opts], split, "en")
            truth = rng.random() < 0.5
            claim_v = v if truth else rng.choice([x for x in COLORS if x != v])
            n = make_row("needle", rid + "/n", state, "The documents state that " + claim_t.format(e=e, v=claim_v),
                         "noul", NOUL_OPTIONS, [0.0, 1.0] if truth else [1.0, 0.0], split, "en")
            c["needle"] = n["needle"] = {"length": L, "depth": round(depth, 3)}
            out += [c, n]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="groot-longctx")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("augment")
    a.add_argument("--rows", required=True)
    a.add_argument("--distractors", nargs="+", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=20000)
    a.add_argument("--min-tokens", type=int, default=4000)
    a.add_argument("--max-tokens", type=int, default=32000)
    a.add_argument("--seed", type=int, default=0)
    nd = sub.add_parser("needles")
    nd.add_argument("--distractors", nargs="+", required=True)
    nd.add_argument("--out", required=True)
    nd.add_argument("--split", choices=["train", "eval"], required=True)
    nd.add_argument("--per-length", type=int, default=100)
    nd.add_argument("--lengths", type=int, nargs="+", default=[1000, 4000, 16000, 32000, 64000])
    nd.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    pool = load_distractors(args.distractors)
    if args.cmd == "augment":
        n = write_jsonl(augment(list(read_jsonl(args.rows)), pool, args.n, args.min_tokens, args.max_tokens, args.seed), args.out)
    else:
        n = write_jsonl(needles(pool, args.split, args.per_length, args.lengths, args.seed), args.out)
    print(json.dumps({"wrote": n, "out": args.out, "distractor_pool": len(pool)}))


if __name__ == "__main__":
    main()
