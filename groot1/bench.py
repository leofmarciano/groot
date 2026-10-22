"""Latency benchmark for Groot checkpoints (and, optionally, the Jev API).

    groot-bench --model runs/groot-1-s/final [--device cpu --threads 8] [--jev]

Measures p50/p95 end-to-end latency of `GrootEngine.answer` for synthetic
requests at several state sizes and question counts. With --jev and a
TYPESAFE_API_KEY in the environment, the same requests are sent to
https://api.typesafe.ai/v1/systemone for a side-by-side table.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from .schema import SystemOneRequest

LOREM = (
    "The customer reported that the invoice was charged twice on the corporate card and asked for a refund. "
    "Support confirmed the duplicate transaction and escalated the case to billing with high priority. "
)


def synthetic_request(state_words: int, n_questions: int) -> SystemOneRequest:
    words = (LOREM * (state_words // len(LOREM.split()) + 1)).split()[:state_words]
    state = " ".join(words)
    questions = {}
    kinds = ["choice", "score", "noul"]
    for i in range(n_questions):
        k = kinds[i % 3]
        if k == "choice":
            questions[f"q{i}"] = {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "payments", "shipping": "delivery", "access": "login", "other": "anything else"}}
        elif k == "score":
            questions[f"q{i}"] = {"type": "score", "instructions": "How urgent is it?", "criteria": ["can wait", "this week", "today"]}
        else:
            questions[f"q{i}"] = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
    return SystemOneRequest(state=state, questions=questions)


def time_fn(fn, warmup: int = 2, runs: int = 10) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"p50_ms": round(statistics.median(times), 1), "p95_ms": round(times[int(0.95 * (len(times) - 1))], 1), "min_ms": round(times[0], 1)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="groot-bench")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--jev", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    from .serve import GrootEngine

    engine = GrootEngine(args.model, args.device)
    grid = [(64, 1), (256, 1), (256, 5), (1024, 5), (1024, 20), (4096, 5)]
    rows = []
    for words, nq in grid:
        req = synthetic_request(words, nq)
        enc = engine.ser.encode_request(req.state, req.questions)
        res = time_fn(lambda: engine.answer(req.state, req.questions), runs=args.runs)
        row = {"state_tokens": enc.state.tokens, "chunks": len(enc.state.chunks), "questions": nq, **{f"groot_{k}": v for k, v in res.items()}}
        if args.jev and os.environ.get("TYPESAFE_API_KEY"):
            import httpx

            body = {"model": "jev-latest", "state": req.state, "questions": {k: v.model_dump(exclude_none=True) for k, v in req.questions.items()}}
            headers = {"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"}
            with httpx.Client(timeout=30) as client:
                res_j = time_fn(lambda: client.post("https://api.typesafe.ai/v1/systemone", json=body, headers=headers).raise_for_status(), warmup=1, runs=max(3, args.runs // 2))
            row.update({f"jev_{k}": v for k, v in res_j.items()})
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"device": str(engine.device), "threads": torch.get_num_threads(), "model": engine.name, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
