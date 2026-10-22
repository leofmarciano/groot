"""Head-to-head benchmark: Groot vs Julia-1 vs Jev on identical requests.

    python scripts/benchmark.py build --test data/v1/test.jsonl --per-source 500 --sources typed_decisions agnews ... --out bench.jsonl
    python scripts/benchmark.py run --rows bench.jsonl --groot runs/x/final --groot-set state_retrieval=true \
        --julia models/Julia-1 --jev --out bench_results.json
    python scripts/benchmark.py table --results bench_results.json

Accuracy is argmax agreement with the gold distribution's argmax; Brier is the
multiclass Brier score against the (possibly soft) gold distribution. Latency is
the median wall time of one request (one question). Groot and Julia run
in-process on the same GPU; Jev is called over its public HTTPS API, so its
latency includes the network round trip.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def build(test: str, sources: list[str], per_source: int, out: str, seed: int = 0) -> None:
    by: dict[str, list[dict]] = defaultdict(list)
    for line in open(test, encoding="utf-8"):
        r = json.loads(line)
        if r["source"] in sources:
            by[r["source"]].append(r)
    rng = random.Random(seed)
    rows, counts = [], {}
    for s in sources:
        xs = by.get(s, [])
        rng.shuffle(xs)
        take = xs if s == "typed_decisions" else xs[:per_source]
        counts[s] = len(take)
        rows += take
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(counts))


def to_question(r: dict) -> tuple[dict, list[str]]:
    if r["type"] == "choice":
        crit = {}
        for o in r["options"]:
            name, sep, desc = o.partition(": ")
            crit[name] = desc if sep else None
        return {"type": "choice", "instructions": r["question"], "criteria": crit}, list(crit)
    if r["type"] == "score":
        levels = [o.split(": ", 1)[1] if o.startswith("level ") and ": " in o else o for o in r["options"]]
        return {"type": "score", "instructions": r["question"], "criteria": levels}, [str(i) for i in range(len(levels))]
    q = {"type": "noul", "instructions": r["question"]}
    crit = {}
    for o, k in zip(r["options"], ("false", "true")):
        _, sep, desc = o.partition(": ")
        if sep and desc:
            crit[k] = desc
    if crit:  # keep the dataset's Boolean descriptions (false/true order); literal false/true otherwise
        q["criteria"] = crit
    return q, ["false", "true"]


def probs_of(answer: dict, kind: str, keys: list[str]) -> list[float]:
    if kind == "noul":
        p = float(answer["noul"])
        return [1 - p, p]
    pr = answer["probabilities"]
    v = [float(pr.get(k, 0.0)) for k in keys]
    s = sum(v) or 1.0
    return [x / s for x in v]


def score(r: dict, p: list[float]) -> tuple[bool, float]:
    t = r["target"]
    ok = max(range(len(p)), key=p.__getitem__) == max(range(len(t)), key=t.__getitem__)
    return ok, sum((a - b) ** 2 for a, b in zip(p, t))


def run_groot(rows, ckpt, overrides):
    from groot1.config import parse_overrides
    from groot1.schema import SystemOneRequest
    from groot1.serve import GrootEngine

    eng = GrootEngine(ckpt, overrides=parse_overrides(overrides))
    out = []
    for r in rows:
        q, keys = to_question(r)
        req = SystemOneRequest(state=r["state"], questions={"q": q})
        t0 = time.perf_counter()
        a = eng.answer(req.state, req.questions).answers["q"].model_dump()
        dt = (time.perf_counter() - t0) * 1000
        ok, br = score(r, probs_of(a, r["type"], keys))
        out.append({"ok": ok, "brier": br, "ms": dt})
    return out


def run_julia(rows, path, device="cuda"):
    import julia

    try:  # Julia's optional fast path breaks on recent transformers; the standard forward is numerically equivalent
        import julia.router.encoder as je

        je.specialize_decision_encoder = lambda model: False
    except Exception:
        pass
    eng = julia.load_model(path, device=device, max_length=8192, head_length=512)
    out = []
    for r in rows:
        q, keys = to_question(r)
        if r["type"] == "choice" and not 2 <= len(keys) <= 20:
            out.append({"skip": f"{len(keys)} options > Julia's 20-option native limit"})
            continue
        if q["type"] == "choice":  # Julia requires a nonempty description per option; bare labels describe themselves
            q = {**q, "criteria": {k: v or k.replace("_", " ") for k, v in q["criteria"].items()}}
        try:
            t0 = time.perf_counter()
            res = eng.predict(state=r["state"], questions={"q": q})
            dt = (time.perf_counter() - t0) * 1000
            pr = res["answers"]["q"]["probabilities"]
            probs = [float(pr.get(k, 0.0)) for k in keys]
            s = sum(probs) or 1.0
            ok, br = score(r, [x / s for x in probs])
            out.append({"ok": ok, "brier": br, "ms": dt})
        except Exception as e:
            out.append({"skip": f"error: {str(e)[:120]}"})
    return out


def run_jev(rows, workers=8):
    import httpx

    key = os.environ["TYPESAFE_API_KEY"]

    def one(r):
        q, keys = to_question(r)
        body = {"model": "jev-latest", "state": r["state"], "questions": {"q": q}}
        headers = {"Authorization": f"Bearer {key}"}
        err = None
        for attempt in range(5):
            try:
                t0 = time.perf_counter()
                resp = httpx.post("https://api.typesafe.ai/v1/systemone", json=body, headers=headers, timeout=120)
                dt = (time.perf_counter() - t0) * 1000
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code != 200:
                    return {"skip": f"{resp.status_code} {resp.text[:100]}"}
                ok, br = score(r, probs_of(resp.json()["answers"]["q"], r["type"], keys))
                return {"ok": ok, "brier": br, "ms": dt}
            except Exception as e:
                err = str(e)[:100]
                time.sleep(2 ** attempt)
        return {"skip": f"error: {err}"}

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, rows))


def summarize(rows, res):
    by = defaultdict(list)
    for r, x in zip(rows, res):
        by[r["source"]].append(x)
        if r["source"] == "typed_decisions":
            by["typed_decisions / " + r["type"]].append(x)
        if r["source"].startswith("massive_"):  # scenario (18 labels) and intent (60 labels) questions
            by[f"{r['source']} / {len(r['options'])} labels"].append(x)
    out = {}
    for k, xs in by.items():
        done = [x for x in xs if "skip" not in x]
        if not done:
            out[k] = {"n": 0, "skipped": len(xs), "reason": xs[0].get("skip")}
            continue
        out[k] = {"n": len(done), "correct": sum(x["ok"] for x in done), "acc": sum(x["ok"] for x in done) / len(done),
                  "brier": sum(x["brier"] for x in done) / len(done), "p50_ms": statistics.median(x["ms"] for x in done),
                  "skipped": len(xs) - len(done)}
    return out


def table(results: dict) -> str:
    systems = [s for s in ("groot", "julia", "jev") if s in results]
    keys = sorted({k for s in systems for k in results[s]}, key=lambda k: (" / " in k, k))
    lines = ["| Benchmark | n | " + " | ".join(f"{s.capitalize()} acc | {s.capitalize()} Brier" for s in systems) + " |",
             "|" + "---|" * (2 + 2 * len(systems))]
    for k in keys:
        n = max(results[s].get(k, {}).get("n", 0) for s in systems)
        cells = []
        for s in systems:
            v = results[s].get(k)
            cells += [f"{v['acc']:.3f}", f"{v['brier']:.3f}"] if v and v.get("n") else ["n/a", "n/a"]
        lines.append(f"| {k} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--test", required=True)
    b.add_argument("--sources", nargs="+", required=True)
    b.add_argument("--per-source", type=int, default=500)
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rows", required=True)
    r.add_argument("--groot", default=None)
    r.add_argument("--groot-set", action="append", default=[])
    r.add_argument("--julia", default=None)
    r.add_argument("--jev", action="store_true")
    r.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("--results", required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.test, args.sources, args.per_source, args.out)
        return
    if args.cmd == "table":
        print(table(json.loads(Path(args.results).read_text())))
        return
    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8")]
    results = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    for name, fn in (("groot", lambda: run_groot(rows, args.groot, args.groot_set) if args.groot else None),
                     ("julia", lambda: run_julia(rows, args.julia) if args.julia else None),
                     ("jev", lambda: run_jev(rows) if args.jev else None)):
        res = fn()
        if res is None:
            continue
        results[name] = summarize(rows, res)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"{name} done", flush=True)
    print(table(results))


if __name__ == "__main__":
    main()
