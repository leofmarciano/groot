"""Needle-in-a-haystack: Groot vs Jev on the same requests, accuracy and latency per state length.

    python scripts/needle_eval.py --rows needle_eval.jsonl --model runs/x/final [--jev] [--limit-per-length 50] --out needle.json
"""
import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from groot1.data import read_jsonl


def to_question(r):
    if r["type"] == "choice":
        crit = {}
        for o in r["options"]:
            name, _, desc = o.partition(": ")
            crit[name] = desc or None
        return {"type": "choice", "instructions": r["question"], "criteria": crit}, list(crit)
    return {"type": "noul", "instructions": r["question"]}, ["false", "true"]


def gold_of(r, names):
    return names[max(range(len(r["target"])), key=r["target"].__getitem__)]


def pred_of(ans, kind):
    if kind == "choice":
        return ans["choice"]
    return "true" if ans["noul"] >= 0.5 else "false"


def run_groot(rows, model, device, overrides=None):
    from groot1.serve import GrootEngine
    from groot1.schema import SystemOneRequest

    eng = GrootEngine(model, device=device, overrides=overrides)
    out = []
    for r in rows:
        q, names = to_question(r)
        req = SystemOneRequest(state=r["state"], questions={"q": q})
        t0 = time.perf_counter()
        resp = eng.answer(req.state, req.questions)
        dt = (time.perf_counter() - t0) * 1000
        a = resp.answers["q"].model_dump()
        out.append({"ok": pred_of(a, r["type"]) == gold_of(r, names), "ms": dt, "tokens": resp.usage.input_tokens})
    return out


def run_jev(rows, workers=8):
    import httpx

    key = os.environ["TYPESAFE_API_KEY"]

    def one(r):
        q, names = to_question(r)
        long = r["needle"]["length"] > 30000
        headers = {"Authorization": f"Bearer {key}", **({"TypeSafe-Danger-Zone-Long-Context": "true"} if long else {})}
        body = {"model": "jev-latest", "state": r["state"], "questions": {"q": q}}
        for attempt in range(4):
            try:
                t0 = time.perf_counter()
                resp = httpx.post("https://api.typesafe.ai/v1/systemone", json=body, headers=headers, timeout=120)
                dt = (time.perf_counter() - t0) * 1000
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code != 200:
                    return {"ok": False, "ms": dt, "error": f"{resp.status_code} {resp.text[:120]}"}
                a = resp.json()["answers"]["q"]
                return {"ok": pred_of(a, r["type"]) == gold_of(r, names), "ms": dt, "tokens": resp.json()["usage"]["input_tokens"]}
            except Exception as e:  # network hiccup
                err = str(e)
                time.sleep(2 ** attempt)
        return {"ok": False, "ms": None, "error": err}

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, rows))


def summarize(rows, res):
    by = defaultdict(list)
    for r, x in zip(rows, res):
        by[r["needle"]["length"]].append(x)
    out = {}
    for L in sorted(by):
        xs = by[L]
        ms = [x["ms"] for x in xs if x.get("ms") is not None]
        out[L] = {"n": len(xs), "acc": round(sum(x["ok"] for x in xs) / len(xs), 3), "p50_ms": round(statistics.median(ms), 1) if ms else None,
                  "errors": sum(1 for x in xs if "error" in x)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jev", action="store_true")
    ap.add_argument("--limit-per-length", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    args = ap.parse_args()
    rows, seen = [], defaultdict(int)
    for r in read_jsonl(args.rows):
        L = r["needle"]["length"]
        if args.limit_per_length and seen[L] >= args.limit_per_length:
            continue
        seen[L] += 1
        rows.append(r)
    report = {}
    if args.model:
        from groot1.config import parse_overrides

        report["groot"] = summarize(rows, run_groot(rows, args.model, args.device, parse_overrides(args.set)))
        print("groot", json.dumps(report["groot"]), flush=True)
    if args.jev:
        res = run_jev(rows)
        report["jev"] = summarize(rows, res)
        errs = [x["error"] for x in res if "error" in x][:3]
        print("jev", json.dumps(report["jev"]), ("errors: " + str(errs)) if errs else "", flush=True)
    if args.out:
        open(args.out, "w").write(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
