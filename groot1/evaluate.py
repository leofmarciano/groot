"""Evaluation of a Groot checkpoint on decision rows and public benchmarks.

    groot-eval --model runs/groot-1-s/final --rows data/test.jsonl [--max-rows 2000] [--out eval.json]
    groot-eval --model ... --bench typed_decisions   # LocalLLaMA/typed-decisions test split

Reports accuracy per type, NLL, Brier, ECE (with and without the stored
temperatures), coverage/accuracy at confidence thresholds, and per-source
breakdowns. Rows are the JSONL format of data.py.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .calibration import Record, evaluate_records, softmax
from .config import QTYPES
from .data import CONVERTERS, read_jsonl
from .model import GrootDecisionModel
from .schema import confidence_choice, confidence_noul, confidence_score
from .serialize import Serializer
from .train import evaluate


def confidence_of(probabilities: list[float], qtype: str) -> float:
    if qtype == "choice":
        return confidence_choice(probabilities)
    if qtype == "score":
        return confidence_score(probabilities)
    return confidence_noul(probabilities[1])


def coverage_table(records: list[Record], temps: list[float], thresholds=(0.3, 0.5, 0.7, 0.9)) -> dict:
    """Accuracy of answers whose System One confidence passes each threshold."""
    rows = []
    for r in records:
        p = softmax(r.logits, temps[r.qtype]).tolist()
        c = confidence_of(p, QTYPES[r.qtype])
        rows.append((c, int(np.argmax(p) == int(r.target.argmax()))))
    out = {}
    for th in thresholds:
        kept = [a for c, a in rows if c >= th]
        out[str(th)] = {"coverage": len(kept) / max(1, len(rows)), "accuracy": float(np.mean(kept)) if kept else None}
    return out


def per_source(rows: list[dict], records: list[Record], temps: list[float]) -> dict:
    groups: dict[str, list[Record]] = defaultdict(list)
    for row, rec in zip(rows, records):
        groups[row.get("source", "?")].append(rec)
    return {src: evaluate_records(recs, temps).to_dict() for src, recs in groups.items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="groot-eval")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--bench", default=None, help="name of a converter in data.py (uses its test split)")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", action="append", default=[], help="config override key=value (e.g. state_retrieval=true)")
    ap.add_argument("--calibrate-on", default=None, help="fit per-type temperatures on this JSONL before evaluating")
    ap.add_argument("--save-calibration", action="store_true", help="write the fitted temperatures into the checkpoint")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    model = GrootDecisionModel.from_pretrained(args.model, device=device)
    tok = AutoTokenizer.from_pretrained(Path(args.model) / "tokenizer")
    if args.set:
        from .config import GrootConfig, parse_overrides

        model.cfg = GrootConfig.from_dict({**model.cfg.to_dict(), **parse_overrides(args.set)})
    ser = Serializer(tok, model.cfg)
    if args.calibrate_on:
        from .calibration import fit_temperatures

        cal_rows = list(read_jsonl(args.calibrate_on))
        cal_records, _ = evaluate(model, cal_rows, ser, device, args.batch_size, None)
        temps = fit_temperatures(cal_records, n_types=len(model.cfg.qtypes))
        model.temperature.copy_(torch.tensor(temps, dtype=model.temperature.dtype, device=model.temperature.device))
        print(json.dumps({"fitted_temperatures": temps, "on": args.calibrate_on, "rows": len(cal_rows)}), flush=True)
        if args.save_calibration:
            (Path(args.model) / "calibration.json").write_text(json.dumps({"temperature": temps}, indent=2) + "\n")
    if args.rows:
        rows = list(read_jsonl(args.rows, args.max_rows))
    elif args.bench:
        rows = [r for r in CONVERTERS[args.bench](limit=args.max_rows) if r["split"] == "test"]
    else:
        raise SystemExit("pass --rows or --bench")
    records, raw = evaluate(model, rows, ser, device, args.batch_size, None)
    temps = model.temperature.tolist()
    report = {
        "model": args.model,
        "rows": len(rows),
        "raw": raw,
        "calibrated": evaluate_records(records, temps).to_dict(),
        "coverage": coverage_table(records, temps),
        "per_source": per_source(rows, records, temps),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
