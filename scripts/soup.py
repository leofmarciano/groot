"""Weight-space interpolation ("model soup") of Groot checkpoints with the same architecture.

    python scripts/soup.py --out runs/soup --models A B [C ...] [--weights 0.5 0.5]

Floating-point tensors are averaged with the given weights; the tokenizer, configs and
calibration are copied from the first model. Recalibrate afterwards with
`groot-eval --calibrate-on val.jsonl --save-calibration`.
"""
import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    w = args.weights or [1.0 / len(args.models)] * len(args.models)
    if len(w) != len(args.models) or abs(sum(w) - 1) > 1e-6:
        raise SystemExit("weights must match models and sum to 1")
    states = [load_file(str(Path(m) / "model.safetensors")) for m in args.models]
    keys = states[0].keys()
    if any(s.keys() != keys for s in states):
        raise SystemExit("checkpoints have different tensors")
    out = {}
    for k in keys:
        t = states[0][k]
        out[k] = sum(wi * s[k].float() for wi, s in zip(w, states)).to(t.dtype) if t.is_floating_point() else t
    dst = Path(args.out)
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    shutil.copytree(args.models[0], dst, ignore=shutil.ignore_patterns("model.safetensors", "optimizer.pt"))
    save_file({k: v.contiguous() for k, v in out.items()}, str(dst / "model.safetensors"), metadata={"format": "pt", "family": "groot", "soup": str(w)})
    print(f"soup of {len(args.models)} models (weights {w}) -> {dst}")


if __name__ == "__main__":
    main()
