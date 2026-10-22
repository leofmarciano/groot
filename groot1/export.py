"""Export: ONNX graphs for the production path and the Bend weight file.

    groot-export onnx  --model runs/groot-1-s/final --out export/onnx
    groot-export bend  --model runs/groot-1-s/final --out export/groot.gbw [--prune-vocab 65536 --corpus data/train.jsonl]
    groot-export prune --model ... --out ... --keep 65536 --corpus data/train.jsonl

Bend weight file (.gbw):
    magic "GBW1" | u32 header_len | header JSON (utf-8) | zero padding to 64 B | float32 LE tensors
    header = {"config": {...}, "tensors": [{"name", "shape", "offset", "numel"}], "vocab_remap": optional}
The Bend engine (bend/) maps this file and reads each tensor as a flat Array<F32>.
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .data import read_jsonl
from .model import GrootDecisionModel
from .schema import render_content

MAGIC = b"GBW1"


# ------------------------------------------------------------------- bend
def write_gbw(model: GrootDecisionModel, path: str | Path, vocab_remap: np.ndarray | None = None, keep_ids: np.ndarray | None = None) -> dict:
    tensors = []
    blobs = []
    offset = 0
    state = model.state_dict()
    for name, t in state.items():
        arr = t.detach().cpu().float().numpy()
        if keep_ids is not None and name.endswith("embeddings.tok_embeddings.weight"):
            arr = arr[keep_ids]
        arr = np.ascontiguousarray(arr, dtype="<f4")
        tensors.append({"name": name, "shape": list(arr.shape), "offset": offset, "numel": int(arr.size)})
        blobs.append(arr.tobytes())
        offset += arr.size * 4
    header = {
        "format": "gbw",
        "version": 1,
        "config": model.cfg.to_dict(),
        "encoder_config": {k: v for k, v in model.encoder.config.to_dict().items() if isinstance(v, (int, float, str, bool))},
        "tensors": tensors,
        "temperature": model.temperature.tolist(),
        "layer_types": list(getattr(model.encoder.config, "layer_types", None) or []),
        "vocab_remap": vocab_remap.tolist() if vocab_remap is not None else None,
    }
    hbytes = json.dumps(header).encode("utf-8")
    pad = (-(4 + 4 + len(hbytes))) % 64
    with Path(path).open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(hbytes)))
        f.write(hbytes)
        f.write(b"\0" * pad)
        for b in blobs:
            f.write(b)
    return {"tensors": len(tensors), "bytes": 8 + len(hbytes) + pad + offset, "path": str(path)}


def read_gbw_header(path: str | Path) -> dict:
    with Path(path).open("rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError("not a GBW file")
        n = struct.unpack("<I", f.read(4))[0]
        return json.loads(f.read(n).decode("utf-8"))


# ----------------------------------------------------------- vocab pruning
def token_frequencies(tokenizer, corpus: str | Path, limit: int | None = 200_000) -> Counter:
    counts: Counter = Counter()
    for i, row in enumerate(read_jsonl(corpus)):
        if limit and i >= limit:
            break
        texts = [render_content(row["state"]), row.get("question") or "", *row.get("options", [])]
        for t in texts:
            counts.update(tokenizer(t, add_special_tokens=False)["input_ids"])
    return counts


def prune_plan(tokenizer, counts: Counter, keep: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (keep_ids sorted, remap[full_vocab] -> compact id or UNK slot)."""
    specials = [i for i in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id) if i is not None]
    chosen = set(specials)
    for tid, _ in counts.most_common():
        if len(chosen) >= keep:
            break
        chosen.add(tid)
    keep_ids = np.array(sorted(chosen), dtype=np.int64)
    unk = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else specials[0]
    remap = np.full(len(tokenizer), -1, dtype=np.int64)
    remap[keep_ids] = np.arange(len(keep_ids))
    remap[remap < 0] = remap[unk]
    return keep_ids, remap


# ------------------------------------------------------------------- onnx
class _EncoderWrapper(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, input_ids, attention_mask, position_ids):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids).last_hidden_state


class _FusionWrapper(torch.nn.Module):
    """Option set [q_token ; markers] + state memory -> option logits (fusion + scorer)."""

    def __init__(self, model: GrootDecisionModel):
        super().__init__()
        self.m = model

    def forward(self, S, S_mask, S_pos, memory, memory_mask):
        from .model import rope_cos_sin

        m = self.m
        cos, sin = rope_cos_sin(S_pos, m.cfg.hidden_size // m.cfg.fusion_heads, m.cfg.rope_theta, S.dtype)
        for layer in m.fusion:
            S = layer(S, cos, sin, S_mask.bool(), memory, memory_mask.bool())
        S = m.final_norm(S)
        return m.scorer(S[:, 1:]).squeeze(-1)


def export_onnx(model: GrootDecisionModel, out_dir: str | Path, opset: int = 17) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    d = model.cfg.hidden_size
    ids = torch.randint(5, 100, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)
    pos = torch.arange(16)[None].repeat(2, 1)
    torch.onnx.export(
        _EncoderWrapper(model.encoder), (ids, mask, pos), str(out / "encoder.onnx"), opset_version=opset,
        input_names=["input_ids", "attention_mask", "position_ids"], output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "b", 1: "l"}, "attention_mask": {0: "b", 1: "l"}, "position_ids": {0: "b", 1: "l"}, "last_hidden_state": {0: "b", 1: "l"}},
        dynamo=False,
    )
    S = torch.randn(2, 5, d)
    S_mask = torch.ones(2, 5, dtype=torch.long)
    S_pos = torch.ones(2, 5, dtype=torch.long)
    S_pos[:, 0] = 0
    memory = torch.randn(2, 40, d)
    mmask = torch.ones(2, 40, dtype=torch.long)
    torch.onnx.export(
        _FusionWrapper(model), (S, S_mask, S_pos, memory, mmask), str(out / "fusion.onnx"), opset_version=opset,
        input_names=["set", "set_mask", "set_pos", "memory", "memory_mask"], output_names=["logits"],
        dynamic_axes={"set": {0: "q", 1: "k1"}, "set_mask": {0: "q", 1: "k1"}, "set_pos": {0: "q", 1: "k1"}, "memory": {0: "q", 1: "m"}, "memory_mask": {0: "q", 1: "m"}, "logits": {0: "q", 1: "k"}},
        dynamo=False,
    )
    misc = {k: v.detach().cpu().numpy().tolist() for k, v in model.state_dict().items() if k.split(".")[0] in ("tree", "seg_emb", "qtype_emb", "level_emb", "conf_head")}
    (out / "tree_and_misc.json").write_text(json.dumps({"weights": misc, "temperature": model.temperature.tolist(), "config": model.cfg.to_dict()}))
    return {"encoder": str(out / "encoder.onnx"), "fusion": str(out / "fusion.onnx")}


# -------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="groot-export")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("onnx")
    o.add_argument("--model", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--opset", type=int, default=17)
    b = sub.add_parser("bend")
    b.add_argument("--model", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--prune-vocab", type=int, default=None)
    b.add_argument("--corpus", default=None)
    args = ap.parse_args(argv)
    model = GrootDecisionModel.from_pretrained(args.model, device="cpu")
    if args.cmd == "onnx":
        print(json.dumps(export_onnx(model, args.out, args.opset), indent=2))
    else:
        keep_ids = remap = None
        if args.prune_vocab:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(Path(args.model) / "tokenizer")
            if not args.corpus:
                raise SystemExit("--prune-vocab needs --corpus")
            keep_ids, remap = prune_plan(tok, token_frequencies(tok, args.corpus), args.prune_vocab)
        print(json.dumps(write_gbw(model, args.out, remap, keep_ids), indent=2))


if __name__ == "__main__":
    main()
