"""Generate the Bend parity fixture under tests/fixtures/parity/.

A deterministic, randomly-initialised tiny Groot model plus one request blob
and the PyTorch reference outputs. `bend/groot/parity.mjs` replays the request
through the Bend engine and compares.

    python scripts/make_parity_fixture.py [out_dir]

Produces:
    ckpt/           save_pretrained tiny model (groot_config.json, encoder/, model.safetensors)
    model.gbw       flat Bend weight file (groot1.export.write_gbw)
    req.bin         u32 request blob (groot1.bendio.RequestBlob)
    ref.npy         float32 [probs(nopt) ; conf_logits(nq) ; logits(nopt)]
    meta.json       {data_offset, n_chunks, n_questions, n_options}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from groot1.bendio import RequestBlob, load_gbw, data_offset
from groot1.config import GrootConfig
from groot1.export import write_gbw
from groot1.model import GrootDecisionModel, collate
from groot1.serialize import EncodedQuestion, EncodedState

CLS, SEP, MASK, PAD = 2, 1, 4, 0


def make_question(qtype: int, head: list[int], state_head: list[int], options: list[list[int]]) -> EncodedQuestion:
    """Same layout as groot1.serialize.Serializer._question (shared option base, option blocks)."""
    ids = [CLS, *head, SEP] + ([*state_head, SEP] if state_head else [])
    pos = list(range(len(ids)))
    blocks = [0] * len(ids)
    base = len(ids)
    markers = []
    for k, opt in enumerate(options):
        markers.append(len(ids))
        ids += [MASK, *opt]
        blocks += [k + 1] * (1 + len(opt))
        pos += list(range(base, base + 1 + len(opt)))
    return EncodedQuestion(name="q", qtype=qtype, ids=ids, position_ids=pos, markers=markers, blocks=blocks)


def build_request(rng: np.random.Generator, vocab: int, early_state_tokens: int = 8):
    def ids(n):
        return rng.integers(5, vocab, size=n).tolist()

    chunks = [[CLS, *ids(n), SEP] for n in (10, 12)]
    state_head = [t for c in chunks for t in c[1:-1]][:early_state_tokens]
    # one of each primitive, options of different lengths
    questions = []
    for qtype, k in ((0, 3), (1, 4), (2, 2)):
        questions.append(make_question(qtype, ids(5), state_head, [ids(2 + (j % 3)) for j in range(k)]))
    return chunks, questions


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/parity")
    ckpt = out / "ckpt"
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = GrootConfig(
        name="groot-parity",
        hidden_size=32,
        chunk_tokens=32,
        max_chunks=8,
        tree_levels=4,
        fusion_layers=2,
        fusion_heads=4,
        fusion_intermediate=64,
        conf_hidden=256,
        max_question_tokens=64,
        max_option_tokens=48,
        early_state_tokens=8,
    )
    model = GrootDecisionModel.tiny(cfg, vocab_size=1024, layers=2).eval()
    model.save_pretrained(ckpt)
    write_gbw(model, out / "model.gbw")

    chunks, questions = build_request(rng, 1024)
    blob = RequestBlob.build(chunks, questions)
    RequestBlob.write(blob, out / "req.bin")

    # ---- torch reference over the identical token structures
    state = EncodedState(chunks=chunks, tokens=sum(len(c) - 2 for c in chunks))
    eqs = questions
    batch = collate([state], eqs, PAD, q_state=[0] * len(eqs))
    with torch.no_grad():
        res = model.forward(batch)
    nopt = sum(q.n_options for q in questions)
    nq = len(questions)
    ref = np.zeros(nopt + nq + nopt, dtype="<f4")
    o0 = 0
    probs, conf, logits = res["probabilities"].numpy(), res["confidence_logit"].numpy(), res["logits"].numpy()
    for qi, q in enumerate(questions):
        k = q.n_options
        ref[o0 : o0 + k] = probs[qi, :k]
        ref[nopt + qi] = conf[qi]
        ref[nopt + nq + o0 : nopt + nq + o0 + k] = logits[qi, :k]
        o0 += k
    np.save(out / "ref.npy", ref)

    meta = {
        "data_offset": data_offset(out / "model.gbw"),
        "n_chunks": len(chunks),
        "n_questions": nq,
        "n_options": nopt,
        "ref_len": int(ref.size),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
