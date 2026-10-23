"""System One compatible server for Groot 1.

    groot-serve --model runs/groot-1-s/final --host 0.0.0.0 --port 8000

Endpoints (same shapes as TypeSafe's API, see schema.py):
    POST /v1/systemone   -> {model, answers{name: ChoiceAnswer|ScoreAnswer|NoulAnswer}, usage}
    GET  /v1/models      -> {models: [ModelMetadata]}
    GET  /healthz

Request handling: the state is encoded once; every option of every question
is encoded pointwise in one batched encoder pass; the questions are then
decided in one batched fusion pass. The `Groot-Extended: true` header adds an
`x_groot` field with the correctness estimate per question.

A checkpoint directory holding `ensemble.json` ({"members": [dir, ...], "weights": [...]},
member paths relative to it) is served as a probability-averaged ensemble of its members.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .model import GrootDecisionModel, collate
from .schema import (
    ChoiceQuestion,
    ModelMetadata,
    Question,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    build_answer,
)
from .serialize import Serializer


class GrootEngine:
    """Thread-safe inference wrapper used by the server, the CLI and the benchmarks."""

    def __init__(self, checkpoint: str | Path, device: str | None = None, model_name: str | None = None, overrides: dict | None = None):
        from transformers import AutoTokenizer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        spec = Path(checkpoint) / "ensemble.json"
        if spec.exists():
            ens = json.loads(spec.read_text())
            self.members = [GrootEngine(Path(checkpoint) / m, device, None, overrides) for m in ens["members"]]
            w = ens.get("weights") or [1.0] * len(self.members)
            self.weights = [x / sum(w) for x in w]
            self.cfg = self.members[0].cfg
            self.name = model_name or ens.get("name") or self.cfg.name
            self.aliases = {"groot-latest": self.name, self.name: self.name, "jev-latest": self.name}
            return
        self.members = []
        self.model = GrootDecisionModel.from_pretrained(checkpoint, device=self.device)
        self.tok = AutoTokenizer.from_pretrained(Path(checkpoint) / "tokenizer")
        if overrides:
            from .config import GrootConfig

            self.model.cfg = GrootConfig.from_dict({**self.model.cfg.to_dict(), **overrides})
        self.cfg = self.model.cfg
        self.ser = Serializer(self.tok, self.cfg)
        self.name = model_name or self.cfg.name
        self.aliases = {"groot-latest": self.name, self.name: self.name, "jev-latest": self.name}

    # -------------------------------------------------------------- core
    @torch.inference_mode()
    def forward(self, state: Any, questions: dict[str, Question]) -> tuple[list[list[float]], list[dict], Usage]:
        """Per-question option probabilities, extras and usage (averaged over members for an ensemble)."""
        if self.members:
            outs = [m.forward(state, questions) for m in self.members]
            probs = [[sum(w * o[0][qi][j] for w, o in zip(self.weights, outs)) for j in range(len(outs[0][0][qi]))] for qi in range(len(questions))]
            extra = [{**outs[0][1][qi], "p_correct": sum(w * o[1][qi]["p_correct"] for w, o in zip(self.weights, outs))} for qi in range(len(questions))]
            return probs, extra, outs[0][2]
        enc = self.ser.encode_request(state, questions)
        batch = collate([enc.state], enc.questions, self.ser.pad, [0] * len(enc.questions)).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.model(batch)
        probs = out["probabilities"].float().cpu()
        conf = torch.sigmoid(out["confidence_logit"].float()).cpu().tolist()
        ks = [q.n_options for q in enc.questions]
        extra = [{"p_correct": float(conf[qi]), "truncated": enc.questions[qi].truncated} for qi in range(len(ks))]
        return [probs[qi, :k].tolist() for qi, k in enumerate(ks)], extra, Usage(input_tokens=enc.input_tokens, output_tokens=enc.output_tokens)

    def answer(self, state: Any, questions: dict[str, Question], extended: bool = False) -> SystemOneResponse:
        probs, extra, usage = self.forward(state, questions)
        answers = {name: build_answer(question, probs[qi]) for qi, (name, question) in enumerate(questions.items())}
        x = {name: extra[qi] for qi, name in enumerate(questions)}
        return SystemOneResponse(model=self.name, answers=answers, usage=usage, x_groot=x if extended else None)

    def metadata(self) -> list[ModelMetadata]:
        return [
            ModelMetadata(name=self.name, description=f"Groot 1 typed-decision model ({self.cfg.name}) by {self.cfg.organization}", release_date=time.strftime("%Y-%m-%d")),
            ModelMetadata(name="groot-latest", description=f"Alias of {self.name}", release_date=time.strftime("%Y-%m-%d")),
        ]


def build_app(engine: GrootEngine):
    from fastapi import FastAPI, Header, HTTPException, Request

    app = FastAPI(title="Groot 1 System One API", version=__version__)

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model": engine.name}

    @app.get("/v1/models")
    def models():
        return {"models": [m.model_dump() for m in engine.metadata()]}

    @app.post("/v1/systemone")
    async def systemone(request: Request, groot_extended: str | None = Header(default=None)):
        try:
            body = await request.json()
            req = SystemOneRequest(**body)
        except Exception as exc:  # 400 like TypeSafe for malformed bodies
            raise HTTPException(status_code=400, detail=str(exc))
        if req.model not in engine.aliases:
            raise HTTPException(status_code=404, detail=f"unknown model {req.model!r}")
        t0 = time.perf_counter()
        resp = engine.answer(req.state, req.questions, extended=(groot_extended or "").lower() == "true")
        payload = resp.model_dump(exclude_none=True)
        payload["usage"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return payload

    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="groot-serve")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--name", default=None)
    ap.add_argument("--once", default=None, help="answer one JSON request file and exit (no server)")
    args = ap.parse_args(argv)
    engine = GrootEngine(args.model, args.device, args.name)
    if args.once:
        body = json.loads(Path(args.once).read_text())
        req = SystemOneRequest(**body)
        print(json.dumps(engine.answer(req.state, req.questions, extended=True).model_dump(exclude_none=True), indent=2, ensure_ascii=False))
        return
    import uvicorn

    uvicorn.run(build_app(engine), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
