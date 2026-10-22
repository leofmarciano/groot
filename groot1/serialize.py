"""Serialization of System One requests into Groot token structures.

A request becomes:

* state chunks  — the rendered state is split into sentence/paragraph units,
                  tokenized, and greedily packed into leaves of at most
                  `chunk_tokens` tokens: `[CLS] ...unit tokens... [SEP]`.
                  Leaves are encoded in parallel and merged by the tree reducer.
* question sequence — one encoder sequence per question (early fusion):
                  `[CLS] "{type} question: {instructions}" [SEP] <first
                  early_state_tokens of the state> [SEP] [MASK] option_1 ...
                  [MASK] option_K`. Attention is block-restricted: the prefix
                  sees only the prefix, each option sees the prefix and itself
                  (never the other options). Every option block starts at the
                  SAME RoPE position, so the encoder is exactly
                  permutation-equivariant over options (list order cannot leak
                  into the answer, a documented Jev weakness) while every option
                  attends to the question and the state in all layers. The `[MASK]` state is the option's score slot. The full
                  state (any length) is still read through the tree memory and the
                  fusion layers.

Score levels and noul outcomes carry a level embedding in the model so that
ordinal roles stay distinguishable; choice options carry none.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .config import QTYPE_IDS, GrootConfig
from .schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, render_content

_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


@dataclass
class EncodedQuestion:
    name: str
    qtype: int
    ids: list[int]  # one encoder sequence: head + state head + option blocks
    position_ids: list[int]
    markers: list[int]  # position in `ids` of each option's [MASK]
    blocks: list[int] = None  # 0 = question/state prefix, k+1 = option k (attention is block-restricted)
    tokens: int = 0
    truncated: bool = False

    @property
    def n_options(self) -> int:
        return len(self.markers)


@dataclass
class EncodedState:
    chunks: list[list[int]]
    tokens: int
    truncated: bool = False
    pieces: list[list[int]] | None = None  # retrievable pieces of the state, in order


@dataclass
class EncodedRequest:
    state: EncodedState
    questions: list[EncodedQuestion]

    @property
    def input_tokens(self) -> int:
        return self.state.tokens + sum(q.tokens for q in self.questions)

    @property
    def output_tokens(self) -> int:
        return sum(q.n_options for q in self.questions)


class Serializer:
    """Turns requests into token structures using the backbone tokenizer."""

    def __init__(self, tokenizer: Any, config: GrootConfig):
        self.tok = tokenizer
        self.cfg = config
        for name in ("cls_token_id", "sep_token_id", "mask_token_id", "pad_token_id"):
            if getattr(tokenizer, name, None) is None:
                raise ValueError(f"tokenizer must define {name}")
        self.cls = tokenizer.cls_token_id
        self.sep = tokenizer.sep_token_id
        self.mask = tokenizer.mask_token_id
        self.pad = tokenizer.pad_token_id
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------- helpers
    def encode(self, text: str) -> list[int]:
        ids = self._cache.get(text)
        if ids is None:
            clean = text.replace(self.tok.mask_token, " ") if self.tok.mask_token else text
            ids = self.tok(clean, add_special_tokens=False)["input_ids"]
            if len(self._cache) < 65536:
                self._cache[text] = ids
        return list(ids)

    # --------------------------------------------------------------- state
    def units(self, text: str) -> list[str]:
        parts = [p.strip() for p in _SENTENCE_RE.split(text) if p and p.strip()]
        return parts or [text.strip() or " "]

    def encode_state(self, state: Any) -> EncodedState:
        text = render_content(state)
        body = self.cfg.chunk_tokens - 2
        chunks: list[list[int]] = []
        current: list[int] = []
        total = 0
        for unit in self.units(text):
            ids = self.encode(unit)
            if not ids:
                continue
            pieces = [ids[i : i + body] for i in range(0, len(ids), body)]  # oversized unit: hard split
            for piece in pieces:
                if current and len(current) + len(piece) > body:
                    chunks.append(current)
                    current = []
                current = current + piece
                total += len(piece)
        if current or not chunks:
            chunks.append(current)
        truncated = False
        if len(chunks) > self.cfg.max_chunks:
            chunks = chunks[: self.cfg.max_chunks]
            truncated = True
        wrapped = [[self.cls] + c + [self.sep] for c in chunks]
        pieces = None
        if self.cfg.state_retrieval and total > self.cfg.early_state_tokens:
            k = self.cfg.retrieval_piece_tokens
            pieces = [c[i : i + k] for c in chunks for i in range(0, len(c), k)]
        return EncodedState(chunks=wrapped, tokens=total, truncated=truncated, pieces=pieces)

    # ----------------------------------------------------------- questions
    def state_head(self, state: EncodedState, query: Sequence[int] | None = None) -> list[int]:
        """State tokens placed inside the question sequence.

        Short states (or retrieval off): the first `early_state_tokens` tokens, in order.
        Long states with retrieval on: the first `retrieval_lead_tokens` tokens plus the pieces
        with the highest BM25 score against the question and option tokens, kept in state order.
        """
        n = self.cfg.early_state_tokens
        if state.pieces is None or not query:
            out: list[int] = []
            for c in state.chunks:
                if len(out) >= n:
                    break
                out += c[1:-1]
            return out[:n]
        return self._retrieve(state.pieces, query, n)

    def _retrieve(self, pieces: list[list[int]], query: Sequence[int], budget: int, k1: float = 1.2, b: float = 0.75) -> list[int]:
        import math
        from collections import Counter

        special = {self.cls, self.sep, self.mask, self.pad}
        q = {t for t in query if t not in special}
        N = len(pieces)
        tfs = [Counter(p) for p in pieces]
        df = Counter(t for tf in tfs for t in tf.keys() & q)
        avg = sum(len(p) for p in pieces) / max(1, N)
        idf = {t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in q if df[t]}
        scores = []
        for i, (p, tf) in enumerate(zip(pieces, tfs)):
            s = 0.0
            for t, w in idf.items():
                f = tf.get(t, 0)
                if f:
                    s += w * f * (k1 + 1) / (f + k1 * (1 - b + b * len(p) / avg))
            scores.append(s)
        lead, chosen, used = [], set(), 0
        for i, p in enumerate(pieces):  # always keep the beginning of the state
            if used + len(p) > self.cfg.retrieval_lead_tokens:
                break
            chosen.add(i)
            used += len(p)
        for i in sorted(range(N), key=lambda j: -scores[j]):
            if used >= budget:
                break
            if i in chosen or scores[i] <= 0:
                continue
            if used + len(pieces[i]) > budget:
                continue
            chosen.add(i)
            used += len(pieces[i])
        out: list[int] = []
        for i in sorted(chosen):
            out += pieces[i]
        return out[:budget]

    def _question(self, name: str, qtype: str, instructions: str, option_texts: Sequence[str], head_state: list[int]) -> EncodedQuestion:
        head = self.encode(f"{qtype} question: {instructions}".strip())
        truncated = False
        hlimit = max(16, self.cfg.max_question_tokens - 4)
        if len(head) > hlimit:
            head, truncated = head[:hlimit], True
        ids = [self.cls] + head + [self.sep]
        if head_state:
            ids += head_state + [self.sep]
        pos = list(range(len(ids)))
        blocks = [0] * len(ids)
        base = len(ids)
        markers: list[int] = []
        limit = self.cfg.max_option_tokens
        for t in option_texts:
            opt = self.encode(" " + t)
            if len(opt) > limit:
                opt, truncated = opt[:limit], True
            markers.append(len(ids))
            ids += [self.mask] + opt
            blocks += [len(markers)] * (1 + len(opt))
            if self.cfg.option_shared_positions:
                pos += list(range(base, base + 1 + len(opt)))
            else:
                start = pos[-1] + 1
                pos += list(range(start, start + 1 + len(opt)))
        tokens = len(head) + sum(len(self.encode(" " + t)[:limit]) for t in option_texts)
        return EncodedQuestion(name=name, qtype=QTYPE_IDS[qtype], ids=ids, position_ids=pos, markers=markers, blocks=blocks,
                               tokens=tokens, truncated=truncated)

    def _query(self, instructions: str, option_texts: Sequence[str]) -> list[int]:
        ids = list(self.encode(instructions))
        for t in option_texts:
            ids += self.encode(" " + t)
        return ids

    def encode_question(self, name: str, question: Question, state: EncodedState | None = None) -> EncodedQuestion:
        instr = render_content(question.instructions)
        head_state = self.state_head(state, self._query(instr, question.option_texts)) if state is not None else []
        return self._question(name, question.type, render_content(question.instructions), question.option_texts, head_state)

    def encode_request(self, state: Any, questions: dict[str, Question]) -> EncodedRequest:
        enc_state = self.encode_state(state)
        return EncodedRequest(
            state=enc_state,
            questions=[self.encode_question(name, q, enc_state) for name, q in questions.items()],
        )

    # --------------------------------------------------------- training rows
    def encode_row(self, row: dict[str, Any], state: EncodedState | None = None) -> tuple[EncodedState, EncodedQuestion]:
        """Encode a flat training row: {state, question, type, options[, target]}.

        Options are already rendered texts (`name: description` for choice,
        `level i: text` for score, `[false, true]` descriptions for noul).
        """
        st = state if state is not None else self.encode_state(row["state"])
        query = self._query(row.get("question") or "", row["options"])
        q = self._question(str(row.get("id", "row")), row.get("type", "choice"), row.get("question") or "", row["options"], self.state_head(st, query))
        return st, q


def question_from_row(row: dict[str, Any]) -> Question:
    """Build a schema question from a flat row (used by evaluation tooling)."""
    qtype = row.get("type", "choice")
    if qtype == "choice":
        return ChoiceQuestion(instructions=row.get("question"), criteria={str(o): None for o in row["options"]})
    if qtype == "score":
        return ScoreQuestion(instructions=row.get("question"), criteria=list(row["options"]))
    return NoulQuestion(instructions=row.get("question") or "true")


def flatten_question(name: str, question: Question, state: Any) -> dict[str, Any]:
    """Inverse of `question_from_row` for logging and synthetic data."""
    return {
        "name": name,
        "state": state,
        "question": render_content(question.instructions),
        "type": question.type,
        "options": question.option_texts,
    }


def iter_option_texts(questions: Iterable[Question]) -> Iterable[str]:
    for q in questions:
        yield from q.option_texts
