"""Request/response schema and confidence formulas.

The schema mirrors TypeSafe's System One contract (OpenAPI 0.2.0, observed
Oct 2026) so Groot 1 is a drop-in replacement for `POST /v1/systemone`:

* `choice`  — 1..255 named criteria, returns choice + confidence + probabilities
* `score`   — 1..10 ordered levels, returns expected score + confidence + legend
* `noul`    — yes/no statement, returns the probability of "true"

Confidence formulas are the ones TypeSafe documents (docs.typesafe.ai/confidence):

* choice: (p_max - 1/n) / (1 - 1/n)
* score : max(0, 1 - sum_i p_i |i - m| / MAD_unif),  MAD_unif = mean_i |i - (n-1)/2|
* noul  : no confidence field; |2p - 1| is the suggested equivalent.
"""
from __future__ import annotations

import json
import math
from typing import Any, Literal, Mapping, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Content = Union[str, dict, list, None]


# ----------------------------------------------------------------- rendering
def render_content(value: Content, *, indent: int = 0) -> str:
    """Render a state / instruction / criterion value as model-readable text.

    JSON objects are rendered as indented `key: value` lines (key paths help
    the model with indirection questions); arrays become numbered items.
    Strings pass through unchanged.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    pad = "  " * indent
    if isinstance(value, Mapping):
        lines = []
        for key, item in value.items():
            if isinstance(item, (Mapping, list)) and item:
                lines.append(f"{pad}{key}:")
                lines.append(render_content(item, indent=indent + 1))
            else:
                lines.append(f"{pad}{key}: {render_content(item)}")
        return "\n".join(lines)
    if isinstance(value, Sequence):
        lines = []
        for i, item in enumerate(value):
            if isinstance(item, (Mapping, list)) and item:
                lines.append(f"{pad}[{i}]")
                lines.append(render_content(item, indent=indent + 1))
            else:
                lines.append(f"{pad}[{i}] {render_content(item)}")
        return "\n".join(lines)
    return json.dumps(value, ensure_ascii=False)


# ----------------------------------------------------------------- questions
class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice"] = "choice"
    instructions: Content = None
    criteria: dict[str, Content] = Field(..., min_length=1, max_length=255)

    @property
    def option_names(self) -> list[str]:
        return list(self.criteria)

    @property
    def option_texts(self) -> list[str]:
        out = []
        for name, desc in self.criteria.items():
            text = render_content(desc)
            out.append(f"{name}: {text}" if text else name)
        return out


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["score"] = "score"
    instructions: Content = None
    criteria: list[Content] = Field(..., min_length=1, max_length=10)

    @property
    def option_names(self) -> list[str]:
        return [str(i) for i in range(len(self.criteria))]

    @property
    def option_texts(self) -> list[str]:
        return [f"level {i}: {render_content(c)}" for i, c in enumerate(self.criteria)]


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")
    true: Content = None
    false: Content = None


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["noul"] = "noul"
    instructions: Content = None
    criteria: NoulCriteria | None = None

    @model_validator(mode="after")
    def _needs_text(self) -> "NoulQuestion":
        has_criteria = self.criteria is not None and (
            self.criteria.true is not None or self.criteria.false is not None
        )
        if self.instructions is None and not has_criteria:
            raise ValueError("noul requires instructions or a described outcome")
        return self

    @property
    def option_names(self) -> list[str]:
        return ["false", "true"]

    @property
    def option_texts(self) -> list[str]:
        f = render_content(self.criteria.false) if self.criteria else ""
        t = render_content(self.criteria.true) if self.criteria else ""
        return [f"false: {f}" if f else "false", f"true: {t}" if t else "true"]


Question = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion]


def parse_question(raw: Mapping[str, Any]) -> Question:
    kind = raw.get("type")
    if kind == "choice":
        return ChoiceQuestion(**raw)
    if kind == "score":
        return ScoreQuestion(**raw)
    if kind == "noul":
        return NoulQuestion(**raw)
    raise ValueError(f"unsupported question type {kind!r} (box/images are not supported by Groot 1)")


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "groot-latest"
    state: Union[str, dict, list]
    questions: dict[str, Any] = Field(..., min_length=1)
    assets: dict[str, Any] | None = None

    @field_validator("questions")
    @classmethod
    def _parse(cls, value: dict[str, Any]) -> dict[str, Question]:
        parsed: dict[str, Question] = {}
        for name, q in value.items():
            if not isinstance(name, str) or not name:
                raise ValueError("question names must be non-empty strings")
            parsed[name] = q if isinstance(q, (ChoiceQuestion, ScoreQuestion, NoulQuestion)) else parse_question(q)
        return parsed

    @model_validator(mode="after")
    def _no_assets(self) -> "SystemOneRequest":
        if self.assets:
            raise ValueError("assets/images are not supported by Groot 1")
        return self


# ------------------------------------------------------------------ answers
class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, Content]
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]]
    usage: Usage
    x_groot: dict[str, Any] | None = None  # optional extension (correctness estimate etc.)


class ModelMetadata(BaseModel):
    name: str
    description: str
    release_date: str  # YYYY-MM-DD


# -------------------------------------------------------------- confidence
def confidence_choice(probabilities: Sequence[float]) -> float:
    n = len(probabilities)
    if n <= 1:
        return 1.0
    p_max = max(probabilities)
    return max(0.0, min(1.0, (p_max - 1.0 / n) / (1.0 - 1.0 / n)))


def confidence_score(probabilities: Sequence[float]) -> float:
    n = len(probabilities)
    if n <= 1:
        return 1.0
    m = max(range(n), key=probabilities.__getitem__)
    spread = sum(p * abs(i - m) for i, p in enumerate(probabilities))
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, min(1.0, 1.0 - spread / even))


def confidence_noul(p_true: float) -> float:
    return abs(2.0 * p_true - 1.0)


def expected_score(probabilities: Sequence[float]) -> float:
    return float(sum(i * p for i, p in enumerate(probabilities)))


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not logits:
        return []
    scaled = [x / temperature for x in logits]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def build_answer(question: Question, probabilities: Sequence[float]) -> ChoiceAnswer | ScoreAnswer | NoulAnswer:
    names = question.option_names
    if len(names) != len(probabilities):
        raise ValueError("probability count must match option count")
    probs = {name: float(p) for name, p in zip(names, probabilities)}
    if isinstance(question, ChoiceQuestion):
        best = max(range(len(names)), key=lambda i: probabilities[i])
        return ChoiceAnswer(choice=names[best], confidence=confidence_choice(probabilities), probabilities=probs)
    if isinstance(question, ScoreQuestion):
        legend = {str(i): c for i, c in enumerate(question.criteria)}
        return ScoreAnswer(
            score=expected_score(probabilities),
            confidence=confidence_score(probabilities),
            legend=legend,
            probabilities=probs,
        )
    return NoulAnswer(noul=float(probabilities[1]))
