"""Datasets: conversion of public corpora and synthetic generation into decision rows.

Decision row (JSONL, one per line):

    {"id": "agnews:123", "source": "agnews", "lang": "en",
     "state": "<text or JSON object>",
     "question": "What is the topic of this news article?",
     "type": "choice" | "score" | "noul",
     "options": ["World: ...", "Sports: ...", ...],      # rendered option texts
     "target": [0.0, 1.0, 0.0, 0.0],                      # distribution (one-hot for hard labels)
     "split": "train" | "val" | "test",
     "state_key": "<sha1 of the rendered state>"}         # lets one state serve many questions

Every public converter emits *rendered* options exactly as the serializer
renders live requests (`name: description`, `level i: text`, `false/true`),
so training and inference share one encoding.

CLI:
    groot-data build  --sources open_jev,agnews,emotion,banking77,massive,assin2,b2w --out data/
    groot-data inspect <hf_id> [--config c] [--split s]
    groot-data synth  --teacher openrouter/<model> --n 1000 --out data/synth.jsonl
    groot-data stats  data/train.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .schema import render_content

Row = dict[str, Any]


def state_key(state: Any) -> str:
    return hashlib.sha1(render_content(state).encode("utf-8")).hexdigest()[:16]


def one_hot(index: int, n: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(n)]


def make_row(
    source: str,
    idx: Any,
    state: Any,
    question: str,
    qtype: str,
    options: list[str],
    target: list[float],
    split: str,
    lang: str = "en",
) -> Row:
    if len(options) != len(target):
        raise ValueError("target length must match options")
    if qtype == "noul" and len(options) != 2:
        raise ValueError("noul rows need [false, true] options")
    total = float(sum(target))
    if total <= 0:
        raise ValueError("target must have mass")
    return {
        "id": f"{source}:{idx}",
        "source": source,
        "lang": lang,
        "state": state,
        "question": question,
        "type": qtype,
        "options": options,
        "target": [float(t) / total for t in target],
        "split": split,
        "state_key": state_key(state),
    }


def choice_options(names: list[str], descriptions: dict[str, str] | None = None) -> list[str]:
    out = []
    for n in names:
        d = (descriptions or {}).get(n)
        out.append(f"{n}: {d}" if d else n)
    return out


def score_options(levels: list[str]) -> list[str]:
    return [f"level {i}: {t}" for i, t in enumerate(levels)]


NOUL_OPTIONS = ["false", "true"]


# ------------------------------------------------------------- HF helpers
def _load(hf_id: str, config: str | None = None, split: str | None = None, streaming: bool = False, revision: str | None = None):
    from datasets import load_dataset

    return load_dataset(hf_id, config, split=split, streaming=streaming, revision=revision)


def _truthy(v) -> bool:
    """Labels arrive as bool, int or the strings 'True'/'False'/'1'/'0' depending on the mirror."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "sim")
    return bool(v)


def _text_label_dataset(source: str, hf_id: str, question: str, config: str | None = None, lang: str = "en", limit: int | None = None,
                        text_field: str = "text", label_field: str = "label_text",
                        splits=(("train", "train"), ("validation", "val"), ("test", "test"))) -> Iterator[Row]:
    """Datasets whose label is a string name (mteb mirrors). The label set is taken from the train split, sorted."""
    try:
        names = sorted(set(_load(hf_id, config, split="train")[label_field]))
    except Exception:
        return
    options = choice_options(names)
    index = {n: i for i, n in enumerate(names)}
    for hf_split, split in splits:
        try:
            ds = _load(hf_id, config, split=hf_split)
        except Exception:
            continue
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            if r[label_field] not in index:
                continue
            yield make_row(source, f"{hf_split}{i}", r[text_field], question, "choice", options, one_hot(index[r[label_field]], len(names)), split, lang)


def _split_name(i: int, val_frac: float = 0.02, test_frac: float = 0.02) -> str:
    r = (i * 2654435761) % 1000 / 1000.0  # deterministic pseudo-random
    if r < test_frac:
        return "test"
    if r < test_frac + val_frac:
        return "val"
    return "train"


# ------------------------------------------------------------- converters
def conv_open_jev(limit: int | None = None) -> Iterator[Row]:
    """ZefanCai/Open-Jev (CC0 generated records): soft typed-decision targets."""
    name_map = {"train": "train", "validation": "val", "test": "test", "calibration": "val"}
    for hf_split, split in name_map.items():
        ds = _load("ZefanCai/Open-Jev", "release-v2-redistributable", split=hf_split, streaming=True)
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            state = r.get("state_json")
            if isinstance(state, str):
                try:
                    state = json.loads(state)
                except json.JSONDecodeError:
                    pass
            kind = r["kind"]
            options = list(r["options"])
            if kind == "noul" and len(options) == 2:
                options = NOUL_OPTIONS if options[0].lower() in ("no", "false") else options
            try:
                yield make_row("open_jev", r.get("id", i), state, r["question"], kind, options, list(r["target"]), split)
            except ValueError:
                continue


def _questions_rows(source: str, hf_id: str, config: str | None, splits, limit: int | None = None, lang: str = "en") -> Iterator[Row]:
    """Rows of the form {state: json, questions: json{name: {type, instructions, criteria}}, gold|answers: json{name: {...}}}.

    Used by LocalLLaMA/typed-decisions (`gold`) and tasksource/procedural-typed-decisions (`answers`).
    Choice probabilities are keyed by option name, score probabilities by level index or level text,
    noul carries `noul` or probabilities {false, true}.
    """
    for hf_split, split in splits:
        try:
            ds = _load(hf_id, config, split=hf_split, streaming=True)
        except Exception:
            continue
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            try:
                state = r.get("state")
                state = json.loads(state) if isinstance(state, str) else state
                questions = r.get("questions")
                questions = json.loads(questions) if isinstance(questions, str) else questions
                gold = r.get("gold", r.get("answers"))
                gold = json.loads(gold) if isinstance(gold, str) else gold
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(questions, dict) or not isinstance(gold, dict):
                continue
            for name, q in questions.items():
                g = gold.get(name)
                if not isinstance(q, dict) or not isinstance(g, dict):
                    continue
                qtype = q.get("type") or g.get("type")
                instr = render_content(q.get("instructions"))
                probs = g.get("probabilities") or {}
                try:
                    if qtype == "choice":
                        crit = q.get("criteria") or {}
                        names = list(crit) if isinstance(crit, dict) else [str(c) for c in crit]
                        options = choice_options(names, crit if isinstance(crit, dict) else None)
                        target = [float(probs.get(n, 0.0)) for n in names]
                        if sum(target) <= 0 and g.get("choice") in names:
                            target = one_hot(names.index(g["choice"]), len(names))
                        if sum(target) <= 0 and g.get("label") in names:
                            target = one_hot(names.index(g["label"]), len(names))
                    elif qtype == "score":
                        crit = q.get("criteria") or []
                        levels = [render_content(c) for c in crit]
                        options = score_options(levels)
                        target = [float(probs.get(str(k), probs.get(levels[k], 0.0))) for k in range(len(levels))]
                    elif qtype == "noul":
                        p = g.get("noul")
                        if p is None:
                            p = probs.get("true")
                        if p is None:
                            continue
                        crit = q.get("criteria") if isinstance(q.get("criteria"), dict) else {}
                        f, t = render_content(crit.get("false")), render_content(crit.get("true"))
                        options = [f"false: {f}" if f else "false", f"true: {t}" if t else "true"]
                        target = [1.0 - float(p), float(p)]
                    else:
                        continue
                    yield make_row(source, f"{r.get('id', i)}/{name}", state, instr, qtype, options, target, split, lang)
                except (ValueError, TypeError, IndexError):
                    continue


def conv_typed_decisions(limit: int | None = None) -> Iterator[Row]:
    """LocalLLaMA/typed-decisions: 5 typed questions per case, soft gold (mean of teacher samples)."""
    return _questions_rows("typed_decisions", "LocalLLaMA/typed-decisions", "all", (("train", "train"), ("test", "test")), limit)


def _label_dataset(
    source: str,
    hf_id: str,
    text_field: str,
    label_field: str,
    question: str,
    descriptions: dict[str, str] | None,
    config: str | None = None,
    lang: str = "en",
    limit: int | None = None,
    label_names: list[str] | None = None,
    splits: tuple[tuple[str, str], ...] = (("train", "train"), ("validation", "val"), ("test", "test")),
) -> Iterator[Row]:
    for hf_split, split in splits:
        try:
            ds = _load(hf_id, config, split=hf_split)
        except Exception:
            continue
        names = label_names or ds.features[label_field].names
        options = choice_options(names, descriptions)
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            label = r[label_field]
            idx = label if isinstance(label, int) else names.index(label)
            yield make_row(source, f"{hf_split}{i}", r[text_field], question, "choice", options, one_hot(idx, len(names)), split, lang)


def conv_agnews(limit=None):
    return _label_dataset(
        "agnews", "fancyzhx/ag_news", "text", "label", "What is the topic of this news article?",
        {"World": "international and political news", "Sports": "sports", "Business": "business and finance", "Sci/Tech": "science and technology"},
        limit=limit, splits=(("train", "train"), ("test", "test")),
    )


def conv_emotion(limit=None):
    return _label_dataset(
        "emotion", "dair-ai/emotion", "text", "label", "Which emotion does the text express?",
        {"sadness": "sad or unhappy", "joy": "happy or joyful", "love": "affection or love", "anger": "angry", "fear": "afraid or anxious", "surprise": "surprised"},
        config="split", limit=limit,
    )


def conv_banking77(limit=None):
    """mteb/banking77 (parquet mirror of PolyAI/banking77): 77 intents in one question."""
    return _text_label_dataset("banking77", "mteb/banking77", "What is the customer's banking intent?", limit=limit,
                               splits=(("train", "train"), ("test", "test")))


def conv_clinc150(limit=None):
    return _label_dataset(
        "clinc150", "clinc/clinc_oos", "text", "intent", "What is the user's intent?", None, config="plus", limit=limit,
    )


MASSIVE_Q = {
    "en": ("What is the scenario of this request?", "What is the intent of this request?"),
    "pt": ("Qual é o cenário deste pedido?", "Qual é a intenção deste pedido?"),
}


def conv_massive(limit=None, langs=("en", "pt", "es", "fr", "de", "it")) -> Iterator[Row]:
    """mteb/amazon_massive_{scenario,intent} (parquet mirror of AmazonScience/massive)."""
    for lang in langs:
        q_scn, q_int = MASSIVE_Q.get(lang, MASSIVE_Q["en"])
        yield from _text_label_dataset(f"massive_{lang}", "mteb/amazon_massive_scenario", q_scn, config=lang, lang=lang, limit=limit)
        yield from _text_label_dataset(f"massive_{lang}", "mteb/amazon_massive_intent", q_int, config=lang, lang=lang, limit=limit)


def conv_xnli(limit=None, langs=("en", "es", "fr", "de")) -> Iterator[Row]:
    labels = ["entailment", "neutral", "contradiction"]
    for lang in langs:
        for hf_split, split in (("train", "train"), ("validation", "val"), ("test", "test")):
            try:
                ds = _load("facebook/xnli", lang, split=hf_split)
            except Exception:
                continue
            for i, r in enumerate(ds):
                if limit and i >= limit:
                    break
                state = {"premise": r["premise"], "hypothesis": r["hypothesis"]}
                yield make_row(f"xnli_{lang}", f"{hf_split}{i}", state, "Does the premise entail, contradict, or say nothing about the hypothesis?", "choice", choice_options(labels), one_hot(r["label"], 3), split, lang)
                yield make_row(f"xnli_{lang}", f"{hf_split}{i}/n", state, "The hypothesis follows from the premise.", "noul", NOUL_OPTIONS, [0.0, 1.0] if r["label"] == 0 else [1.0, 0.0], split, lang)


def conv_assin2(limit=None) -> Iterator[Row]:
    """ASSIN2 (pt-BR): RTE as noul and relatedness 1-5 as score."""
    levels = score_options(["sem relação", "pouco relacionadas", "parcialmente relacionadas", "muito relacionadas", "mesmo significado"])
    for hf_split, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        try:
            ds = _load("nilc-nlp/assin2", split=hf_split)
        except Exception:
            continue
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            state = {"premissa": r["premise"], "hipótese": r["hypothesis"]}
            ent = 1 if r["entailment_judgment"] == 1 else 0  # ClassLabel(names=['NONE', 'ENTAILMENT'])
            yield make_row("assin2", f"{hf_split}{i}/rte", state, "A hipótese é consequência da premissa.", "noul", NOUL_OPTIONS, [1.0 - ent, float(ent)], split, "pt")
            rel = float(r["relatedness_score"])  # 1..5
            lvl = min(4, max(0, int(round(rel)) - 1))
            yield make_row("assin2", f"{hf_split}{i}/rel", state, "Quão relacionadas são as duas frases?", "score", levels, one_hot(lvl, 5), split, "pt")


def conv_b2w(limit=None) -> Iterator[Row]:
    """B2W-Reviews01 (pt-BR): star rating 1-5 as an ordinal score."""
    levels = score_options(["péssimo", "ruim", "regular", "bom", "excelente"])
    try:
        ds = _load("ruanchaves/b2w-reviews01", split="train", revision="refs/convert/parquet")
    except Exception:
        return
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        rating = r.get("overall_rating")
        text = r.get("review_text") or ""
        if rating is None or not text.strip():
            continue
        try:
            lvl = min(4, max(0, int(float(rating)) - 1))
        except ValueError:
            continue
        state = {"título": r.get("review_title") or "", "avaliação": text}
        yield make_row("b2w", i, state, "Qual é a satisfação do cliente com o produto?", "score", levels, one_hot(lvl, 5), _split_name(i), "pt")


def conv_hatebr(limit=None) -> Iterator[Row]:
    try:
        ds = _load("ruanchaves/hatebr", split="train", revision="refs/convert/parquet")
    except Exception:
        return
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        text = r.get("instagram_comments") or r.get("text")
        label = r.get("offensive_language")
        if text is None or label is None:
            continue
        off = _truthy(label)
        yield make_row("hatebr", i, text, "O comentário é ofensivo.", "noul", NOUL_OPTIONS, [0.0, 1.0] if off else [1.0, 0.0], _split_name(i), "pt")


def conv_boolq(limit=None) -> Iterator[Row]:
    for hf_split, split in (("train", "train"), ("validation", "val")):
        ds = _load("google/boolq", split=hf_split)
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            state = {"passage": r["passage"]}
            yield make_row("boolq", f"{hf_split}{i}", state, r["question"], "noul", NOUL_OPTIONS, [0.0, 1.0] if r["answer"] else [1.0, 0.0], split)


def conv_race(limit=None) -> Iterator[Row]:
    for hf_split, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        ds = _load("ehovy/race", "all", split=hf_split)
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            names = ["A", "B", "C", "D"][: len(r["options"])]
            options = [f"{n}: {o}" for n, o in zip(names, r["options"])]
            yield make_row("race", f"{hf_split}{i}", {"article": r["article"]}, r["question"], "choice", options, one_hot(names.index(r["answer"]), len(names)), split)


def conv_ultrafeedback(limit=None) -> Iterator[Row]:
    """Pairwise preference as a 2-way choice (which response is better?)."""
    for hf_split, split in (("train_prefs", "train"), ("test_prefs", "test")):
        ds = _load("HuggingFaceH4/ultrafeedback_binarized", split=hf_split, streaming=True)
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            chosen = r["chosen"][-1]["content"] if r.get("chosen") else None
            rejected = r["rejected"][-1]["content"] if r.get("rejected") else None
            if not chosen or not rejected:
                continue
            flip = (i % 2) == 1
            a, b = (rejected, chosen) if flip else (chosen, rejected)
            state = {"prompt": r["prompt"], "response_A": a, "response_B": b}
            yield make_row("ultrafeedback", f"{hf_split}{i}", state, "Which response answers the prompt better?", "choice", ["A: response_A is better", "B: response_B is better"], one_hot(1 if flip else 0, 2), split)


def conv_helpsteer2(limit=None) -> Iterator[Row]:
    levels = score_options(["not helpful", "slightly helpful", "partially helpful", "mostly helpful", "perfectly helpful"])
    for hf_split, split in (("train", "train"), ("validation", "val")):
        ds = _load("nvidia/HelpSteer2", split=hf_split)
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            state = {"prompt": r["prompt"], "response": r["response"]}
            yield make_row("helpsteer2", f"{hf_split}{i}", state, "How helpful is the response?", "score", levels, one_hot(int(r["helpfulness"]), 5), split)


CONVERTERS: dict[str, Callable[..., Iterator[Row]]] = {
    "open_jev": conv_open_jev,
    "typed_decisions": conv_typed_decisions,
    "agnews": conv_agnews,
    "emotion": conv_emotion,
    "banking77": conv_banking77,
    "clinc150": conv_clinc150,
    "massive": conv_massive,
    "xnli": conv_xnli,
    "assin2": conv_assin2,
    "b2w": conv_b2w,
    "hatebr": conv_hatebr,
    "boolq": conv_boolq,
    "race": conv_race,
    "ultrafeedback": conv_ultrafeedback,
    "helpsteer2": conv_helpsteer2,
}


# ------------------------------------------------- generic typed corpora
_NOUL_WORDS = {("false", "true"), ("no", "yes"), ("não", "sim"), ("nao", "sim")}


def _coerce_generic(r: dict) -> dict | None:
    """Map the common field variants of Jev-style corpora onto a decision row."""
    state = r.get("state", r.get("state_json", r.get("context", r.get("text"))))
    if state is None:
        return None
    if isinstance(state, str) and state[:1] in "{[":
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            pass
    qtype = (r.get("kind") or r.get("type") or r.get("qtype") or "").lower()
    question = r.get("question") or r.get("instructions") or r.get("prompt") or ""
    options = r.get("options") or r.get("choices") or r.get("criteria")
    names = None
    if isinstance(options, dict):
        names = list(options)
        options = [f"{k}: {v}" if v else str(k) for k, v in options.items()]
    if not options:
        return None
    options = [str(o) for o in options]
    target = r.get("target")
    if target is None:
        target = r.get("probabilities", r.get("probs"))
    if isinstance(target, dict):
        keys = names or list(target)
        target = [float(target.get(k, 0.0)) for k in keys]
    if target is None:
        label = r.get("label", r.get("answer"))
        if label is None:
            return None
        if isinstance(label, str) and label in options:
            label = options.index(label)
        if isinstance(label, bool):
            label = int(label)
        if not isinstance(label, int) or not 0 <= label < len(options):
            return None
        target = one_hot(label, len(options))
    if qtype in ("yesno", "boolean", "bool", "noul"):
        qtype = "noul"
    elif qtype in ("ordinal", "score", "rating", "scale"):
        qtype = "score"
    elif qtype:
        qtype = "choice"
    else:
        qtype = "noul" if tuple(o.lower() for o in options) in _NOUL_WORDS else "choice"
    if qtype == "noul":
        if len(options) != 2:
            return None
        options = NOUL_OPTIONS
    return {"state": state, "question": render_content(question), "type": qtype, "options": options, "target": [float(t) for t in target]}


def _generic_rows(source, hf_id, config=None, limit=None, lang="en",
                  splits=(("train", "train"), ("validation", "val"), ("test", "test"))) -> Iterator[Row]:
    for hf_split, split in splits:
        try:
            ds = _load(hf_id, config, split=hf_split, streaming=True)
        except Exception:
            continue
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            row = _coerce_generic(r)
            if row is None:
                continue
            try:
                yield make_row(source, r.get("id", f"{hf_split}{i}"), row["state"], row["question"], row["type"], row["options"], row["target"], split, r.get("lang", lang))
            except ValueError:
                continue


def conv_tasksource(limit=None):
    """tasksource/tasksource-jev-typed-decisions: ~1M human-labelled recasts (per-row licenses)."""
    return _generic_rows("tasksource_jev", "tasksource/tasksource-jev-typed-decisions", None, limit)


def conv_procedural(limit=None):
    """tasksource/procedural-typed-decisions: ~396k rule-computed counting/dates/state-tracking rows (Apache-2.0)."""
    return _questions_rows("procedural", "tasksource/procedural-typed-decisions", "all",
                           (("train", "train"), ("validation", "val"), ("test", "test")), limit)


def conv_jev_distill(limit=None):
    """SargeDev/jev-distill-corpus-v3 (yuri_v3): distributions distilled from Jev 1.13.

    OPT-IN ONLY: these labels come from TypeSafe's API; check TypeSafe's terms of
    service before training on them. Not part of the default mixture.
    """
    return _generic_rows("jev_distill", "SargeDev/jev-distill-corpus-v3", "yuri_v3", limit)


CONVERTERS.update({
    "tasksource": conv_tasksource,
    "procedural": conv_procedural,
    "jev_distill": conv_jev_distill,
})


# -------------------------------------------------------------- synthetic
TEACHER_SYSTEM = (
    "You are a calibrated decision engine. You receive a STATE and one typed QUESTION with OPTIONS. "
    "Respond ONLY with a JSON object {\"probabilities\": [p_0, ..., p_{n-1}]} giving your honest probability "
    "for each option in the given order (they must sum to 1). For 'score' questions the options are ordered "
    "levels; for 'noul' the options are [false, true]. Do not explain."
)


def teacher_prompt(row: Row) -> list[dict[str, str]]:
    opts = "\n".join(f"[{i}] {o}" for i, o in enumerate(row["options"]))
    user = (
        f"STATE:\n{render_content(row['state'])}\n\nQUESTION TYPE: {row['type']}\n"
        f"QUESTION: {row['question']}\nOPTIONS:\n{opts}\n\nJSON:"
    )
    return [{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": user}]


def label_with_teacher(rows: Iterable[Row], model: str, samples: int = 3, permute: bool = True, base_url: str | None = None) -> Iterator[Row]:
    """Label rows with an OpenAI-compatible chat model (e.g. OpenRouter).

    Each row is queried `samples` times (with random option permutations when
    `permute`), and the returned distributions are averaged. The result is a
    soft target; disagreement between samples is kept in `teacher_std`.
    """
    import httpx

    url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/") + "/chat/completions"
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("set OPENROUTER_API_KEY (or OPENAI_API_KEY) to label with a teacher")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    rng = random.Random(0)
    with httpx.Client(timeout=120) as client:
        for row in rows:
            n = len(row["options"])
            acc = [0.0] * n
            got = 0
            dists = []
            for s in range(samples):
                perm = list(range(n))
                if permute and row["type"] == "choice" and s > 0:
                    rng.shuffle(perm)
                view = dict(row, options=[row["options"][j] for j in perm])
                body = {"model": model, "messages": teacher_prompt(view), "temperature": 0.7 if samples > 1 else 0.0,
                        "response_format": {"type": "json_object"}}
                try:
                    resp = client.post(url, headers=headers, json=body)
                    resp.raise_for_status()
                    text = resp.json()["choices"][0]["message"]["content"]
                    probs = json.loads(text)["probabilities"]
                    if len(probs) != n:
                        continue
                    total = float(sum(probs))
                    probs = [float(p) / total for p in probs]
                except Exception:
                    continue
                unperm = [0.0] * n
                for pos, j in enumerate(perm):
                    unperm[j] = probs[pos]
                dists.append(unperm)
                acc = [a + p for a, p in zip(acc, unperm)]
                got += 1
            if not got:
                continue
            target = [a / got for a in acc]
            std = [0.0] * n
            if got > 1:
                for j in range(n):
                    std[j] = (sum((d[j] - target[j]) ** 2 for d in dists) / got) ** 0.5
            yield dict(row, target=target, teacher=model, teacher_samples=got, teacher_std=std)


# -------------------------------------------------------------------- mix
DEFAULT_CAPS = {
    # per-source caps for the training mixture (rows); sources not listed use --default-cap
    "xnli": 120_000, "agnews": 40_000, "b2w": 60_000, "race": 60_000, "ultrafeedback": 40_000,
    "procedural": 150_000, "massive": 120_000,
}


def _source_family(source: str) -> str:
    return source.split("_")[0] if source.startswith(("xnli_", "massive_")) else source


def mix(in_path: str | Path, out_path: str | Path, caps: dict[str, int], default_cap: int, seed: int = 0,
        per_source: int | None = None) -> dict[str, int]:
    """Cap every source family, shuffle globally, write JSONL. `per_source` (val/test): take exactly that many per source."""
    by: dict[str, list[str]] = {}
    with Path(in_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            src = json.loads(line)["source"]
            key = src if per_source else _source_family(src)
            by.setdefault(key, []).append(line)
    rng = random.Random(seed)
    out: list[str] = []
    counts: dict[str, int] = {}
    for key, lines in sorted(by.items()):
        rng.shuffle(lines)
        cap = per_source if per_source else caps.get(key, default_cap)
        take = lines[:cap]
        counts[key] = len(take)
        out += take
    rng.shuffle(out)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as f:
        f.writelines(out)
    return counts


# -------------------------------------------------------------------- IO
def write_jsonl(rows: Iterable[Row], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, limit: int | None = None) -> Iterator[Row]:
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def stats(path: str | Path) -> dict[str, Any]:
    by_source, by_type, by_split, by_lang = Counter(), Counter(), Counter(), Counter()
    n = 0
    for r in read_jsonl(path):
        n += 1
        by_source[r["source"]] += 1
        by_type[r["type"]] += 1
        by_split[r["split"]] += 1
        by_lang[r.get("lang", "?")] += 1
    return {"rows": n, "by_source": dict(by_source), "by_type": dict(by_type), "by_split": dict(by_split), "by_lang": dict(by_lang)}


# ------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="groot-data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="convert public datasets to decision rows")
    b.add_argument("--sources", default="open_jev,typed_decisions,tasksource,procedural,agnews,emotion,banking77,clinc150,massive,xnli,assin2,b2w,hatebr,boolq,race,ultrafeedback,helpsteer2")
    b.add_argument("--out", default="data")
    b.add_argument("--limit", type=int, default=None, help="rows per split per source (debug)")
    i = sub.add_parser("inspect", help="print the first record of a HF dataset")
    i.add_argument("hf_id")
    i.add_argument("--config", default=None)
    i.add_argument("--split", default="train")
    s = sub.add_parser("synth", help="label rows with a teacher model")
    s.add_argument("--rows", required=True, help="input JSONL of unlabeled rows")
    s.add_argument("--teacher", required=True)
    s.add_argument("--samples", type=int, default=3)
    s.add_argument("--out", required=True)
    mx = sub.add_parser("mix", help="cap sources, shuffle, and build a stratified val/test")
    mx.add_argument("--src", required=True, help="directory produced by `build` (train/val/test.jsonl)")
    mx.add_argument("--out", required=True)
    mx.add_argument("--default-cap", type=int, default=100_000)
    mx.add_argument("--cap", action="append", default=[], help="source=N override, repeatable")
    mx.add_argument("--val-per-source", type=int, default=300)
    mx.add_argument("--test-per-source", type=int, default=1000)
    t = sub.add_parser("stats")
    t.add_argument("path")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        handles = {sp: (out / f"{sp}.jsonl").open("w", encoding="utf-8") for sp in ("train", "val", "test")}
        counts: Counter = Counter()
        for name in [s.strip() for s in args.sources.split(",") if s.strip()]:
            conv = CONVERTERS[name]
            try:
                for row in conv(limit=args.limit):
                    handles[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[(name, row["split"])] += 1
                print(f"[{name}] " + ", ".join(f"{sp}={counts[(name, sp)]}" for sp in ("train", "val", "test")), flush=True)
            except Exception as exc:  # keep going: a missing dataset must not kill the build
                print(f"[{name}] FAILED: {exc}", file=sys.stderr, flush=True)
        for h in handles.values():
            h.close()
        print(json.dumps({f"{k[0]}/{k[1]}": v for k, v in counts.items()}, indent=2))
    elif args.cmd == "inspect":
        ds = _load(args.hf_id, args.config, split=args.split, streaming=True)
        first = next(iter(ds))
        print(json.dumps({k: (str(v)[:300]) for k, v in first.items()}, indent=2, ensure_ascii=False))
    elif args.cmd == "synth":
        n = write_jsonl(label_with_teacher(read_jsonl(args.rows), args.teacher, args.samples), args.out)
        print(f"wrote {n} labelled rows to {args.out}")
    elif args.cmd == "mix":
        caps = dict(DEFAULT_CAPS)
        for c in args.cap:
            k, v = c.split("=")
            caps[k] = int(v)
        src, out = Path(args.src), Path(args.out)
        report = {
            "train": mix(src / "train.jsonl", out / "train.jsonl", caps, args.default_cap),
            "val": mix(src / "val.jsonl", out / "val.jsonl", caps, args.default_cap, per_source=args.val_per_source),
            "test": mix(src / "test.jsonl", out / "test.jsonl", caps, args.default_cap, per_source=args.test_per_source),
        }
        (out / "mix_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: sum(v.values()) for k, v in report.items()}))
        print(json.dumps(report["train"], indent=1))
    elif args.cmd == "stats":
        print(json.dumps(stats(args.path), indent=2))


if __name__ == "__main__":
    main()
