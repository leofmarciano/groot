"""Model configuration and size presets for Groot 1.

The architecture ("Tree-Reduce Decision Transformer") has four parts:

1. Leaf encoder  — a pretrained ModernBERT-style encoder (mmBERT) applied to
                   state *chunks* in parallel and to the question block.
2. Tree reducer  — a parallel fork/join merge of chunk summaries into a
                   balanced binary tree (log-depth global context).
3. Fusion        — a few transformer layers over the question block with
                   cross-attention to the state memory (tokens + tree nodes).
4. Typed heads   — one logit per option marker, per-type temperature, and a
                   confidence head that predicts correctness.

The same structure is mirrored in the Bend engine (bend/groot.bend).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

QTYPES: tuple[str, ...] = ("choice", "score", "noul")
QTYPE_IDS: dict[str, int] = {name: i for i, name in enumerate(QTYPES)}

# TypeSafe contract limits (observed in https://api.typesafe.ai/openapi.json, Oct 2026).
JEV_MAX_CHOICE_OPTIONS = 255
JEV_MAX_SCORE_LEVELS = 10
JEV_PER_REQUEST_CONTEXT = 64_000
JEV_PER_QUESTION_CONTEXT = 32_000
JEV_LONG_CONTEXT = 262_000


@dataclass
class GrootConfig:
    """Hyper-parameters of a Groot 1 model.

    Everything is JSON-serialisable so a checkpoint directory carries its own
    `groot_config.json`.
    """

    name: str = "groot-1-s"
    # --- leaf encoder -------------------------------------------------------
    backbone: str = "jhu-clsp/mmBERT-small"
    hidden_size: int = 384
    encoder_layers: int | None = None  # None = keep all backbone layers
    vocab_keep: int | None = None  # None = full 256k vocab; int = pruned vocab size
    # --- chunking / state memory -------------------------------------------
    chunk_tokens: int = 512  # leaf length (tokens incl. CLS/SEP)
    max_chunks: int = 512  # 512 * 512 = 262k tokens (JEV long-context parity)
    # --- tree reducer -------------------------------------------------------
    tree_levels: int = 10  # ceil(log2(max_chunks)) + 1
    tree_merge_zero_init: bool = True
    chunk_pos: str = "learned"  # "learned" (one embedding per chunk index, <= max_chunks) or "sinusoidal" (any length)
    # --- question block -----------------------------------------------------
    max_question_tokens: int = 1024
    max_option_tokens: int = 48
    max_options_per_block: int = 32  # above this: anchored two-stage scoring
    option_shared_positions: bool = True  # every option block starts at the same RoPE position (order-free)
    early_state_tokens: int = 512  # state tokens placed inside each question sequence (early fusion)
    state_retrieval: bool = False  # long states: fill the early-state budget with the pieces most relevant to the question (BM25)
    retrieval_piece_tokens: int = 64  # granularity of retrievable pieces
    retrieval_lead_tokens: int = 64  # always keep the first tokens of the state (headers, ids)
    global_attention: bool = True  # all encoder layers global (options must see the state in every layer)
    # --- fusion -------------------------------------------------------------
    fusion_layers: int = 3
    fusion_heads: int = 6
    fusion_intermediate: int = 1152
    fusion_init_from_encoder: bool = True
    rope_theta: float = 160000.0
    norm_eps: float = 1e-5
    dropout: float = 0.1
    # --- heads --------------------------------------------------------------
    scorer_hidden: int | None = None  # None = hidden_size
    conf_hidden: int = 256
    temperature_init: float = 1.0
    # --- misc ---------------------------------------------------------------
    qtypes: tuple[str, ...] = QTYPES
    dtype: str = "float32"
    organization: str = "Marciano Theory SI"
    family: str = "Groot"
    version: str = "1"

    # ------------------------------------------------------------------ utils
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["qtypes"] = list(self.qtypes)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GrootConfig":
        names = {f.name for f in fields(cls)}
        clean = {k: v for k, v in d.items() if k in names}
        if "qtypes" in clean:
            clean["qtypes"] = tuple(clean["qtypes"])
        return cls(**clean)

    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / "groot_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "GrootConfig":
        path = Path(directory) / "groot_config.json"
        return cls.from_dict(json.loads(path.read_text()))

    @property
    def max_state_tokens(self) -> int:
        return self.chunk_tokens * self.max_chunks


# ---------------------------------------------------------------- presets
PRESETS: dict[str, dict[str, Any]] = {
    # Nano: a 6-layer student distilled from Groot-1-S with a pruned vocabulary.
    # Small enough to run end-to-end inside the Bend engine on CPU/GPU.
    "groot-1-n": dict(
        name="groot-1-n",
        backbone="jhu-clsp/mmBERT-small",
        hidden_size=384,
        encoder_layers=6,
        vocab_keep=65536,
        chunk_tokens=256,
        max_chunks=256,
        tree_levels=9,
        fusion_layers=2,
        fusion_heads=6,
        fusion_intermediate=1152,
    ),
    # Small: mmBERT-small (22 layers, d=384, 256k vocab) — Julia-1 class, ~150M.
    "groot-1-s": dict(
        name="groot-1-s",
        backbone="jhu-clsp/mmBERT-small",
        hidden_size=384,
        fusion_layers=3,
        fusion_heads=6,
        fusion_intermediate=1152,
    ),
    # Medium: mmBERT-base (22 layers, d=768, 12 heads) — ~320M.
    "groot-1-m": dict(
        name="groot-1-m",
        backbone="jhu-clsp/mmBERT-base",
        hidden_size=768,
        fusion_layers=3,
        fusion_heads=12,
        fusion_intermediate=1152,
    ),
}


def preset(name: str, **overrides: Any) -> GrootConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    cfg = dict(PRESETS[name])
    cfg.update(overrides)
    return GrootConfig(**cfg)


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """`--set key=value` pairs -> typed overrides (bool/int/float/str by the field's default type)."""
    out: dict[str, Any] = {}
    defaults = GrootConfig().to_dict()
    for item in items or []:
        k, _, v = item.partition("=")
        if k not in defaults:
            raise KeyError(f"unknown config field {k!r}")
        d = defaults[k]
        out[k] = (v.lower() in ("1", "true", "yes")) if isinstance(d, bool) else type(d)(v) if d is not None else v
    return out
