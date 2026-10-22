"""Groot 1 — Tree-Reduce Decision Transformer.

Forward pass (one request, Q questions):

    chunks  = split(state)                      # parallel leaves
    H_c     = Encoder(chunks)                   # one batched pass, shared weights
    leaves  = mean_pool(H_c) + seg_emb          # one vector per chunk
    nodes   = TreeReduce(leaves)                # fork/join merge, log depth
    memory  = [H_c tokens ; nodes]              # state memory, built ONCE
    for each option of each question (parallel, batched):
        h_i = Encoder([CLS] head [SEP] [MASK] option_i [SEP])[MASK]   # pointwise
    for each question (parallel):
        S   = [q_token ; h_1 .. h_K]            # a SET: identical positions
        S   = Fusion(S, memory)                 # set self-attn + cross-attn + GeGLU
        logits = scorer(S[1:])                  # one logit per option
        p = softmax(logits / T[qtype])

The state is encoded once and every question reads it through cross-attention,
so the latency of a request grows only with the (small) question blocks. The
same dataflow is a fork/join tree, which is exactly what Bend parallelises
(see bend/groot.bend).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from .config import GrootConfig, QTYPES
from .serialize import EncodedQuestion, EncodedState

NEG = -1e4


# --------------------------------------------------------------------- RoPE
def rope_cos_sin(position_ids: torch.Tensor, dim: int, theta: float, dtype: torch.dtype):
    """cos/sin tables for rotary embeddings; position_ids (B, L) -> (B, L, dim)."""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=position_ids.device, dtype=torch.float32) / dim))
    freqs = position_ids[..., None].float() * inv  # (B, L, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, L, Dh); cos/sin: (B, L, Dh)
    return x * cos[:, None] + rotate_half(x) * sin[:, None]


# ------------------------------------------------------------------ fusion
class SelfAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.Wqkv = nn.Linear(d, 3 * d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.dropout = dropout

    def forward(self, x, cos, sin, mask):
        B, L, d = x.shape
        q, k, v = self.Wqkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[:, None, None, :], dropout_p=self.dropout if self.training else 0.0
        )
        return self.Wo(out.transpose(1, 2).reshape(B, L, d))


class CrossAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wkv = nn.Linear(d, 2 * d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.Wo.weight)  # start as identity residual
        self.dropout = dropout

    def forward(self, x, memory, memory_mask):
        B, L, d = x.shape
        M = memory.shape[1]
        q = self.Wq(x).view(B, L, self.h, self.dh).transpose(1, 2)
        k, v = self.Wkv(memory).view(B, M, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=memory_mask[:, None, None, :], dropout_p=self.dropout if self.training else 0.0
        )
        return self.Wo(out.transpose(1, 2).reshape(B, L, d))


class GeGLU(nn.Module):
    def __init__(self, d: int, inter: int, dropout: float):
        super().__init__()
        self.Wi = nn.Linear(d, 2 * inter, bias=False)
        self.Wo = nn.Linear(inter, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, g = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(self.drop(F.gelu(a) * g))


class FusionLayer(nn.Module):
    """Pre-norm block: self-attention (RoPE) -> cross-attention to memory -> GeGLU."""

    def __init__(self, cfg: GrootConfig):
        super().__init__()
        d = cfg.hidden_size
        self.attn_norm = nn.LayerNorm(d, eps=cfg.norm_eps, bias=False)
        self.attn = SelfAttention(d, cfg.fusion_heads, cfg.dropout)
        self.cross_norm = nn.LayerNorm(d, eps=cfg.norm_eps, bias=False)
        self.cross = CrossAttention(d, cfg.fusion_heads, cfg.dropout)
        self.mlp_norm = nn.LayerNorm(d, eps=cfg.norm_eps, bias=False)
        self.mlp = GeGLU(d, cfg.fusion_intermediate, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, mask, memory, memory_mask):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, mask))
        x = x + self.drop(self.cross(self.cross_norm(x), memory, memory_mask))
        x = x + self.drop(self.mlp(self.mlp_norm(x)))
        return x


# ------------------------------------------------------------------- tree
class TreeReducer(nn.Module):
    """Balanced binary fork/join merge of chunk summaries.

    node = LN(l + r + W [l ; r ; l*r ; |l-r|]) + level_emb[level]
    A node with a single valid child passes that child through unchanged.
    Returns every internal node (multi-scale memory) and the root.
    """

    def __init__(self, cfg: GrootConfig):
        super().__init__()
        d = cfg.hidden_size
        self.merge = nn.Linear(4 * d, d)
        if cfg.tree_merge_zero_init:
            nn.init.zeros_(self.merge.weight)
            nn.init.zeros_(self.merge.bias)
        self.norm = nn.LayerNorm(d, eps=cfg.norm_eps, bias=False)
        self.level_emb = nn.Embedding(cfg.tree_levels, d)
        nn.init.normal_(self.level_emb.weight, std=0.02)
        self.levels = cfg.tree_levels

    def forward(self, leaves: torch.Tensor, mask: torch.Tensor):
        B, m, d = leaves.shape
        P = 1 << max(1, (m - 1).bit_length())  # next power of two >= m (at least 2)
        x = F.pad(leaves, (0, 0, 0, P - m))
        valid = F.pad(mask, (0, P - m), value=False)
        nodes, node_masks = [], []
        level = 0
        while x.shape[1] > 1:
            l, r = x[:, 0::2], x[:, 1::2]
            vl, vr = valid[:, 0::2], valid[:, 1::2]
            merged = self.norm(l + r + self.merge(torch.cat((l, r, l * r, (l - r).abs()), dim=-1)))
            merged = merged + self.level_emb.weight[min(level, self.levels - 1)]
            both = (vl & vr)[..., None]
            only_l = (vl & ~vr)[..., None]
            x = torch.where(both, merged, torch.where(only_l, l, r))
            valid = vl | vr
            keep = (vl & vr)  # only true merges are new information
            nodes.append(x)
            node_masks.append(keep)
            level += 1
        root = x[:, 0]
        if nodes:
            nodes_t = torch.cat(nodes, dim=1)
            masks_t = torch.cat(node_masks, dim=1)
        else:
            nodes_t = leaves[:, :0]
            masks_t = mask[:, :0]
        return nodes_t, masks_t, root


# ------------------------------------------------------------------ model
@dataclass
class Batch:
    """Padded tensors for a batch of states and questions (one encoder sequence per question)."""

    chunk_ids: torch.Tensor  # (C, Lc)
    chunk_mask: torch.Tensor  # (C, Lc) bool
    chunk_owner: torch.Tensor  # (C,) state index of each chunk
    chunk_index: torch.Tensor  # (C,) position of the chunk inside its state
    q_ids: torch.Tensor  # (Q, Lq)
    q_mask: torch.Tensor  # (Q, Lq) bool
    q_pos: torch.Tensor  # (Q, Lq) RoPE positions (option blocks share a base)
    q_markers: torch.Tensor  # (Q, K) [MASK] index of each option (0 where padded)
    q_block: torch.Tensor  # (Q, Lq) 0 = prefix, k+1 = option k, -1 = padding
    q_state: torch.Tensor  # (Q,) state index per question
    q_type: torch.Tensor  # (Q,)
    q_count: torch.Tensor  # (Q,) number of options

    def to(self, device):
        return Batch(**{k: v.to(device) for k, v in self.__dict__.items()})

    @property
    def n_states(self) -> int:
        return int(self.chunk_owner.max()) + 1

    @property
    def n_questions(self) -> int:
        return int(self.q_type.shape[0])


def _pad(seqs, L, value):
    out = torch.full((len(seqs), L), value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def collate(states: Sequence[EncodedState], questions: Sequence[EncodedQuestion], pad_id: int, q_state: Sequence[int] | None = None) -> Batch:
    """Build a Batch; `q_state[i]` is the state index of question i (default: question i -> state i)."""
    if q_state is None:
        q_state = list(range(len(questions)))
    chunk_ids, chunk_owner, chunk_index = [], [], []
    for s, st in enumerate(states):
        for i, c in enumerate(st.chunks):
            chunk_ids.append(c)
            chunk_owner.append(s)
            chunk_index.append(i)
    cids = _pad(chunk_ids, max(len(c) for c in chunk_ids), pad_id)
    Lq = max(len(q.ids) for q in questions)
    Lq = ((Lq + 7) // 8) * 8
    qids = _pad([q.ids for q in questions], Lq, pad_id)
    qmask = torch.zeros_like(qids, dtype=torch.bool)
    for i, q in enumerate(questions):
        qmask[i, : len(q.ids)] = True
    qpos = _pad([q.position_ids for q in questions], Lq, 0)
    qblock = _pad([q.blocks if q.blocks is not None else [0] * len(q.ids) for q in questions], Lq, -1)
    K = max(q.n_options for q in questions)
    return Batch(
        chunk_ids=cids,
        chunk_mask=cids.ne(pad_id),
        chunk_owner=torch.tensor(chunk_owner),
        chunk_index=torch.tensor(chunk_index),
        q_ids=qids,
        q_mask=qmask,
        q_pos=qpos,
        q_markers=_pad([q.markers for q in questions], K, 0),
        q_block=qblock,
        q_state=torch.tensor(list(q_state)),
        q_type=torch.tensor([q.qtype for q in questions]),
        q_count=torch.tensor([q.n_options for q in questions]),
    )


class GrootDecisionModel(nn.Module):
    def __init__(self, config: GrootConfig, encoder: nn.Module):
        super().__init__()
        self.cfg = config
        self.encoder = encoder
        d = config.hidden_size
        if getattr(encoder.config, "hidden_size", d) != d:
            raise ValueError("config.hidden_size must match the backbone hidden size")
        if config.encoder_layers is not None and hasattr(encoder, "layers"):
            encoder.layers = encoder.layers[: config.encoder_layers]
            encoder.config.num_hidden_layers = config.encoder_layers
        self.seg_emb = nn.Embedding(config.max_chunks, d)
        nn.init.normal_(self.seg_emb.weight, std=0.02)
        # sinusoidal chunk positions generalise to any number of chunks; the learned table
        # stays in the state dict so older checkpoints load unchanged
        self.chunk_pos_scale = nn.Parameter(torch.tensor(0.02))
        self.qtype_emb = nn.Embedding(len(config.qtypes), d)
        nn.init.zeros_(self.qtype_emb.weight)
        self.level_emb = nn.Embedding(256, d)  # ordinal roles for score levels / noul outcomes
        nn.init.normal_(self.level_emb.weight, std=0.02)
        self.tree = TreeReducer(config)
        self.fusion = nn.ModuleList([FusionLayer(config) for _ in range(config.fusion_layers)])
        self.final_norm = nn.LayerNorm(d, eps=config.norm_eps, bias=False)
        dh = config.scorer_hidden or d
        self.scorer = nn.Sequential(nn.LayerNorm(d, eps=config.norm_eps), nn.Linear(d, dh), nn.GELU(), nn.Linear(dh, 1))
        self.conf_head = nn.Sequential(nn.Linear(2 * d + 4, config.conf_hidden), nn.GELU(), nn.Linear(config.conf_hidden, 1))
        self.register_buffer("temperature", torch.full((len(config.qtypes),), float(config.temperature_init)))
        if config.fusion_init_from_encoder:
            self._init_fusion_from_encoder()

    # ------------------------------------------------------------ building
    @classmethod
    def from_backbone(cls, config: GrootConfig, backbone: str | None = None) -> "GrootDecisionModel":
        from transformers import AutoModel

        kw = {}
        if config.global_attention:
            from transformers import AutoConfig

            n = AutoConfig.from_pretrained(backbone or config.backbone).num_hidden_layers
            kw["layer_types"] = ["full_attention"] * n
        encoder = AutoModel.from_pretrained(backbone or config.backbone, attn_implementation="sdpa", **kw)
        encoder.config.reference_compile = False
        return cls(config, encoder)

    @classmethod
    def tiny(cls, config: GrootConfig, vocab_size: int = 1024, layers: int = 2) -> "GrootDecisionModel":
        """A randomly initialised miniature (for tests and the Bend parity harness)."""
        from transformers import AutoConfig, AutoModel

        enc_cfg = AutoConfig.for_model(
            "modernbert",
            vocab_size=vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.fusion_intermediate,
            num_hidden_layers=layers,
            num_attention_heads=config.fusion_heads,
            max_position_embeddings=8192,
            global_attn_every_n_layers=3,
            local_attention=128,
            global_rope_theta=config.rope_theta,
            local_rope_theta=config.rope_theta,
            pad_token_id=0,
            cls_token_id=2,
            sep_token_id=1,
            mask_token_id=4,
            **({"layer_types": ["full_attention"] * layers} if config.global_attention else {}),
        )
        enc_cfg.reference_compile = False
        encoder = AutoModel.from_config(enc_cfg, attn_implementation="sdpa")
        cfg = GrootConfig.from_dict({**config.to_dict(), "encoder_layers": None})
        return cls(cfg, encoder)

    def _init_fusion_from_encoder(self) -> None:
        layers = getattr(self.encoder, "layers", None)
        if not layers:
            return
        src = list(layers)[-len(self.fusion) :]
        for fl, el in zip(self.fusion, src):
            try:
                fl.attn.Wqkv.weight.data.copy_(el.attn.Wqkv.weight.data)
                fl.attn.Wo.weight.data.copy_(el.attn.Wo.weight.data)
                fl.mlp.Wi.weight.data.copy_(el.mlp.Wi.weight.data)
                fl.mlp.Wo.weight.data.copy_(el.mlp.Wo.weight.data)
                if hasattr(el.attn_norm, "weight"):
                    fl.attn_norm.weight.data.copy_(el.attn_norm.weight.data)
                fl.mlp_norm.weight.data.copy_(el.mlp_norm.weight.data)
            except (AttributeError, RuntimeError):
                break  # shapes differ (custom backbone): keep random init

    @property
    def device(self) -> torch.device:
        return self.temperature.device

    # -------------------------------------------------------------- memory
    def sync_chunk_pos_trainable(self) -> None:
        """Freeze the parameters of the chunk-position scheme that is not in use (DDP needs every trainable param used)."""
        sinus = getattr(self.cfg, "chunk_pos", "learned") == "sinusoidal"
        self.seg_emb.weight.requires_grad_(not sinus)
        self.chunk_pos_scale.requires_grad_(sinus)

    def chunk_position(self, idx: torch.Tensor) -> torch.Tensor:
        if getattr(self.cfg, "chunk_pos", "learned") != "sinusoidal":
            return self.seg_emb(idx.clamp(max=self.seg_emb.num_embeddings - 1))
        d = self.cfg.hidden_size
        inv = torch.exp(torch.arange(0, d, 2, device=idx.device, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d))
        ang = idx.float()[:, None] * inv[None, :]
        pe = torch.zeros(idx.shape[0], d, device=idx.device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
        return (pe * self.chunk_pos_scale).to(self.seg_emb.weight.dtype)


    def encode_chunks(self, chunk_ids, chunk_mask):
        return self.encoder(input_ids=chunk_ids, attention_mask=chunk_mask.long()).last_hidden_state

    def build_memory(self, batch: Batch, n_examples: int):
        """State memory (tokens + tree nodes) for every example in the batch."""
        H = self.encode_chunks(batch.chunk_ids, batch.chunk_mask)  # (C, Lc, d)
        H = H + self.chunk_position(batch.chunk_index)[:, None, :]
        C, Lc, d = H.shape
        # per-example chunk counts
        counts = torch.bincount(batch.chunk_owner, minlength=n_examples)
        m = int(counts.max())
        # leaves: masked mean per chunk
        denom = batch.chunk_mask.sum(-1, keepdim=True).clamp_min(1)
        leaf = (H * batch.chunk_mask[..., None]).sum(1) / denom  # (C, d)
        leaves = H.new_zeros(n_examples, m, d)
        leaf_mask = torch.zeros(n_examples, m, dtype=torch.bool, device=H.device)
        leaves[batch.chunk_owner, batch.chunk_index] = leaf
        leaf_mask[batch.chunk_owner, batch.chunk_index] = True
        nodes, node_mask, root = self.tree(leaves, leaf_mask)
        # token memory: (B, m*Lc, d) laid out by chunk index
        tokens = H.new_zeros(n_examples, m, Lc, d)
        token_mask = torch.zeros(n_examples, m, Lc, dtype=torch.bool, device=H.device)
        tokens[batch.chunk_owner, batch.chunk_index] = H
        token_mask[batch.chunk_owner, batch.chunk_index] = batch.chunk_mask
        memory = torch.cat((tokens.view(n_examples, m * Lc, d), nodes), dim=1)
        memory_mask = torch.cat((token_mask.view(n_examples, m * Lc), node_mask), dim=1)
        return memory, memory_mask, root

    # ------------------------------------------------------------- decide
    def encode_questions(self, batch: Batch):
        """Early fusion: each question sequence (head + state head + option blocks) through the encoder."""
        blk = batch.q_block
        # query i may attend key j iff j is real and (j is prefix or j is in i's own option block)
        allowed = (blk[:, None, :] >= 0) & ((blk[:, None, :] == 0) | (blk[:, None, :] == blk[:, :, None]))
        mask4d = allowed[:, None]  # (Q, 1, Lq, Lq) bool, True = attend
        masks = {"full_attention": mask4d, "sliding_attention": mask4d}
        Hq = self.encoder(input_ids=batch.q_ids, attention_mask=masks, position_ids=batch.q_pos).last_hidden_state
        marker = Hq.gather(1, batch.q_markers[..., None].expand(-1, -1, Hq.shape[-1]))  # (Q, K, d)
        return marker, Hq[:, 0]

    def decide(self, batch: Batch, memory, memory_mask, root):
        """Set-level fusion of the options of every question against the state memory."""
        marker, cls = self.encode_questions(batch)
        Q, K, d = marker.shape
        slots = torch.arange(K, device=marker.device)
        valid = slots[None, :] < batch.q_count[:, None]
        ordinal = (batch.q_type != 0).to(marker.dtype)[:, None, None]  # score & noul keep their level roles
        marker = marker + ordinal * self.level_emb(slots.clamp(max=255))[None]
        S = torch.cat((cls[:, None] + self.qtype_emb(batch.q_type)[:, None], marker), dim=1)
        S_mask = torch.cat((torch.ones(Q, 1, dtype=torch.bool, device=marker.device), valid), dim=1)
        pos = torch.ones(Q, 1 + K, dtype=torch.long, device=marker.device)
        pos[:, 0] = 0  # the question token at 0, every option at 1: a set, not a list
        cos, sin = rope_cos_sin(pos, d // self.cfg.fusion_heads, self.cfg.rope_theta, S.dtype)
        mem, mem_mask, rt = memory[batch.q_state], memory_mask[batch.q_state], root[batch.q_state]
        for layer in self.fusion:
            S = layer(S, cos, sin, S_mask, mem, mem_mask)
        S = self.final_norm(S)
        logits = self.scorer(S[:, 1:]).squeeze(-1).float().masked_fill(~S_mask[:, 1:], NEG)
        probs = (logits / self.temperature[batch.q_type][:, None]).softmax(-1)
        top = probs.topk(min(2, probs.shape[1]), dim=-1).values
        if top.shape[1] == 1:
            top = torch.cat((top, torch.zeros_like(top)), dim=-1)
        count = batch.q_count.clamp_min(2).float()
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1) / count.log()
        feats = torch.stack((top[:, 0], top[:, 0] - top[:, 1], entropy, count / 255.0), dim=-1)
        conf_in = torch.cat((rt.float(), S[:, 0].float(), feats.detach()), dim=-1)
        conf_logit = self.conf_head(conf_in).squeeze(-1)
        return {"logits": logits, "probabilities": probs, "confidence_logit": conf_logit}

    def forward(self, batch: Batch):
        memory, memory_mask, root = self.build_memory(batch, batch.n_states)
        return self.decide(batch, memory, memory_mask, root)

    # ------------------------------------------------------------ persist
    def save_pretrained(self, directory: str | Path, tokenizer=None) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.cfg.save(root)
        self.encoder.config.save_pretrained(root / "encoder")
        if tokenizer is not None:
            tokenizer.save_pretrained(root / "tokenizer")
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(state, str(root / "model.safetensors"), metadata={"format": "pt", "family": "groot", "version": "1"})
        (root / "calibration.json").write_text(json.dumps({"temperature": self.temperature.tolist()}, indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, directory: str | Path, device: str | torch.device = "cpu") -> "GrootDecisionModel":
        from transformers import AutoConfig, AutoModel

        root = Path(directory)
        cfg = GrootConfig.load(root)
        enc_cfg = AutoConfig.from_pretrained(root / "encoder")
        enc_cfg.reference_compile = False
        encoder = AutoModel.from_config(enc_cfg, attn_implementation="sdpa")
        cfg = GrootConfig.from_dict({**cfg.to_dict(), "encoder_layers": None, "fusion_init_from_encoder": False})
        model = cls(cfg, encoder)
        missing, unexpected = model.load_state_dict(load_file(str(root / "model.safetensors")), strict=False)
        if unexpected or set(missing) - {"chunk_pos_scale"}:
            raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
        calib = root / "calibration.json"
        if calib.exists():
            temps = json.loads(calib.read_text()).get("temperature")
            if temps:
                model.temperature.copy_(torch.tensor(temps, dtype=model.temperature.dtype))
        return model.to(device).eval()

    def count_parameters(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        total = sum(p.numel() for p in self.parameters())
        emb = self.encoder.get_input_embeddings().weight.numel()
        return {"total": total, "encoder": enc, "embeddings": emb, "head": total - enc}
