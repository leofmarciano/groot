"""Training for Groot 1: proper-scoring objective with calibration built in.

Objective per question (soft target t over k options, model probabilities p):

    L = w_soft * CE(t, p) + w_hard * CE(onehot(argmax t), p) + w_brier * ||p - t||^2
        + w_ord * E_p|i - argmax t|   (score questions only: ordinal distance)
        + w_conf * BCE(conf_logit, 1[argmax p == argmax t])  (correctness head)

CE with a soft target and the Brier term are proper scoring rules, so the
model is rewarded for honest probabilities; this is the supervised core of
"reinforcement learning for calibrated decisions". Augmentation permutes
choice options (the model must not learn list positions) and, optionally,
replaces option names by neutral ids.

Usage (single GPU):
    groot-train --preset groot-1-s --train data/train.jsonl --val data/val.jsonl --out runs/groot-1-s
Multi-GPU:
    torchrun --nproc_per_node 8 -m groot1.train --preset groot-1-s ...
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import mmap
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .calibration import Record, evaluate_records, fit_temperatures
from .config import QTYPE_IDS, GrootConfig, preset
from .data import read_jsonl
from .model import Batch, GrootDecisionModel, collate
from .serialize import Serializer


# ------------------------------------------------------------------- data
class RowStore:
    """Memory-light random access to a JSONL file of decision rows.

    Only an int64 offset per line is kept in RAM; rows are parsed on demand from
    an mmap of the file, so 4 DDP ranks x N loader workers share one page cache
    instead of each holding a full Python copy of the dataset. Picklable (the
    mmap is reopened lazily), so it works with fork and forkserver workers.
    """

    def __init__(self, path: str | Path, limit: int | None = None):
        self.path = str(path)
        offsets = [0]
        with open(self.path, "rb") as f:
            for line in f:
                offsets.append(offsets[-1] + len(line))
                if limit and len(offsets) > limit:
                    break
        starts = np.array(offsets[:-1], dtype=np.int64)
        ends = np.array(offsets[1:], dtype=np.int64)
        keep = (ends - starts) > 1  # skip blank lines
        self.starts, self.ends = starts[keep], ends[keep]
        if len(self.starts) == 0:
            raise ValueError(f"{path}: no rows")
        self._mm = None

    def __len__(self) -> int:
        return len(self.starts)

    def estimate_lengths(self, early_state_tokens: int, chars_per_token: float = 3.2) -> tuple[np.ndarray, np.ndarray]:
        """Approximate (question-sequence tokens, state tokens) per row, from characters. Cached on the store."""
        if getattr(self, "_qlen", None) is not None:
            return self._qlen, self._slen
        q = np.empty(len(self), dtype=np.int32)
        st = np.empty(len(self), dtype=np.int32)
        for i in range(len(self)):
            r = self[i]
            opts = r.get("options") or []
            qchars = len(r.get("question") or "") + sum(len(o) for o in opts)
            schars = max(0, self.length_key(i) - qchars - 200)
            stok = int(schars / chars_per_token)
            st[i] = stok
            q[i] = int(qchars / chars_per_token) + 2 * len(opts) + 8 + min(stok, early_state_tokens)
        self._qlen, self._slen = q, st
        return q, st

    def length_key(self, i: int) -> int:
        return int(self.ends[i] - self.starts[i])

    def __getitem__(self, i: int) -> dict:
        if self._mm is None:
            with open(self.path, "rb") as f:
                self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return json.loads(self._mm[int(self.starts[i]) : int(self.ends[i])])

    def take(self, n: int | None = None) -> list[dict]:
        n = len(self) if n is None else min(n, len(self))
        return [self[i] for i in range(n)]

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_mm"] = None
        return d


def permute_options(row: dict, rng: random.Random) -> dict:
    if row["type"] != "choice" or len(row["options"]) < 2:
        return row
    perm = list(range(len(row["options"])))
    rng.shuffle(perm)
    return dict(row, options=[row["options"][j] for j in perm], target=[row["target"][j] for j in perm])


def build_batch(rows: Sequence[dict], ser: Serializer, rng: random.Random | None, max_k: int | None = None) -> tuple[Batch, torch.Tensor]:
    """Encode rows; states shared by several rows are encoded once (q_state map)."""
    states, q_state, questions, targets = [], [], [], []
    key_to_idx: dict[str, int] = {}
    for row in rows:
        if rng is not None:
            row = permute_options(row, rng)
        key = row.get("state_key") or str(row["state"])
        if key not in key_to_idx:
            key_to_idx[key] = len(states)
            states.append(ser.encode_state(row["state"]))
        q_state.append(key_to_idx[key])
        _, q = ser.encode_row(row, states[key_to_idx[key]])
        questions.append(q)
        targets.append([float(t) for t in row["target"]][: q.n_options])
    batch = collate(states, questions, ser.pad, q_state)
    K = int(batch.q_count.max())
    target = torch.tensor([t + [0.0] * (K - len(t)) for t in targets], dtype=torch.float32)
    return batch, target


def epoch_plan(store: RowStore, batch_size: int, world: int, seed: int, epoch: int,
               qlen: np.ndarray | None = None, slen: np.ndarray | None = None,
               max_tokens: int | None = None, max_attn: float | None = None, chunk_tokens: int = 512, max_chunks: int = 8) -> list[list[int]]:
    """Batches of row indices for one epoch, identical on every rank.

    Global shuffle -> sort by length inside mega-chunks -> greedy batches capped by
    `batch_size` rows and, when given, by padded token budgets:
        rows * Lmax         <= max_tokens   (question sequences, activations)
        rows * Lmax^2       <= max_attn     (block-masked attention matrices)
        rows * state_tokens <= 4*max_tokens (state chunks)
    -> shuffle batches -> truncate to a multiple of `world`, so every rank gets
    exactly the same number of steps (no DDP hang at the end of an epoch).
    """
    rng = np.random.default_rng([seed, epoch])
    idx = rng.permutation(len(store))
    key = qlen if qlen is not None else (store.ends - store.starts)
    mega = batch_size * 64
    batches: list[list[int]] = []
    for start in range(0, len(idx), mega):
        chunk = idx[start : start + mega]
        chunk = chunk[np.argsort(key[chunk], kind="stable")]
        cur: list[int] = []
        lmax = 0
        smax = 0
        for j in chunk.tolist():
            lj = int(qlen[j]) if qlen is not None else 0
            sj = min(int(slen[j]), chunk_tokens * max_chunks) if slen is not None else 0
            nl, ns, n = max(lmax, lj), max(smax, sj), len(cur) + 1
            over = n > batch_size
            if max_tokens and qlen is not None:
                over = over or (n * nl > max_tokens) or (n * ns > 4 * max_tokens)
            if max_attn and qlen is not None:
                over = over or (n * nl * nl > max_attn)
            if over and cur:
                batches.append(cur)
                cur, lmax, smax = [], 0, 0
                nl, ns = lj, sj
            cur.append(j)
            lmax, smax = nl, ns
        if cur:
            batches.append(cur)
    order = rng.permutation(len(batches))
    batches = [batches[i] for i in order]
    usable = (len(batches) // world) * world
    if usable == 0:
        raise ValueError(f"not enough rows for {world} ranks")
    return batches[:usable]


class EpochBatches(torch.utils.data.Dataset):
    """One item = one fully collated training batch (tokenized in loader workers)."""

    def __init__(self, store: RowStore, batches: list[list[int]], ser: Serializer, seed: int, epoch: int, permute: bool, start: int = 0):
        self.store, self.batches, self.ser = store, batches, ser
        self.seed, self.epoch, self.permute, self.start = seed, epoch, permute, start

    def __len__(self) -> int:
        return len(self.batches) - self.start

    def __getitem__(self, i: int):
        j = i + self.start
        rows = [self.store[k] for k in self.batches[j]]
        rng = random.Random(f"{self.seed}:{self.epoch}:{j}") if self.permute else None
        return build_batch(rows, self.ser, rng)


def _first(items):
    return items[0]


def make_loader(ds: EpochBatches, workers: int) -> torch.utils.data.DataLoader:
    kw: dict[str, Any] = dict(batch_size=1, shuffle=False, collate_fn=_first, num_workers=workers)
    if workers > 0:
        kw.update(prefetch_factor=4, persistent_workers=False)
        if sys.platform.startswith("linux"):
            kw["multiprocessing_context"] = "fork"
    return torch.utils.data.DataLoader(ds, **kw)


# ------------------------------------------------------------------- loss
def decision_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    marker_mask: torch.Tensor,
    qtype: torch.Tensor,
    conf_logit: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    logp = logits.masked_fill(~marker_mask, -1e4).log_softmax(-1)
    p = logp.exp()
    soft_ce = -(target * logp).sum(-1)
    hard = target.argmax(-1)
    hard_ce = -logp.gather(1, hard[:, None]).squeeze(1)
    brier = ((p - target) ** 2 * marker_mask).sum(-1)
    idx = torch.arange(logits.shape[1], device=logits.device)[None, :].float()
    ordinal = (p * (idx - hard[:, None].float()).abs()).sum(-1) * (qtype == QTYPE_IDS["score"]).float()
    correct = (logits.argmax(-1) == hard).float()
    conf_bce = F.binary_cross_entropy_with_logits(conf_logit, correct)
    loss = (
        weights["soft"] * soft_ce.mean()
        + weights["hard"] * hard_ce.mean()
        + weights["brier"] * brier.mean()
        + weights["ord"] * ordinal.mean()
        + weights["conf"] * conf_bce
    )
    parts = {
        "loss": float(loss.detach()),
        "soft_ce": float(soft_ce.detach().mean()),
        "brier": float(brier.detach().mean()),
        "acc": float(correct.mean()),
        "conf_bce": float(conf_bce.detach()),
    }
    return loss, parts


# ---------------------------------------------------------------- evaluate
@torch.no_grad()
def evaluate(model: GrootDecisionModel, rows: list[dict], ser: Serializer, device, batch_size: int = 16, max_rows: int | None = None) -> tuple[list[Record], dict]:
    model.eval()
    records: list[Record] = []
    subset = rows[:max_rows] if max_rows else rows
    for start in range(0, len(subset), batch_size):
        chunk = subset[start : start + batch_size]
        batch, target = build_batch(chunk, ser, None)
        batch = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch)
        logits = out["logits"].float().cpu()
        conf = torch.sigmoid(out["confidence_logit"].float()).cpu()
        for i, row in enumerate(chunk):
            k = len(row["options"])
            records.append(Record(logits=logits[i, :k].numpy(), target=np.array(row["target"][:k], dtype=np.float64), qtype=QTYPE_IDS[row["type"]], correct_conf=float(conf[i])))
    report = evaluate_records(records).to_dict()
    model.train()
    return records, report


# ------------------------------------------------------------------- main
def setup_distributed() -> tuple[int, int, torch.device]:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        cuda = torch.cuda.is_available()
        dist.init_process_group("nccl" if cuda else "gloo", timeout=datetime.timedelta(minutes=90))
        rank, world = dist.get_rank(), dist.get_world_size()
        if cuda:
            local = int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count()
            torch.cuda.set_device(local)
            return rank, world, torch.device("cuda", local)
        return rank, world, torch.device("cpu")
    if torch.cuda.is_available():
        return 0, 1, torch.device("cuda")
    if torch.backends.mps.is_available():
        return 0, 1, torch.device("mps")
    return 0, 1, torch.device("cpu")


def barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


# ---------------------------------------------------------- checkpointing
def save_checkpoint(out: Path, model: GrootDecisionModel, opt: torch.optim.Optimizer, tok, state: dict) -> None:
    """Atomic resumable checkpoint at out/ckpt (weights + optimizer + trainer state)."""
    tmp, final = out / "ckpt.tmp", out / "ckpt"
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp, tokenizer=tok)
    torch.save(opt.state_dict(), tmp / "optimizer.pt")
    (tmp / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n")
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)


def load_checkpoint(ckpt: Path, model: GrootDecisionModel, opt: torch.optim.Optimizer, device) -> dict:
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(ckpt / "model.safetensors")), strict=True)
    opt.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location=device, weights_only=True))
    return json.loads((ckpt / "trainer_state.json").read_text())


def lr_at(step: int, total: int, warmup: int, peak: float, floor: float = 0.1) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="groot-train")
    ap.add_argument("--preset", default="groot-1-s")
    ap.add_argument("--backbone", default=None, help="override backbone HF id or local path")
    ap.add_argument("--init", default=None, help="resume/initialise from a Groot checkpoint directory")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--w-soft", type=float, default=1.0)
    ap.add_argument("--w-hard", type=float, default=0.3)
    ap.add_argument("--w-brier", type=float, default=0.3)
    ap.add_argument("--w-ord", type=float, default=0.1)
    ap.add_argument("--w-conf", type=float, default=0.2)
    ap.add_argument("--no-permute", action="store_true")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-rows", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training rows (debug)")
    ap.add_argument("--tiny", action="store_true", help="random tiny encoder (tests)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--workers", type=int, default=0, help="loader workers per rank (tokenization off the training thread)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/ckpt if it exists")
    ap.add_argument("--ckpt-minutes", type=float, default=15.0, help="resumable checkpoint interval")
    ap.add_argument("--final-eval-rows", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke tests)")
    ap.add_argument("--find-unused", action="store_true", help="DDP find_unused_parameters")
    ap.add_argument("--max-tokens", type=int, default=None, help="padded question tokens per batch (rows * Lmax)")
    ap.add_argument("--max-attn", type=float, default=None, help="padded attention budget per batch (rows * Lmax^2)")
    ap.add_argument("--max-state-chunks", type=int, default=8, help="truncate states to N chunks during training")
    ap.add_argument("--save-every-eval", action="store_true", help="also save out/step_N at every evaluation")
    ap.add_argument("--spike-factor", type=float, default=1.8, help="rollback when fast loss EMA > factor * best slow EMA")
    ap.add_argument("--max-rollbacks", type=int, default=3)
    ap.add_argument("--chunk-pos", choices=["learned", "sinusoidal"], default=None, help="override the chunk-position scheme")
    ap.add_argument("--set", action="append", default=[], help="config override key=value (e.g. state_retrieval=true)")
    args = ap.parse_args(argv)

    rank, world, device = setup_distributed()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    barrier(world)

    from transformers import AutoTokenizer

    cfg = preset(args.preset, **({"backbone": args.backbone} if args.backbone else {}))
    if args.init:
        model = GrootDecisionModel.from_pretrained(args.init, device="cpu")
        tok = AutoTokenizer.from_pretrained(Path(args.init) / "tokenizer")
        cfg = model.cfg
    elif args.tiny:
        from huggingface_hub import snapshot_download

        tok = AutoTokenizer.from_pretrained(snapshot_download(cfg.backbone, allow_patterns=["tokenizer*", "special_tokens_map.json"]))
        torch.manual_seed(args.seed)
        model = GrootDecisionModel.tiny(cfg, vocab_size=len(tok), layers=2)
    else:
        tok = AutoTokenizer.from_pretrained(cfg.backbone)
        model = GrootDecisionModel.from_backbone(cfg)
    if args.chunk_pos:
        model.cfg = GrootConfig.from_dict({**model.cfg.to_dict(), "chunk_pos": args.chunk_pos})
        cfg = model.cfg
    if args.set:
        from .config import parse_overrides

        model.cfg = GrootConfig.from_dict({**model.cfg.to_dict(), **parse_overrides(args.set)})
        cfg = model.cfg
    model.sync_chunk_pos_trainable()
    model.to(device)
    if args.grad_checkpoint and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    train_cfg = GrootConfig.from_dict({**cfg.to_dict(), "max_chunks": min(cfg.max_chunks, args.max_state_chunks)})
    ser = Serializer(tok, train_cfg)

    store = RowStore(args.train, args.limit)
    qlen = slen = None
    if args.max_tokens or args.max_attn:
        t_est = time.time()
        qlen, slen = store.estimate_lengths(cfg.early_state_tokens)
        if rank == 0:
            print(json.dumps({"length_estimate_s": round(time.time() - t_est, 1), "qlen_p50": int(np.median(qlen)),
                              "qlen_p99": int(np.percentile(qlen, 99)), "qlen_max": int(qlen.max())}), flush=True)
    plan_kw = dict(qlen=qlen, slen=slen, max_tokens=args.max_tokens, max_attn=args.max_attn,
                   chunk_tokens=cfg.chunk_tokens, max_chunks=args.max_state_chunks)
    val_rows = RowStore(args.val).take(max(args.eval_rows, args.final_eval_rows)) if args.val else []
    if rank == 0:
        print(json.dumps({"params": model.count_parameters(), "train_rows": len(store), "val_rows": len(val_rows),
                          "device": str(device), "world": world, "workers": args.workers}), flush=True)

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    groups = [{"params": enc_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    state = {"step": 0, "epoch": 0, "batch_pos": 0, "best": -1.0, "lr_scale": 1.0, "rollbacks": 0}
    ckpt = out / "ckpt"
    if args.resume and ckpt.exists():
        state = load_checkpoint(ckpt, model, opt, device)
        if rank == 0:
            print(json.dumps({"resumed": state}), flush=True)

    ddp = model
    if world > 1:
        ddp = nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                                  find_unused_parameters=args.find_unused)

    batches_per_rank = len(epoch_plan(store, args.batch_size, world, args.seed, 0, **plan_kw)) // world
    steps_per_epoch = max(1, batches_per_rank // args.grad_accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    weights = {"soft": args.w_soft, "hard": args.w_hard, "brier": args.w_brier, "ord": args.w_ord, "conf": args.w_conf}
    log_path = out / "train_log.jsonl"

    def log(rec: dict) -> None:
        print(json.dumps(rec), flush=True)
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    step, epoch, best = state["step"], state["epoch"], state["best"]
    lr_scale, rollbacks = state.get("lr_scale", 1.0), state.get("rollbacks", 0)
    ema_fast = ema_slow = None
    best_slow = float("inf")
    last_gn = 0.0
    start_pos = state["batch_pos"]
    t0 = time.time()
    last_ckpt = time.time()
    win_tokens, win_rows, win_t = 0, 0, time.time()
    model.train()
    done = step >= total_steps
    while not done:
        plan = epoch_plan(store, args.batch_size, world, args.seed, epoch, **plan_kw)[rank::world]
        loader = make_loader(EpochBatches(store, plan, ser, args.seed + rank, epoch, not args.no_permute, start_pos), args.workers)
        pos = start_pos
        micro = 0
        for batch, target in loader:
            pos += 1
            batch, target = batch.to(device), target.to(device)
            win_tokens += int(batch.chunk_mask.sum()) + int(batch.q_mask.sum())
            win_rows += int(target.shape[0])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outp = ddp(batch)
            K = int(batch.q_count.max())
            marker_mask = torch.arange(K, device=device)[None, :] < batch.q_count[:, None]
            loss, parts = decision_loss(outp["logits"].float(), target, marker_mask, batch.q_type, outp["confidence_logit"].float(), weights)
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum:
                continue
            step += 1
            for g, base in zip(opt.param_groups, (args.lr, args.head_lr)):
                g["lr"] = lr_at(step, total_steps, args.warmup, base) * lr_scale
            last_gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip))
            opt.step()
            opt.zero_grad(set_to_none=True)
            # ---- divergence guard (decision identical on every rank)
            lv = torch.tensor([parts["loss"]], device=device)
            if world > 1:
                dist.all_reduce(lv)
                lv /= world
            lval = float(lv)
            ema_fast = lval if ema_fast is None else 0.9 * ema_fast + 0.1 * lval
            ema_slow = lval if ema_slow is None else 0.99 * ema_slow + 0.01 * lval
            if step > args.warmup + 200:
                best_slow = min(best_slow, ema_slow)
            diverged = not math.isfinite(lval) or (step > args.warmup + 300 and ema_fast > args.spike_factor * best_slow)
            if diverged and rollbacks < args.max_rollbacks and ckpt.exists():
                barrier(world)
                saved = load_checkpoint(ckpt, model, opt, device)
                rollbacks += 1
                lr_scale *= 0.5
                ema_fast = ema_slow = None
                if rank == 0:
                    log({"step": step, "event": "rollback", "to_step": saved["step"], "lr_scale": lr_scale, "rollbacks": rollbacks,
                         "loss_fast": None if not math.isfinite(lval) else round(lval, 4), "best_slow": round(best_slow, 4)})
                barrier(world)
                continue
            if rank == 0 and step % args.log_every == 0:
                dt = max(1e-6, time.time() - win_t)
                rec = {"step": step, "epoch": epoch, "lr": opt.param_groups[0]["lr"], "elapsed": round(time.time() - t0, 1), "grad_norm": round(last_gn, 3),
                       "tok_s": round(win_tokens * world / dt), "rows_s": round(win_rows * world / dt, 1), **parts}
                if device.type == "cuda":
                    rec["mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
                log(rec)
                win_tokens, win_rows, win_t = 0, 0, time.time()
            if val_rows and step % args.eval_every == 0:
                if rank == 0:
                    _, report = evaluate(model, val_rows, ser, device, args.batch_size, args.eval_rows)
                    log({"step": step, "val": report})
                    if report["accuracy"].get("all", -1) > best:
                        best = report["accuracy"]["all"]
                        model.save_pretrained(out / "best", tokenizer=tok)
                    if args.save_every_eval:  # keep every evaluated checkpoint for external model selection
                        model.save_pretrained(out / f"step_{step}", tokenizer=tok)
                barrier(world)
            want_ckpt = step % args.save_every == 0 or (time.time() - last_ckpt) > args.ckpt_minutes * 60
            if world > 1:
                flag = torch.tensor([1 if want_ckpt else 0], device=device)
                dist.broadcast(flag, 0)
                want_ckpt = bool(flag.item())
            if want_ckpt:
                if rank == 0:
                    save_checkpoint(out, model, opt, tok, {"step": step, "epoch": epoch, "batch_pos": pos, "best": best,
                                                          "lr_scale": lr_scale, "rollbacks": rollbacks})
                last_ckpt = time.time()
                barrier(world)
            if step >= total_steps:
                done = True
                break
        start_pos = 0
        if not done:
            epoch += 1
    barrier(world)
    if rank == 0:
        if val_rows:
            records, report = evaluate(model, val_rows, ser, device, args.batch_size, args.final_eval_rows)
            temps = fit_temperatures(records, n_types=len(cfg.qtypes))
            model.temperature.copy_(torch.tensor(temps, dtype=model.temperature.dtype, device=model.temperature.device))
            calibrated = evaluate_records(records, temps).to_dict()
            log({"final_val": report, "calibrated": calibrated, "temperatures": temps})
            (out / "eval.json").write_text(json.dumps({"final_val": report, "calibrated": calibrated}, indent=2))
        model.save_pretrained(out / "final", tokenizer=tok)
        print(f"saved {out / 'final'}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
