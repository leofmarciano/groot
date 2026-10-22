# Groot 1 — ModernBERT-style encoder forward pass for ONE sequence, in pure Bend.
#
# Mirrors HF `ModernBertModel` (as used by mmBERT): token embeddings + LayerNorm,
# then per layer: pre-norm (identity on layer 0) -> Wqkv -> RoPE on q,k ->
# attention (global every `global_every` layers, else a sliding window of
# ±window tokens) -> Wo -> residual -> pre-norm -> Wi -> GeGLU -> Wo -> residual;
# finally a LayerNorm. All weights live in one flat F32 array at the offsets
# compiled into layout.bend.
#
# Positions and attention blocks come from an auxiliary U32 array `aux`:
# position of token i = aux[po + i]; block of token i = aux[bo + i]; key j is
# visible to query i iff block j == 0 or block j == block i. State chunks use
# identity positions and block 0 (`encode`); question sequences use the
# shared-base positions and option blocks stored in the request (`encode_x`).
import Base
import ./kernels.bend as K
import ./layout.bend as L

# Working buffers for a sequence of n tokens (flat, row-major)
type Bufs is Type:
  Bufs{x: Array<F32>, h: Array<F32>, qkv: Array<F32>, att: Array<F32>, mlp: Array<F32>, act: Array<F32>, s: Array<F32>}

def identity_off() -> U32:
  4294967295

def bufs.new(+n: U32) -> Bufs:
  +d = L.d()
  +nd = (n * d : U32)
  Bufs{K.zeros(K.depth_for(nd)), K.zeros(K.depth_for(nd)), K.zeros(K.depth_for((nd * 3 : U32))), K.zeros(K.depth_for(nd)),
       K.zeros(K.depth_for((n * L.inter() * 2 : U32))), K.zeros(K.depth_for((n * L.inter() : U32))), K.zeros(K.depth_for(n))}

# ------------------------------------------------------------ embeddings

def embed.pack(ids: Array<U32>, r: Array<F32> & Array<F32>) -> (Array<F32> & Array<U32>) & Array<F32>:
  (w, x) = r
  ((w, ids), x)

def embed.row(+i: U32, w: Array<F32>, x: Array<F32>, r: Array<U32> & U32) -> (Array<F32> & Array<U32>) & Array<F32>:
  (ids, id) = r
  +d = L.d()
  embed.pack(ids, K.copy(w, x, (L.tok_emb() + (id * d) : U32), (i * d : U32), d))

def embed.go(n: Nat, +i: U32, +ido: U32, r: (Array<F32> & Array<U32>) & Array<F32>) -> (Array<F32> & Array<U32>) & Array<F32>:
  match n:
    case 0n:
      r
    case 1n+p:
      ((w, ids), x) = r
      embed.go(p, (i + 1 : U32), ido, embed.row(i, w, x, Array.get(U32, ids, (ido + i : U32))))

# x <- tok_emb[ids[ido .. ido+n)]  (n rows)

def embed(w: Array<F32>, ids: Array<U32>, x: Array<F32>, +ido: U32, +n: U32) -> (Array<F32> & Array<U32>) & Array<F32>:
  embed.go(U32.to_nat(n), 0, ido, ((w, ids), x))

# ------------------------------------------------------------ re-packers
# Each kernel returns the arrays it touched; these put them back into Bufs
# beside the weights handle (Bend destructures parameters only).

def pk_xwh(qkv: Array<F32>, att: Array<F32>, mlp: Array<F32>, act: Array<F32>, s: Array<F32>, r: (Array<F32> & Array<F32>) & Array<F32>) -> Array<F32> & Bufs:
  ((x, w), h) = r
  (w, Bufs{x, h, qkv, att, mlp, act, s})

def pk_hwq(x: Array<F32>, att: Array<F32>, mlp: Array<F32>, act: Array<F32>, s: Array<F32>, r: (Array<F32> & Array<F32>) & Array<F32>) -> Array<F32> & Bufs:
  ((h, w), qkv) = r
  (w, Bufs{x, h, qkv, att, mlp, act, s})

def pk_att(x: Array<F32>, h: Array<F32>, mlp: Array<F32>, act: Array<F32>, r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> Array<F32> & Bufs:
  (((qkv, w), att), s) = r
  (w, Bufs{x, h, qkv, att, mlp, act, s})

def pk_awh(x: Array<F32>, qkv: Array<F32>, mlp: Array<F32>, act: Array<F32>, s: Array<F32>, r: (Array<F32> & Array<F32>) & Array<F32>) -> Array<F32> & Bufs:
  ((att, w), h) = r
  (w, Bufs{x, h, qkv, att, mlp, act, s})

def pk_hx(w: Array<F32>, qkv: Array<F32>, att: Array<F32>, mlp: Array<F32>, act: Array<F32>, s: Array<F32>, r: Array<F32> & Array<F32>) -> Array<F32> & Bufs:
  (h, x) = r
  (w, Bufs{x, h, qkv, att, mlp, act, s})

def pk_hwm(x: Array<F32>, qkv: Array<F32>, att: Array<F32>, act: Array<F32>, s: Array<F32>, r: (Array<F32> & Array<F32>) & Array<F32>) -> Array<F32> & Bufs:
  ((h, w), mlp) = r
  (w, Bufs{x, h, qkv, att, mlp, act, s})

def pk_ma(w: Array<F32>, x: Array<F32>, h: Array<F32>, qkv: Array<F32>, att: Array<F32>, s: Array<F32>, r: Array<F32> & Array<F32>) -> Array<F32> & Bufs:
  (mlp, act) = r
  (w, Bufs{x, h, qkv, att, mlp, act, s})

def pk_cwh(x: Array<F32>, qkv: Array<F32>, att: Array<F32>, mlp: Array<F32>, s: Array<F32>, r: (Array<F32> & Array<F32>) & Array<F32>) -> Array<F32> & Bufs:
  ((act, w), h) = r
  (w, Bufs{x, h, qkv, att, mlp, act, s})

def copy.pack(w: Array<F32>, r: Array<F32> & Array<F32>) -> (Array<F32> & Array<F32>) & Array<F32>:
  (x, h) = r
  ((x, w), h)

# h <- LayerNorm(x) with gamma at `go`, or h <- x when the norm is the identity

def norm_or_copy(ident: Bool, +n: U32, +go: U32, x: Array<F32>, w: Array<F32>, h: Array<F32>) -> (Array<F32> & Array<F32>) & Array<F32>:
  match ident:
    case True{}:
      copy.pack(w, K.copy(x, h, 0, 0, (n * L.d() : U32)))
    case False{}:
      K.layernorm(x, w, h, n, L.d(), 0, go, 0, False{}, 0, L.eps())

# ---------------------------------------------------------------- steps

def step.anorm(+li: U32, +n: U32, w: Array<F32>, b: Bufs) -> Array<F32> & Bufs:
  Bufs{x, h, qkv, att, mlp, act, s} = b
  +go = L.attn_norm(li)
  pk_xwh(qkv, att, mlp, act, s, norm_or_copy(U32.is_eq(go, identity_off()), n, go, x, w, h))

def step.qkv(+li: U32, +n: U32, r: Array<F32> & Bufs) -> Array<F32> & Bufs:
  (w, b) = r
  Bufs{x, h, qkv, att, mlp, act, s} = b
  +d = L.d()
  pk_hwq(x, att, mlp, act, s, K.plinear(h, w, qkv, n, d, (d * 3 : U32), 0, L.wqkv(li), 0))

def pk_q(w: Array<F32>, x: Array<F32>, h: Array<F32>, att: Array<F32>, mlp: Array<F32>, act: Array<F32>, s: Array<F32>, qkv: Array<F32>) -> Array<F32> & Bufs:
  (w, Bufs{x, h, qkv, att, mlp, act, s})

def step.wo(+li: U32, +n: U32, r: Array<F32> & Bufs) -> Array<F32> & Bufs:
  (w, b) = r
  Bufs{x, h, qkv, att, mlp, act, s} = b
  +d = L.d()
  pk_awh(x, qkv, mlp, act, s, K.plinear(att, w, h, n, d, d, 0, L.wo(li), 0))

# x <- x + h

def step.residual(+n: U32, r: Array<F32> & Bufs) -> Array<F32> & Bufs:
  (w, b) = r
  Bufs{x, h, qkv, att, mlp, act, s} = b
  pk_hx(w, qkv, att, mlp, act, s, K.axpy(h, x, 0, 0, (n * L.d() : U32)))

def step.mnorm(+li: U32, +n: U32, r: Array<F32> & Bufs) -> Array<F32> & Bufs:
  (w, b) = r
  Bufs{x, h, qkv, att, mlp, act, s} = b
  pk_xwh(qkv, att, mlp, act, s, K.layernorm(x, w, h, n, L.d(), 0, L.mlp_norm(li), 0, False{}, 0, L.eps()))

def step.wi(+li: U32, +n: U32, r: Array<F32> & Bufs) -> Array<F32> & Bufs:
  (w, b) = r
  Bufs{x, h, qkv, att, mlp, act, s} = b
  pk_hwm(x, qkv, att, act, s, K.plinear(h, w, mlp, n, L.d(), (L.inter() * 2 : U32), 0, L.wi(li), 0))

def step.geglu(+n: U32, r: Array<F32> & Bufs) -> Array<F32> & Bufs:
  (w, b) = r
  Bufs{x, h, qkv, att, mlp, act, s} = b
  pk_ma(w, x, h, qkv, att, s, K.geglu(mlp, act, n, L.inter(), 0, 0))

def step.wmo(+li: U32, +n: U32, r: Array<F32> & Bufs) -> Array<F32> & Bufs:
  (w, b) = r
  Bufs{x, h, qkv, att, mlp, act, s} = b
  pk_cwh(x, qkv, att, mlp, s, K.plinear(act, w, h, n, L.inter(), L.d(), 0, L.wmo(li), 0))

# ------------------------------------------------- positions / blocks aux

def aux.fill(k: Nat, +i: U32, a: Array<U32>) -> Array<U32>:
  match k:
    case 0n:
      a
    case 1n+p:
      aux.fill(p, (i + 1 : U32), Array.set(U32, a, i, i))

# identity positions at [0, n), block 0 at [n, 2n)

def aux.new(+n: U32) -> Array<U32>:
  aux.fill(U32.to_nat(n), 0, Array.new(U32, K.depth_for((n * 2 : U32)), 0))

# ------------------------------------------------------------- one layer

def theta_of(+li: U32) -> F32:
  K.pickf(U32.is_zero(U32.mod(li, L.global_every())), L.theta(), L.local_theta())

def lay.rope3(x: Array<F32>, h: Array<F32>, att: Array<F32>, mlp: Array<F32>, act: Array<F32>, s: Array<F32>, w: Array<F32>,
              r: Array<F32> & Array<U32>) -> (Array<F32> & Bufs) & Array<U32>:
  (qkv, aux) = r
  ((w, Bufs{x, h, qkv, att, mlp, act, s}), aux)

def lay.rope2(+n: U32, +th: F32, +po: U32, x: Array<F32>, h: Array<F32>, att: Array<F32>, mlp: Array<F32>, act: Array<F32>, s: Array<F32>, w: Array<F32>,
              r: Array<F32> & Array<U32>) -> (Array<F32> & Bufs) & Array<U32>:
  (qkv, aux) = r
  +d = L.d()
  lay.rope3(x, h, att, mlp, act, s, w, K.rope_p(qkv, aux, n, d, (d * 3 : U32), L.heads(), L.dh(), th, po))

# RoPE on the q block (offset 0) and the k block (offset d) of every row, positions from aux

def lay.rope(+li: U32, +n: U32, +po: U32, aux: Array<U32>, r: Array<F32> & Bufs) -> (Array<F32> & Bufs) & Array<U32>:
  (w, b) = r
  Bufs{x, h, qkv, att, mlp, act, s} = b
  +d = L.d()
  +th = theta_of(li)
  lay.rope2(n, th, po, x, h, att, mlp, act, s, w, K.rope_p(qkv, aux, n, 0, (d * 3 : U32), L.heads(), L.dh(), th, po))

def lay.att2(x: Array<F32>, h: Array<F32>, mlp: Array<F32>, act: Array<F32>,
             r: (((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) & Array<U32>) -> (Array<F32> & Bufs) & Array<U32>:
  ((((qkv, w), att), s), aux) = r
  ((w, Bufs{x, h, qkv, att, mlp, act, s}), aux)

def lay.att(+li: U32, +n: U32, +bo: U32, r: (Array<F32> & Bufs) & Array<U32>) -> (Array<F32> & Bufs) & Array<U32>:
  ((w, b), aux) = r
  Bufs{x, h, qkv, att, mlp, act, s} = b
  +win = K.picku(U32.is_zero(U32.mod(li, L.global_every())), 0, L.window())
  lay.att2(x, h, mlp, act, K.attention_b(qkv, w, att, s, aux, n, 0, L.heads(), L.dh(), 0, win, bo))

def lay.rest(+li: U32, +n: U32, r: (Array<F32> & Bufs) & Array<U32>) -> (Array<F32> & Bufs) & Array<U32>:
  (wb, aux) = r
  r1 = step.wo(li, n, wb)
  r2 = step.residual(n, r1)
  r3 = step.mnorm(li, n, r2)
  r4 = step.wi(li, n, r3)
  r5 = step.geglu(n, r4)
  r6 = step.wmo(li, n, r5)
  (step.residual(n, r6), aux)

def layer(+li: U32, +n: U32, +po: U32, +bo: U32, r: (Array<F32> & Bufs) & Array<U32>) -> (Array<F32> & Bufs) & Array<U32>:
  ((w, b), aux) = r
  r1 = step.anorm(li, n, w, b)
  r2 = step.qkv(li, n, r1)
  r3 = lay.rope(li, n, po, aux, r2)
  lay.rest(li, n, lay.att(li, n, bo, r3))

def layers.go(k: Nat, +li: U32, +n: U32, +po: U32, +bo: U32, r: (Array<F32> & Bufs) & Array<U32>) -> (Array<F32> & Bufs) & Array<U32>:
  match k:
    case 0n:
      r
    case 1n+p:
      layers.go(p, (li + 1 : U32), n, po, bo, layer(li, n, po, bo, r))

# ------------------------------------------------------------- forward

def emb.norm.fin(ids: Array<U32>, h: Array<F32>, qkv: Array<F32>, att: Array<F32>, mlp: Array<F32>, act: Array<F32>, s: Array<F32>,
                 r: (Array<F32> & Array<F32>) & Array<F32>) -> (Array<F32> & Array<U32>) & Bufs:
  ((x0, w), x) = r
  ((w, ids), Bufs{x, h, qkv, att, mlp, act, s})

def emb.norm.pack(ids: Array<U32>, b: Bufs, +n: U32, x0: Array<F32>, w: Array<F32>) -> (Array<F32> & Array<U32>) & Bufs:
  Bufs{x, h, qkv, att, mlp, act, s} = b
  emb.norm.fin(ids, h, qkv, att, mlp, act, s, K.layernorm(x0, w, x, n, L.d(), 0, L.emb_norm(), 0, False{}, 0, L.eps()))

def emb.norm(+n: U32, r: (Array<F32> & Array<U32>) & Array<F32>) -> (Array<F32> & Array<U32>) & Bufs:
  ((w, ids), x) = r
  emb.norm.pack(ids, bufs.new(n), n, x, w)

def run.layers.pack(ids: Array<U32>, r: (Array<F32> & Bufs) & Array<U32>) -> ((Array<F32> & Array<U32>) & Bufs) & Array<U32>:
  ((w, b), aux) = r
  (((w, ids), b), aux)

def run.layers(+n: U32, +po: U32, +bo: U32, aux: Array<U32>, r: (Array<F32> & Array<U32>) & Bufs) -> ((Array<F32> & Array<U32>) & Bufs) & Array<U32>:
  ((w, ids), b) = r
  run.layers.pack(ids, layers.go(U32.to_nat(L.layers()), 0, n, po, bo, ((w, b), aux)))

def final.norm.pack(ids: Array<U32>, aux: Array<U32>, r: (Array<F32> & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<U32>) & Array<F32>) & Array<U32>:
  ((x, w), h) = r
  (((w, ids), h), aux)

def final.norm(+n: U32, r: ((Array<F32> & Array<U32>) & Bufs) & Array<U32>) -> ((Array<F32> & Array<U32>) & Array<F32>) & Array<U32>:
  (((w, ids), b), aux) = r
  Bufs{x, h, qkv, att, mlp, act, s} = b
  final.norm.pack(ids, aux, K.layernorm(x, w, h, n, L.d(), 0, L.final_norm(), 0, False{}, 0, L.eps()))

# Encode the n token ids at ids[ido..ido+n) with positions aux[po..] and blocks aux[bo..]
# -> hidden states (n x d) after the final norm; ids and aux are handed back.

def encode_x(w: Array<F32>, ids: Array<U32>, aux: Array<U32>, +ido: U32, +n: U32, +po: U32, +bo: U32) -> ((Array<F32> & Array<U32>) & Array<F32>) & Array<U32>:
  e1 = embed(w, ids, K.zeros(K.depth_for((n * L.d() : U32))), ido, n)
  e2 = emb.norm(n, e1)
  final.norm(n, run.layers(n, po, bo, aux, e2))

def encode.drop(r: ((Array<F32> & Array<U32>) & Array<F32>) & Array<U32>) -> (Array<F32> & Array<U32>) & Array<F32>:
  (wih, aux) = r
  wih

# State chunks: identity positions, one block (plain bidirectional attention).

def encode(w: Array<F32>, ids: Array<U32>, +ido: U32, +n: U32) -> (Array<F32> & Array<U32>) & Array<F32>:
  encode.drop(encode_x(w, ids, aux.new(n), ido, n, 0, n))
