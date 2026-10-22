# Groot 1 — fusion layers and typed heads for ONE question, in pure Bend.
#
# The option set S = [q_token ; option_1 .. option_K] (K+1 rows of d) is a SET:
# the question token sits at position 0 and every option at position 1, so
# self-attention cannot see list order. Each fusion layer is
#   S += Attn(LN(S))  ;  S += CrossAttn(LN(S), memory)  ;  S += GeGLU(LN(S))
# then LN -> per-option scorer -> logits -> softmax(logits / T[qtype]).
# The memory K/V projections (kv) are computed once per request (groot.bend)
# and shared by every question: layer l occupies rows [l*Mlen, (l+1)*Mlen).
import Base
import ./kernels.bend as K
import ./layout.bend as L

type QSt is Type:
  QSt{w: Array<F32>, kv: Array<F32>, s: Array<F32>, h: Array<F32>, qkv: Array<F32>, att: Array<F32>, q: Array<F32>,
      xo: Array<F32>, mlp: Array<F32>, act: Array<F32>, sc: Array<F32>}

type QOut is Type:
  QOut{st: QSt, out: Array<F32>, root: Array<F32>}

def fdh() -> U32:
  U32.div(L.d(), L.fusion_heads())

def qst.new(w: Array<F32>, kv: Array<F32>, s: Array<F32>, +Ks: U32, +Mlen: U32) -> QSt:
  +d = L.d()
  +nd = (Ks * d : U32)
  QSt{w, kv, s, K.zeros(K.depth_for(nd)), K.zeros(K.depth_for((nd * 3 : U32))), K.zeros(K.depth_for(nd)), K.zeros(K.depth_for(nd)),
      K.zeros(K.depth_for(nd)), K.zeros(K.depth_for((Ks * L.fusion_inter() * 2 : U32))), K.zeros(K.depth_for((Ks * L.fusion_inter() : U32))),
      K.zeros(K.depth_for(U32.max(Mlen, Ks)))}

# ------------------------------------------------------------ re-packers

def rp_swh(kv: Array<F32>, qkv: Array<F32>, att: Array<F32>, q: Array<F32>, xo: Array<F32>, mlp: Array<F32>, act: Array<F32>, sc: Array<F32>,
           r: (Array<F32> & Array<F32>) & Array<F32>) -> QSt:
  ((s, w), h) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def rp_hwqkv(kv: Array<F32>, s: Array<F32>, att: Array<F32>, q: Array<F32>, xo: Array<F32>, mlp: Array<F32>, act: Array<F32>, sc: Array<F32>,
             r: (Array<F32> & Array<F32>) & Array<F32>) -> QSt:
  ((h, w), qkv) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def rp_att(kv: Array<F32>, s: Array<F32>, h: Array<F32>, q: Array<F32>, xo: Array<F32>, mlp: Array<F32>, act: Array<F32>,
           r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> QSt:
  (((qkv, w), att), sc) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def rp_awh(kv: Array<F32>, s: Array<F32>, qkv: Array<F32>, q: Array<F32>, xo: Array<F32>, mlp: Array<F32>, act: Array<F32>, sc: Array<F32>,
           r: (Array<F32> & Array<F32>) & Array<F32>) -> QSt:
  ((att, w), h) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def rp_hs(w: Array<F32>, kv: Array<F32>, qkv: Array<F32>, att: Array<F32>, q: Array<F32>, xo: Array<F32>, mlp: Array<F32>, act: Array<F32>, sc: Array<F32>,
          r: Array<F32> & Array<F32>) -> QSt:
  (h, s) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def rp_hwq(kv: Array<F32>, s: Array<F32>, qkv: Array<F32>, att: Array<F32>, xo: Array<F32>, mlp: Array<F32>, act: Array<F32>, sc: Array<F32>,
           r: (Array<F32> & Array<F32>) & Array<F32>) -> QSt:
  ((h, w), q) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def rp_x(w: Array<F32>, s: Array<F32>, h: Array<F32>, qkv: Array<F32>, att: Array<F32>, mlp: Array<F32>, act: Array<F32>,
         r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> QSt:
  (((q, kv), xo), sc) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def rp_xwh(kv: Array<F32>, s: Array<F32>, qkv: Array<F32>, att: Array<F32>, q: Array<F32>, mlp: Array<F32>, act: Array<F32>, sc: Array<F32>,
           r: (Array<F32> & Array<F32>) & Array<F32>) -> QSt:
  ((xo, w), h) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def rp_hwm(kv: Array<F32>, s: Array<F32>, qkv: Array<F32>, att: Array<F32>, q: Array<F32>, xo: Array<F32>, act: Array<F32>, sc: Array<F32>,
           r: (Array<F32> & Array<F32>) & Array<F32>) -> QSt:
  ((h, w), mlp) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def rp_ma(w: Array<F32>, kv: Array<F32>, s: Array<F32>, h: Array<F32>, qkv: Array<F32>, att: Array<F32>, q: Array<F32>, xo: Array<F32>, sc: Array<F32>,
          r: Array<F32> & Array<F32>) -> QSt:
  (mlp, act) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def rp_cwh(kv: Array<F32>, s: Array<F32>, qkv: Array<F32>, att: Array<F32>, q: Array<F32>, xo: Array<F32>, mlp: Array<F32>, sc: Array<F32>,
           r: (Array<F32> & Array<F32>) & Array<F32>) -> QSt:
  ((act, w), h) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

# ---------------------------------------------------------------- steps

def f.norm(+go: U32, +Ks: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  rp_swh(kv, qkv, att, q, xo, mlp, act, sc, K.layernorm(s, w, h, Ks, L.d(), 0, go, 0, False{}, 0, L.eps()))

def f.qkv(+li: U32, +Ks: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  +d = L.d()
  rp_hwqkv(kv, s, att, q, xo, mlp, act, sc, K.plinear(h, w, qkv, Ks, d, (d * 3 : U32), 0, L.f_wqkv(li), 0))

def set_pos(i: U32) -> U32:
  U32.min(i, 1)

def f.rope(+Ks: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  +d = L.d()
  QSt{w, kv, s, h, K.rope(~set_pos, K.rope(~set_pos, qkv, Ks, 0, (d * 3 : U32), L.fusion_heads(), fdh(), L.fusion_theta()), Ks, d, (d * 3 : U32), L.fusion_heads(), fdh(), L.fusion_theta()),
      att, q, xo, mlp, act, sc}

def f.att(+Ks: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  rp_att(kv, s, h, q, xo, mlp, act, K.attention(qkv, w, att, sc, Ks, 0, L.fusion_heads(), fdh(), 0, 0))

def f.wo(+li: U32, +Ks: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  rp_awh(kv, s, qkv, q, xo, mlp, act, sc, K.plinear(att, w, h, Ks, L.d(), L.d(), 0, L.f_wo(li), 0))

def f.res(+Ks: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  rp_hs(w, kv, qkv, att, q, xo, mlp, act, sc, K.axpy(h, s, 0, 0, (Ks * L.d() : U32)))

def f.q(+li: U32, +Ks: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  rp_hwq(kv, s, qkv, att, xo, mlp, act, sc, K.plinear(h, w, q, Ks, L.d(), L.d(), 0, L.f_wq(li), 0))

def f.xatt(+li: U32, +Ks: U32, +Mlen: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  rp_x(w, s, h, qkv, att, mlp, act, K.xattention(q, kv, xo, sc, Ks, Mlen, L.fusion_heads(), fdh(), (li * Mlen * L.d() * 2 : U32)))

def f.cwo(+li: U32, +Ks: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  rp_xwh(kv, s, qkv, att, q, mlp, act, sc, K.plinear(xo, w, h, Ks, L.d(), L.d(), 0, L.f_cwo(li), 0))

def f.wi(+li: U32, +Ks: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  rp_hwm(kv, s, qkv, att, q, xo, act, sc, K.plinear(h, w, mlp, Ks, L.d(), (L.fusion_inter() * 2 : U32), 0, L.f_wi(li), 0))

def f.geglu(+Ks: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  rp_ma(w, kv, s, h, qkv, att, q, xo, sc, K.geglu(mlp, act, Ks, L.fusion_inter(), 0, 0))

def f.wmo(+li: U32, +Ks: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  rp_cwh(kv, s, qkv, att, q, xo, mlp, sc, K.plinear(act, w, h, Ks, L.fusion_inter(), L.d(), 0, L.f_wmo(li), 0))

def flayer(+li: U32, +Ks: U32, +Mlen: U32, st: QSt) -> QSt:
  s1 = f.norm(L.f_attn_norm(li), Ks, st)
  s2 = f.qkv(li, Ks, s1)
  s3 = f.rope(Ks, s2)
  s4 = f.att(Ks, s3)
  s5 = f.wo(li, Ks, s4)
  s6 = f.res(Ks, s5)
  s7 = f.norm(L.f_cross_norm(li), Ks, s6)
  s8 = f.q(li, Ks, s7)
  s9 = f.xatt(li, Ks, Mlen, s8)
  s10 = f.cwo(li, Ks, s9)
  s11 = f.res(Ks, s10)
  s12 = f.norm(L.f_mlp_norm(li), Ks, s11)
  s13 = f.wi(li, Ks, s12)
  s14 = f.geglu(Ks, s13)
  s15 = f.wmo(li, Ks, s14)
  f.res(Ks, s15)

def flayers.go(k: Nat, +li: U32, +Ks: U32, +Mlen: U32, st: QSt) -> QSt:
  match k:
    case 0n:
      st
    case 1n+p:
      flayers.go(p, (li + 1 : U32), Ks, Mlen, flayer(li, Ks, Mlen, st))

# all fusion layers, then the final norm into h

def fusion(+Ks: U32, +Mlen: U32, st: QSt) -> QSt:
  f.norm(L.g_final_norm(), Ks, flayers.go(U32.to_nat(L.fusion_layers()), 0, Ks, Mlen, st))

# ---------------------------------------------------------------- heads
# temperature per question type (0 choice, 1 score, 2 noul)

def temp(qtype: U32) -> F32:
  match qtype:
    case 0:
      L.temp_choice()
    case 1:
      L.temp_score()
    case _:
      L.temp_noul()

# scorer over the K option rows of h (rows 1..K): att <- LN_b(h[1..]), mlp <- gelu(att W1^T + b1), lg <- mlp W2^T + b2

def sc.pack1(kv: Array<F32>, s: Array<F32>, qkv: Array<F32>, q: Array<F32>, xo: Array<F32>, mlp: Array<F32>, act: Array<F32>, sc: Array<F32>,
             r: (Array<F32> & Array<F32>) & Array<F32>) -> QSt:
  ((h, w), att) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def scorer.ln(+Kn: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  sc.pack1(kv, s, qkv, q, xo, mlp, act, sc, K.layernorm(h, w, att, Kn, L.d(), L.d(), L.sc_ln_w(), L.sc_ln_b(), True{}, 0, L.eps()))

def sc.pack2(kv: Array<F32>, s: Array<F32>, h: Array<F32>, qkv: Array<F32>, q: Array<F32>, xo: Array<F32>, act: Array<F32>, sc: Array<F32>,
             r: (Array<F32> & Array<F32>) & Array<F32>) -> QSt:
  ((att, w), mlp) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def scorer.l1(+Kn: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  sc.pack2(kv, s, h, qkv, q, xo, act, sc, K.plinear(att, w, mlp, Kn, L.d(), L.scorer_hidden(), 0, L.sc_w1(), 0))

def sc.pack3(kv: Array<F32>, s: Array<F32>, h: Array<F32>, qkv: Array<F32>, att: Array<F32>, q: Array<F32>, xo: Array<F32>, act: Array<F32>, sc: Array<F32>,
             r: Array<F32> & Array<F32>) -> QSt:
  (w, mlp) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def scorer.b1(+Kn: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  sc.pack3(kv, s, h, qkv, att, q, xo, act, sc, K.add_bias(w, mlp, Kn, L.scorer_hidden(), L.sc_b1(), 0))

def scorer.gelu(+Kn: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  QSt{w, kv, s, h, qkv, att, q, xo, K.gelu_rows(mlp, 0, (Kn * L.scorer_hidden() : U32)), act, sc}

# logits into sc[0..K)

def sc.pack4(kv: Array<F32>, s: Array<F32>, h: Array<F32>, qkv: Array<F32>, att: Array<F32>, q: Array<F32>, xo: Array<F32>, act: Array<F32>,
             r: (Array<F32> & Array<F32>) & Array<F32>) -> QSt:
  ((mlp, w), sc) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def scorer.l2(+Kn: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  sc.pack4(kv, s, h, qkv, att, q, xo, act, K.linear(mlp, w, sc, Kn, L.scorer_hidden(), 1, 0, L.sc_w2(), 0))

def sc.pack5(kv: Array<F32>, s: Array<F32>, h: Array<F32>, qkv: Array<F32>, att: Array<F32>, q: Array<F32>, xo: Array<F32>, mlp: Array<F32>, act: Array<F32>,
             r: Array<F32> & Array<F32>) -> QSt:
  (w, sc) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}

def scorer.b2(+Kn: U32, st: QSt) -> QSt:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  sc.pack5(kv, s, h, qkv, att, q, xo, mlp, act, K.add_bias(w, sc, Kn, 1, L.sc_b2(), 0))

# after `fusion`: logits of the Kn options land in sc[0..Kn)

def scorer(+Kn: U32, st: QSt) -> QSt:
  s1 = scorer.ln(Kn, st)
  s2 = scorer.l1(Kn, s1)
  s3 = scorer.b1(Kn, s2)
  s4 = scorer.gelu(Kn, s3)
  s5 = scorer.l2(Kn, s4)
  scorer.b2(Kn, s5)

# probabilities: softmax(logits / T) written into `probs[po..po+Kn)`; logits copied to lg[lo..)

def probs.pack(w: Array<F32>, kv: Array<F32>, s: Array<F32>, h: Array<F32>, qkv: Array<F32>, att: Array<F32>, q: Array<F32>, xo: Array<F32>,
               mlp: Array<F32>, act: Array<F32>, r: Array<F32> & Array<F32>) -> QSt & Array<F32>:
  (sc, out) = r
  (QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}, out)

# out[po..po+Kn) <- softmax(sc[0..Kn) / T), out[lo..lo+Kn) <- raw logits

def probs.max.go(k: Nat, +i: U32, +m: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  match k:
    case 0n:
      (a, v) = r
      (a, F32.max(m, v))
    case 1n+p:
      (a, v) = r
      probs.max.go(p, (i + 1 : U32), F32.max(m, v), Array.get(F32, a, i))

def probs.exp.step(+i: U32, +m: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  (a, v) = r
  +e = F32.exp((v - m : F32))
  (Array.set(F32, a, i, e), e)

def probs.exp.go(k: Nat, +i: U32, +m: F32, +acc: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  match k:
    case 0n:
      (a, e) = r
      (a, (acc + e : F32))
    case 1n+p:
      (a, e) = r
      probs.exp.go(p, (i + 1 : U32), m, (acc + e : F32), probs.exp.step(i, m, Array.get(F32, a, i)))

def probs.sm3(+po: U32, +n: U32, r: Array<F32> & F32) -> Array<F32>:
  (a, total) = r
  K.scale(a, po, n, (1.0 / total : F32))

def probs.sm2(+po: U32, +n: U32, r: Array<F32> & F32) -> Array<F32>:
  (a, +m) = r
  probs.sm3(po, n, probs.exp.go(U32.to_nat(U32.sub(n, 1)), (po + 1 : U32), m, 0.0, probs.exp.step(po, m, Array.get(F32, a, po))))

# ------------------------------------------------------- confidence head
# features from probabilities out[po..po+Kn): p_max, margin (p1 - p2), normalised entropy, K/255

def probs.softmax_at(out: Array<F32>, +po: U32, +n: U32) -> Array<F32>:
  probs.sm2(po, n, probs.max.go(U32.to_nat(U32.sub(n, 1)), (po + 1 : U32), F32.neg(1000000000000.0), Array.get(F32, out, po)))

def probs.soft2(+Kn: U32, +po: U32, r: Array<F32> & Array<F32>, +t: F32) -> Array<F32> & Array<F32>:
  (sc, out) = r
  (sc, probs.softmax_at(K.scale(out, po, Kn, (1.0 / t : F32)), po, Kn))

# softmax in place over out[po..po+n): shift by max, exp, normalise

def probs.soft(+Kn: U32, +po: U32, +t: F32, r: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  (sc, out) = r
  probs.soft2(Kn, po, K.copy(sc, K.scale(out, po, 0, 1.0), 0, po, Kn), t)

def probs.fin(+Kn: U32, +po: U32, +lo: U32, +t: F32, st: QSt, out: Array<F32>) -> QSt & Array<F32>:
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  probs.pack(w, kv, s, h, qkv, att, q, xo, mlp, act, probs.soft(Kn, po, t, K.copy(sc, out, 0, lo, Kn)))

# copy logits scaled by 1/T into `act`-free scratch: we reuse the `out` row itself as the softmax buffer

# conf_in row (2d+4) assembled in q[0..): [root ; h row 0 ; feats]
# feats.go accumulates p1/p2/entropy over out[po..po+Kn) and, in the base case,
# writes the four features into q[d2..d2+4) directly (the checker rejects '+'
# binders on nested pair destructures, so the pack must happen where p1/p2/ent
# are still '+' parameters).

def feats.go(k: Nat, +i: U32, +p1: F32, +p2: F32, +ent: F32, +Kn: U32, +d2: U32, q: Array<F32>, r: Array<F32> & F32) -> Array<F32> & Array<F32>:
  match k:
    case 0n:
      (a, +v) = r
      +e = (ent - (v * F32.log(F32.max(v, 0.000000001))) : F32)
      +np1 = F32.max(p1, v)
      +np2 = K.pickf((v > p1 : F32), p1, F32.max(p2, v))
      +cnt = U32.to_f32(U32.max(Kn, 2))
      (a, Array.set(F32, Array.set(F32, Array.set(F32, Array.set(F32, q, d2, np1), (d2 + 1 : U32), (np1 - np2 : F32)), (d2 + 2 : U32), (e / F32.log(cnt) : F32)), (d2 + 3 : U32), (cnt / 255.0 : F32)))
    case 1n+p:
      (a, +v) = r
      +e = (ent - (v * F32.log(F32.max(v, 0.000000001))) : F32)
      +np1 = F32.max(p1, v)
      +np2 = K.pickf((v > p1 : F32), p1, F32.max(p2, v))
      feats.go(p, (i + 1 : U32), np1, np2, e, Kn, d2, q, Array.get(F32, a, i))

def conf.feats(+Kn: U32, +po: U32, q: Array<F32>, out: Array<F32>) -> Array<F32> & Array<F32>:
  feats.go(U32.to_nat(U32.sub(Kn, 1)), (po + 1 : U32), 0.0, 0.0, 0.0, Kn, (L.d() * 2 : U32), q, Array.get(F32, out, po))

def conf.c1(root: Array<F32>, q: Array<F32>, +d: U32) -> Array<F32> & Array<F32>:
  K.copy(root, q, 0, 0, d)

def conf.c2.pack(root: Array<F32>, r: Array<F32> & Array<F32>) -> (Array<F32> & Array<F32>) & Array<F32>:
  (h, q) = r
  ((root, h), q)

# act[0..ch) <- gelu(q W1^T + b1) ; logit <- act W2^T + b2

def conf.c2(h: Array<F32>, r: Array<F32> & Array<F32>, +d: U32) -> (Array<F32> & Array<F32>) & Array<F32>:
  (root, q) = r
  conf.c2.pack(root, K.copy(h, q, 0, d, d))

def conf.l1(w: Array<F32>, q: Array<F32>, act: Array<F32>, +d2: U32) -> (Array<F32> & Array<F32>) & Array<F32>:
  K.linear(q, w, act, 1, (d2 + 4 : U32), L.conf_hidden(), 0, L.conf_w1(), 0)

def conf.b1.pack(q: Array<F32>, r: Array<F32> & Array<F32>) -> (Array<F32> & Array<F32>) & Array<F32>:
  (w, act) = r
  ((q, w), K.gelu_rows(act, 0, L.conf_hidden()))

# logit written to sc[Kn] (just after the logits), returned in the pair

def conf.b1(r: (Array<F32> & Array<F32>) & Array<F32>) -> (Array<F32> & Array<F32>) & Array<F32>:
  ((q, w), act) = r
  conf.b1.pack(q, K.add_bias(w, act, 1, L.conf_hidden(), L.conf_b1(), 0))

def conf.l2.pack(q: Array<F32>, r: (Array<F32> & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  ((act, w), sc) = r
  (((q, w), act), sc)

def conf.l2(+Kn: U32, sc: Array<F32>, r: (Array<F32> & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  ((q, w), act) = r
  conf.l2.pack(q, K.linear(act, w, sc, 1, L.conf_hidden(), 1, 0, L.conf_w2(), Kn))

def conf.b2.pack(q: Array<F32>, act: Array<F32>, r: Array<F32> & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (w, sc) = r
  (((q, w), act), sc)

# ----------------------------------------------------------- decide one

def conf.b2(+Kn: U32, r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (((q, w), act), sc) = r
  conf.b2.pack(q, act, K.add_bias(w, sc, 1, 1, L.conf_b2(), Kn))

def decide.conf4(kv: Array<F32>, s: Array<F32>, qkv: Array<F32>, att: Array<F32>, xo: Array<F32>, mlp: Array<F32>, out: Array<F32>,
                 root: Array<F32>, h: Array<F32>, r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> QOut:
  (((q, w), act), sc) = r
  QOut{QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc}, out, root}

# Decide one question whose set S is already in st.s (Kn+1 rows):
# probabilities -> out[po..po+Kn), raw logits -> out[lo..lo+Kn), confidence logit -> st.sc[Kn]

def decide.conf3(+Kn: U32, kv: Array<F32>, s: Array<F32>, qkv: Array<F32>, att: Array<F32>, xo: Array<F32>, mlp: Array<F32>, out: Array<F32>,
                 r: (Array<F32> & Array<F32>) & Array<F32>, w: Array<F32>, act: Array<F32>, sc: Array<F32>) -> QOut:
  ((root, h), q) = r
  c1 = conf.l1(w, q, act, (L.d() * 2 : U32))
  c2 = conf.b1(c1)
  c3 = conf.l2(Kn, sc, c2)
  decide.conf4(kv, s, qkv, att, xo, mlp, out, root, h, conf.b2(Kn, c3))

def decide.conf2(+Kn: U32, kv: Array<F32>, s: Array<F32>, qkv: Array<F32>, att: Array<F32>, xo: Array<F32>, mlp: Array<F32>,
                 r: Array<F32> & Array<F32>, root: Array<F32>, h: Array<F32>, w: Array<F32>, act: Array<F32>, sc: Array<F32>) -> QOut:
  (out, q) = r
  decide.conf3(Kn, kv, s, qkv, att, xo, mlp, out, conf.c2(h, conf.c1(root, q, L.d()), L.d()), w, act, sc)

def decide.conf(+Kn: U32, +po: U32, r: QSt & Array<F32>, root: Array<F32>) -> QOut:
  (st, out) = r
  QSt{w, kv, s, h, qkv, att, q, xo, mlp, act, sc} = st
  decide.conf2(Kn, kv, s, qkv, att, xo, mlp, conf.feats(Kn, po, q, out), root, h, w, act, sc)

def decide(+Kn: U32, +Mlen: U32, qtype: U32, +po: U32, +lo: U32, st: QSt, out: Array<F32>, root: Array<F32>) -> QOut:
  s1 = fusion((Kn + 1 : U32), Mlen, st)
  s2 = scorer(Kn, s1)
  decide.conf(Kn, po, probs.fin(Kn, po, lo, temp(qtype), s2, out), root)
