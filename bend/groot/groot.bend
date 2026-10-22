# Groot 1 — the Tree-Reduce Decision Transformer, forward pass in pure Bend.
#
#   chunks  --(parallel fork)-->  Encoder  -->  token memory + chunk leaves
#   leaves  --(fork/join tree)--> TreeReduce --> multi-scale nodes + root
#   memory  -->  K/V projections, computed ONCE per request
#   options --(parallel fork)-->  Encoder  -->  marker + CLS rows (pointwise)
#   questions --(parallel fork)--> set fusion over [q ; options] x memory -> probabilities
#
# Input: the weights array (layout.bend offsets) and a u32 request blob (see
# groot1/bendio.py). Output: [probabilities(n_opt) ; confidence logits(n_q) ; logits(n_opt)].
import Base
import ./kernels.bend as K
import ./layout.bend as L
import ./encoder.bend as E
import ./fusion.bend as F

# ------------------------------------------------------------------ types
type Hs is Type:
  HNil{}
  HLeaf{h: Array<F32>, n: U32}
  HNode{l: Hs, r: Hs}

# memory assembly state: mem rows, leaf vectors, weights, next row, chunk index
type MSt is Type:
  MSt{mem: Array<F32>, lv: Array<F32>, w: Array<F32>, row: U32, c: U32}

# tree reduce state
type TSt is Type:
  TSt{cur: Array<F32>, nxt: Array<F32>, vc: Array<U32>, vn: Array<U32>, mem: Array<F32>, w: Array<F32>, t4: Array<F32>, nd: Array<F32>, row: U32}

# per-request shared arrays for the question stage
type GSt is Type:
  GSt{w: Array<F32>, b: Array<U32>, kv: Array<F32>, mk: Array<F32>, cl: Array<F32>, root: Array<F32>, out: Array<F32>}

# --------------------------------------------------------- blob offsets

type QOff is Data:
  QOff{qo: U32, qlen: U32, kmax: U32, ids: U32, pos: U32, blk: U32, mk: U32}

def off.lens() -> U32:
  6

def off.q(+nch: U32) -> U32:
  (6 + nch : U32)

def off.cids(+nch: U32, +nq: U32) -> U32:
  (off.q(nch) + (nq * 4) : U32)

def off.qids(+nch: U32, +nq: U32, +clen: U32) -> U32:
  (off.cids(nch, nq) + (nch * clen) : U32)

def off.qpos(+nch: U32, +nq: U32, +clen: U32, +qlen: U32) -> U32:
  (off.qids(nch, nq, clen) + (nq * qlen) : U32)

def off.qblk(+nch: U32, +nq: U32, +clen: U32, +qlen: U32) -> U32:
  (off.qpos(nch, nq, clen, qlen) + (nq * qlen) : U32)

def off.qmk(+nch: U32, +nq: U32, +clen: U32, +qlen: U32) -> U32:
  (off.qblk(nch, nq, clen, qlen) + (nq * qlen) : U32)

def flag(v: U32) -> Bool:
  U32.is_eq(v, 1)

# ------------------------------------------------------------- chunks

def enc_next.fin(+n: U32, r: (Array<F32> & Array<U32>) & Array<F32>) -> ((Array<F32> & Array<U32>) & Array<F32>) & U32:
  ((w, b), h) = r
  (((w, b), h), n)

def enc_next.go(+i: U32, +clen: U32, +ids_off: U32, w: Array<F32>, r: Array<U32> & U32) -> ((Array<F32> & Array<U32>) & Array<F32>) & U32:
  (b, +n) = r
  enc_next.fin(n, E.encode(w, b, (ids_off + (i * clen) : U32), n))

# encode chunk i (length at b[5+i], ids at ids_off + i*clen)

def enc_next(w: Array<F32>, b: Array<U32>, +i: U32, +clen: U32, +ids_off: U32) -> ((Array<F32> & Array<U32>) & Array<F32>) & U32:
  enc_next.go(i, clen, ids_off, w, Array.get(U32, b, (off.lens() + i : U32)))

def chunks.seq(k: Nat, +i: U32, +clen: U32, +ids_off: U32, acc: Hs, r: ((Array<F32> & Array<U32>) & Array<F32>) & U32)
  -> (Array<F32> & Array<U32>) & Hs:
  match k:
    case 0n:
      (((w, b), h), n) = r
      ((w, b), HNode{acc, HLeaf{h, n}})
    case 1n+p:
      (((w, b), h), n) = r
      +j = (i + 1 : U32)
      chunks.seq(p, j, clen, ids_off, HNode{acc, HLeaf{h, n}}, enc_next(w, b, j, clen, ids_off))

# chunks [lo, lo+cnt), cnt >= 1, sequentially

def chunks.range(w: Array<F32>, b: Array<U32>, +lo: U32, +cnt: U32, +clen: U32, +ids_off: U32) -> (Array<F32> & Array<U32>) & Hs:
  chunks.seq(U32.to_nat(U32.sub(cnt, 1)), lo, clen, ids_off, HNil{}, enc_next(w, b, lo, clen, ids_off))

def chunks.join(ra: (Array<F32> & Array<U32>) & Hs, rb: (Array<F32> & Array<U32>) & Hs) -> (Array<F32> & Array<U32>) & Hs:
  ((wa, ba), ha) = ra
  ((wb, bb), hb) = rb
  ((Array.join(F32, wa, wb), Array.join(U32, ba, bb)), HNode{ha, hb})

# parallel fork over chunk ranges (every leaf keeps >= 1 chunk when d = floor(log2 cnt))

def chunks.go(d: Nat, fw: Array<F32> & Array<F32>, fb: Array<U32> & Array<U32>, +lo: U32, +cnt: U32, +clen: U32, +ids_off: U32)
  -> (Array<F32> & Array<U32>) & Hs:
  match d:
    case 0n:
      (wa, wb) = fw
      (ba, bb) = fb
      chunks.range(Array.join(F32, wa, wb), Array.join(U32, ba, bb), lo, cnt, clen, ids_off)
    case 1n++p:
      (wa, wb) = fw
      (ba, bb) = fb
      +h = U32.shr(cnt)
      ra rb = chunks.go(p, Array.fork(F32, wa), Array.fork(U32, ba), lo, h, clen, ids_off) chunks.go(p, Array.fork(F32, wb), Array.fork(U32, bb), (lo + h : U32), U32.sub(cnt, h), clen, ids_off)
      chunks.join(ra, rb)

def chunks(w: Array<F32>, b: Array<U32>, +nch: U32, +clen: U32, +ids_off: U32) -> (Array<F32> & Array<U32>) & Hs:
  chunks.go(U32.log2(nch), Array.fork(F32, w), Array.fork(U32, b), 0, nch, clen, ids_off)

# ------------------------------------------------------- token memory
# mem row <- h row i + seg_emb[c] ; lv row c += h row i

def leaf.row.c(h: Array<F32>, r: Array<F32> & Array<F32>) -> (Array<F32> & Array<F32>) & Array<F32>:
  (w, mem) = r
  ((h, w), mem)

def leaf.row.b(+d: U32, +segoff: U32, +memrow: U32, w: Array<F32>, r: Array<F32> & Array<F32>) -> (Array<F32> & Array<F32>) & Array<F32>:
  (h, mem) = r
  leaf.row.c(h, K.axpy(w, mem, segoff, memrow, d))

def leaf.row.e(w: Array<F32>, mem: Array<F32>, r: Array<F32> & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (h, lv) = r
  (((h, w), mem), lv)

# one token row i of a chunk leaf

def leaf.row.d(+d: U32, +hi: U32, +lvrow: U32, lv: Array<F32>, r: (Array<F32> & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  ((h, w), mem) = r
  leaf.row.e(w, mem, K.axpy(h, lv, hi, lvrow, d))

def leaf.row(+i: U32, +d: U32, +segoff: U32, +memrow: U32, +lvrow: U32, h: Array<F32>, mem: Array<F32>, lv: Array<F32>, w: Array<F32>)
  -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  +hi = (i * d : U32)
  leaf.row.d(d, hi, lvrow, lv, leaf.row.b(d, segoff, memrow, w, K.copy(h, mem, hi, memrow, d)))

def leaf.rows.go(k: Nat, +i: U32, +d: U32, +segoff: U32, +memrow: U32, +lvrow: U32, r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>)
  -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  match k:
    case 0n:
      r
    case 1n+p:
      (((h, w), mem), lv) = r
      leaf.rows.go(p, (i + 1 : U32), d, segoff, (memrow + d : U32), lvrow, leaf.row(i, d, segoff, memrow, lvrow, h, mem, lv, w))

# finish a leaf: lv row <- lv row / n + seg_emb[c]

def leaf.fin2(mem: Array<F32>, +row: U32, +c: U32, r: Array<F32> & Array<F32>) -> MSt:
  (w, lv) = r
  MSt{mem, lv, w, row, c}

def leaf.fin(+n: U32, +d: U32, +segoff: U32, +lvrow: U32, +row: U32, +c: U32, r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> MSt:
  (((h, w), mem), lv) = r
  leaf.fin2(mem, row, c, K.axpy(w, K.scale(lv, lvrow, d, (1.0 / U32.to_f32(n) : F32)), segoff, lvrow, d))

def leaf.fill(h: Array<F32>, +n: U32, st: MSt) -> MSt:
  MSt{mem, lv, w, +row, +c} = st
  +d = L.d()
  +segoff = (L.seg_emb() + (c * d) : U32)
  +lvrow = (c * d : U32)
  leaf.fin(n, d, segoff, lvrow, (row + n : U32), (c + 1 : U32),
    leaf.rows.go(U32.to_nat(n), 0, d, segoff, (row * d : U32), lvrow, (((h, w), mem), K.fill(lv, lvrow, d, 0.0))))

# walk the chunk tree in order, filling mem rows and leaf vectors

def mem.fill(hs: Hs, st: MSt) -> MSt:
  match hs:
    case HNil{}:
      st
    case HLeaf{h, n}:
      leaf.fill(h, n, st)
    case HNode{l, r}:
      mem.fill(r, mem.fill(l, st))

# ---------------------------------------------------------- tree reduce
# node = LN(l + r + W[l;r;l*r;|l-r|] + b) + level_emb[lv]

def pair.cat.step(+k: U32, +d: U32, +lv: F32, t4: Array<F32>, r: Array<F32> & F32) -> Array<F32> & Array<F32>:
  (cur, rv) = r
  +rv2 = rv
  +lv2 = lv
  (cur, Array.set(F32, Array.set(F32, Array.set(F32, Array.set(F32, t4, k, lv2), (k + d : U32), rv2), (k + (d * 2) : U32), (lv2 * rv2 : F32)), (k + (d * 3) : U32), F32.abs((lv2 - rv2 : F32))))

def pair.cat.l(+k: U32, +d: U32, +ro: U32, t4: Array<F32>, r: Array<F32> & F32) -> Array<F32> & Array<F32>:
  (cur, lv) = r
  pair.cat.step(k, d, lv, t4, Array.get(F32, cur, (ro + k : U32)))

def pair.cat.go(n: Nat, +k: U32, +d: U32, +lo: U32, +ro: U32, r: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      r
    case 1n+p:
      (cur, t4) = r
      pair.cat.go(p, (k + 1 : U32), d, lo, ro, pair.cat.l(k, d, ro, t4, Array.get(F32, cur, (lo + k : U32))))

# t4 <- [l ; r ; l*r ; |l-r|] from cur rows lo, ro

def pair.cat(cur: Array<F32>, t4: Array<F32>, +d: U32, +lo: U32, +ro: U32) -> Array<F32> & Array<F32>:
  pair.cat.go(U32.to_nat(d), 0, d, lo, ro, (cur, t4))

# nd <- linear(t4) + b ; nd += l ; nd += r ; nd <- LN(nd) ; nd += level_emb

def merge.c(cur: Array<F32>, t4: Array<F32>, r: Array<F32> & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (w, nd) = r
  (((cur, t4), w), nd)

def merge.b(cur: Array<F32>, r: (Array<F32> & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  ((t4, w), nd) = r
  merge.c(cur, t4, K.add_bias(w, nd, 1, L.d(), L.tree_b(), 0))

def merge.a(+d: U32, w: Array<F32>, nd: Array<F32>, r: Array<F32> & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (cur, t4) = r
  merge.b(cur, K.linear(t4, w, nd, 1, (d * 4 : U32), d, 0, L.tree_w(), 0))

def merge.f(t4: Array<F32>, w: Array<F32>, r: Array<F32> & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (cur, nd) = r
  (((cur, t4), w), nd)

# LN in place via t4 as scratch (t4[0..d) <- LN(nd)) then copy back and add the level embedding

def merge.e(t4: Array<F32>, w: Array<F32>, +ro: U32, +d: U32, r: Array<F32> & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (cur, nd) = r
  merge.f(t4, w, K.axpy(cur, nd, ro, 0, d))

def merge.d(+d: U32, +lo: U32, +ro: U32, r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (((cur, t4), w), nd) = r
  merge.e(t4, w, ro, d, K.axpy(cur, nd, lo, 0, d))

def merge.j(cur: Array<F32>, t4: Array<F32>, r: Array<F32> & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (w, nd) = r
  (((cur, t4), w), nd)

# full merge of cur rows lo, ro into nd

def merge.i(cur: Array<F32>, w: Array<F32>, +d: U32, +lvl: U32, r: Array<F32> & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (t4, nd) = r
  +lv = U32.min(lvl, U32.sub(L.tree_levels(), 1))
  merge.j(cur, t4, K.axpy(w, nd, (L.tree_level() + (lv * d) : U32), 0, d))

def merge.h(cur: Array<F32>, +d: U32, +lvl: U32, r: (Array<F32> & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  ((nd, w), t4) = r
  merge.i(cur, w, d, lvl, K.copy(t4, nd, 0, 0, d))

def merge.g(+d: U32, +lvl: U32, r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (((cur, t4), w), nd) = r
  merge.h(cur, d, lvl, K.layernorm(nd, w, t4, 1, d, 0, L.tree_norm(), 0, False{}, 0, L.eps()))

def merge(cur: Array<F32>, t4: Array<F32>, w: Array<F32>, nd: Array<F32>, +d: U32, +lo: U32, +ro: U32, +lvl: U32)
  -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  merge.g(d, lvl, merge.d(d, lo, ro, merge.a(d, w, nd, pair.cat(cur, t4, d, lo, ro))))

# write nd into nxt row i and mem row `row`

def node.store3(cur: Array<F32>, t4: Array<F32>, w: Array<F32>, vn: Array<U32>, +i: U32, +row: U32, nxt: Array<F32>, r: Array<F32> & Array<F32>) -> TSt & U32:
  (nd, mem) = r
  (TSt{cur, nxt, Array.new(U32, 0n, 0), Array.set(U32, vn, i, 1), mem, w, t4, nd, (row + 1 : U32)}, 1)

# pass a single valid child through: nxt row i <- cur row src

def node.store2(cur: Array<F32>, t4: Array<F32>, w: Array<F32>, vn: Array<U32>, +i: U32, +row: U32, mem: Array<F32>, r: Array<F32> & Array<F32>, +d: U32) -> TSt & U32:
  (nd, nxt) = r
  node.store3(cur, t4, w, vn, i, row, nxt, K.copy(nd, mem, 0, (row * d : U32), d))

def node.store(+d: U32, +i: U32, +row: U32, nxt: Array<F32>, mem: Array<F32>, vn: Array<U32>, r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> TSt & U32:
  (((cur, t4), w), nd) = r
  node.store2(cur, t4, w, vn, i, row, mem, K.copy(nd, nxt, 0, (i * d : U32), d), d)

def node.pass2(vc: Array<U32>, vn: Array<U32>, mem: Array<F32>, w: Array<F32>, t4: Array<F32>, nd: Array<F32>, +row: U32, r: Array<F32> & Array<F32>) -> TSt & U32:
  (cur, nxt) = r
  (TSt{cur, nxt, vc, vn, mem, w, t4, nd, row}, 0)

# vc is returned separately by node.store (it drops it): re-attach

def node.pass(+d: U32, +i: U32, +src: U32, st: TSt) -> TSt & U32:
  TSt{cur, nxt, vc, vn, mem, w, t4, nd, row} = st
  node.pass2(vc, Array.set(U32, vn, i, 1), mem, w, t4, nd, row, K.copy(cur, nxt, (src * d : U32), (i * d : U32), d))

def node.both2(vc: Array<U32>, r: TSt & U32) -> TSt & U32:
  (st, f) = r
  TSt{cur, nxt, vc0, vn, mem, w, t4, nd, row} = st
  (TSt{cur, nxt, vc, vn, mem, w, t4, nd, row}, f)

def node.both(+d: U32, +i: U32, +lvl: U32, st: TSt) -> TSt & U32:
  TSt{cur, nxt, vc, vn, mem, w, t4, nd, row} = st
  node.both2(vc, node.store(d, i, row, nxt, mem, vn, merge(cur, t4, w, nd, d, (i * d * 2 : U32), ((i * 2 + 1) * d : U32), lvl)))

def node.none(+i: U32, st: TSt) -> TSt & U32:
  TSt{cur, nxt, vc, vn, mem, w, t4, nd, row} = st
  (TSt{cur, nxt, vc, Array.set(U32, vn, i, 0), mem, w, t4, nd, row}, 0)

def node.sel(vl: Bool, vr: Bool, +d: U32, +i: U32, +lvl: U32, st: TSt) -> TSt & U32:
  match vl vr:
    case True{} True{}:
      node.both(d, i, lvl, st)
    case True{} False{}:
      node.pass(d, i, (i * 2 : U32), st)
    case False{} True{}:
      node.pass(d, i, (i * 2 + 1 : U32), st)
    case False{} False{}:
      node.none(i, st)

def pair.flags2(+vl: U32, +d: U32, +i: U32, +lvl: U32, cur: Array<F32>, nxt: Array<F32>, vn: Array<U32>, mem: Array<F32>, w: Array<F32>, t4: Array<F32>, nd: Array<F32>, +row: U32, r: Array<U32> & U32) -> TSt & U32:
  (vc, vr) = r
  node.sel(flag(vl), flag(vr), d, i, lvl, TSt{cur, nxt, vc, vn, mem, w, t4, nd, row})

def pair.flags(+d: U32, +i: U32, +lvl: U32, cur: Array<F32>, nxt: Array<F32>, vn: Array<U32>, mem: Array<F32>, w: Array<F32>, t4: Array<F32>, nd: Array<F32>, +row: U32, r: Array<U32> & U32) -> TSt & U32:
  (vc, vl) = r
  pair.flags2(vl, d, i, lvl, cur, nxt, vn, mem, w, t4, nd, row, Array.get(U32, vc, (i * 2 + 1 : U32)))

# process pair i of the current level

def pair(+d: U32, +i: U32, +lvl: U32, st: TSt) -> TSt & U32:
  TSt{cur, nxt, vc, vn, mem, w, t4, nd, row} = st
  pair.flags(d, i, lvl, cur, nxt, vn, mem, w, t4, nd, row, Array.get(U32, vc, (i * 2 : U32)))

def pairs.go(k: Nat, +i: U32, +d: U32, +lvl: U32, r: TSt & U32) -> TSt:
  match k:
    case 0n:
      (st, f) = r
      st
    case 1n+p:
      (st, f) = r
      pairs.go(p, (i + 1 : U32), d, lvl, pair(d, i, lvl, st))

# one level: P pairs -> P nodes in nxt; then swap buffers

def level.swap(st: TSt) -> TSt:
  TSt{cur, nxt, vc, vn, mem, w, t4, nd, row} = st
  TSt{nxt, cur, vn, vc, mem, w, t4, nd, row}

def level(+P: U32, +d: U32, +lvl: U32, st: TSt) -> TSt:
  level.swap(pairs.go(U32.to_nat(U32.sub(P, 1)), 1, d, lvl, pair(d, 0, lvl, st)))

def levels.go(k: Nat, +P: U32, +d: U32, +lvl: U32, st: TSt) -> TSt:
  match k:
    case 0n:
      st
    case 1n+p:
      levels.go(p, U32.shr(P), d, (lvl + 1 : U32), level(P, d, lvl, st))

# valid flags for the leaves: 1 for c < m, else 0 (P slots)

def vflags.go(k: Nat, +i: U32, +m: U32, v: Array<U32>) -> Array<U32>:
  match k:
    case 0n:
      v
    case 1n+p:
      vflags.go(p, (i + 1 : U32), m, Array.set(U32, v, i, K.picku(U32.is_lt(i, m), 1, 0)))

# tree over m leaves (lv rows 0..m) -> nodes appended to mem from row `row0`; returns (mem, w, root, rows_used)

def tree.fin2(mem: Array<F32>, w: Array<F32>, +row: U32, r: Array<F32> & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & U32:
  (cur, root) = r
  (((mem, w), root), row)

# --------------------------------------------------------------- K/V

def tree.fin3(mem: Array<F32>, w: Array<F32>, +row: U32, r: Array<F32> & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & U32:
  (cur, root) = r
  (((mem, w), root), row)

def tree.fin(+d: U32, +row0: U32, st: TSt) -> ((Array<F32> & Array<F32>) & Array<F32>) & U32:
  TSt{cur, nxt, vc, vn, mem, w, t4, nd, row} = st
  tree.fin3(mem, w, row, K.copy(cur, nd, 0, 0, d))

def tree(lv: Array<F32>, mem: Array<F32>, w: Array<F32>, +m: U32, +row0: U32) -> ((Array<F32> & Array<F32>) & Array<F32>) & U32:
  +d = L.d()
  +P = U32.pow(2, K.depth_for(U32.max(m, 2)))
  +pd = K.depth_for(U32.max(m, 2))
  tree.fin(d, row0, levels.go(pd, U32.shr(P), d, 0,
    TSt{lv, K.zeros(K.depth_for((P * d : U32))), vflags.go(U32.to_nat(P), 0, m, Array.new(U32, pd, 0)), Array.new(U32, pd, 0), mem, w,
        K.zeros(K.depth_for((d * 4 : U32))), K.zeros(K.depth_for(d)), row0}))

def kv.go(k: Nat, +li: U32, +Mlen: U32, r: (Array<F32> & Array<F32>) & Array<F32>) -> (Array<F32> & Array<F32>) & Array<F32>:
  match k:
    case 0n:
      r
    case 1n+p:
      ((mem, w), kv) = r
      +d = L.d()
      kv.go(p, (li + 1 : U32), Mlen, K.plinear(mem, w, kv, Mlen, d, (d * 2 : U32), 0, L.f_wkv(li), (li * Mlen * d * 2 : U32)))

# kv[l] = mem W_kv(l)^T for every fusion layer

def kvs(mem: Array<F32>, w: Array<F32>, +Mlen: U32) -> (Array<F32> & Array<F32>) & Array<F32>:
  kv.go(U32.to_nat(L.fusion_layers()), 0, Mlen, ((mem, w), K.zeros(K.depth_for((L.fusion_layers() * Mlen * L.d() * 2 : U32)))))

# ------------------------------------------------- question sequences
# Each question is one encoder sequence (head + state head + option blocks, shared
# option positions, block-restricted attention). mk row (o0 + k) <- h[marker_k];
# cl row q <- h[0] (the question token).

def qe.k.fin(b: Array<U32>, r: Array<F32> & Array<F32>) -> (Array<U32> & Array<F32>) & Array<F32>:
  (h, mk) = r
  ((b, h), mk)

def qe.k.pend(h: Array<F32>, mk: Array<F32>, r: Array<U32> & U32) -> ((Array<U32> & Array<F32>) & Array<F32>) & U32:
  (b, m) = r
  (((b, h), mk), m)

# copy the K marker rows of h into mk rows o .. o+K

def qe.k.next(+j: U32, +o: U32, +mko: U32, b: Array<U32>, r: Array<F32> & Array<F32>) -> ((Array<U32> & Array<F32>) & Array<F32>) & U32:
  (h, mk) = r
  +jn = (j + 1 : U32)
  qe.k.pend(h, mk, Array.get(U32, b, (mko + jn : U32)))

def qe.store.k(k: Nat, +j: U32, +o: U32, +mko: U32, r: ((Array<U32> & Array<F32>) & Array<F32>) & U32) -> (Array<U32> & Array<F32>) & Array<F32>:
  match k:
    case 0n:
      (((b, h), mk), +mpos) = r
      qe.k.fin(b, K.copy(h, mk, (mpos * L.d() : U32), ((o + j) * L.d() : U32), L.d()))
    case 1n+p:
      (((b, h), mk), +mpos) = r
      qe.store.k(p, (j + 1 : U32), o, mko, qe.k.next(j, o, mko, b, K.copy(h, mk, (mpos * L.d() : U32), ((o + j) * L.d() : U32), L.d())))

def qe.g(w: Array<F32>, b: Array<U32>, mk: Array<F32>, r: Array<F32> & Array<F32>) -> (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>):
  (h, cl) = r
  ((w, b), (mk, cl))

def qe.f(+q: U32, w: Array<F32>, cl: Array<F32>, r: (Array<U32> & Array<F32>) & Array<F32>) -> (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>):
  ((b, h), mk) = r
  qe.g(w, b, mk, K.copy(h, cl, 0, (q * L.d() : U32), L.d()))

def qe.e2(+q: U32, +Kq: U32, +o0: U32, +mko: U32, w: Array<F32>, mk: Array<F32>, cl: Array<F32>, h: Array<F32>, b: Array<U32>)
  -> (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>):
  qe.f(q, w, cl, qe.store.k(U32.to_nat(U32.sub(Kq, 1)), 0, o0, mko, qe.k.pend(h, mk, Array.get(U32, b, mko))))

def qe.e(+q: U32, +Kq: U32, +o0: U32, +mko: U32, mk: Array<F32>, cl: Array<F32>, r: ((Array<F32> & Array<U32>) & Array<F32>) & Array<U32>)
  -> (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>):
  (((w, b1), h), b2) = r
  qe.e2(q, Kq, o0, mko, w, mk, cl, h, Array.join(U32, b1, b2))

def qe.d(+q: U32, +L: U32, +Kq: U32, +o0: U32, +of: QOff, w: Array<F32>, mk: Array<F32>, cl: Array<F32>, fb: Array<U32> & Array<U32>)
  -> (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>):
  QOff{qo, qlen, kmax, ids, pos, blk, mko} = of
  (b1, b2) = fb
  +row = (q * qlen : U32)
  qe.e(q, Kq, o0, (mko + (q * kmax) : U32), mk, cl, E.encode_x(w, b1, b2, (ids + row : U32), L, (pos + row : U32), (blk + row : U32)))

def qe.c(+q: U32, +L: U32, +Kq: U32, +of: QOff, w: Array<F32>, mk: Array<F32>, cl: Array<F32>, r: Array<U32> & U32)
  -> (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>):
  (b, +o0) = r
  qe.d(q, L, Kq, o0, of, w, mk, cl, Array.fork(U32, b))

def qe.b(+q: U32, +L: U32, +of: QOff, w: Array<F32>, mk: Array<F32>, cl: Array<F32>, r: Array<U32> & U32)
  -> (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>):
  QOff{qo, qlen, kmax, ids, pos, blk, mko} = of
  (b, +Kq) = r
  qe.c(q, L, Kq, of, w, mk, cl, Array.get(U32, b, (qo + (q * 4) + 3 : U32)))

def qe.a(+q: U32, +of: QOff, w: Array<F32>, mk: Array<F32>, cl: Array<F32>, r: Array<U32> & U32)
  -> (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>):
  QOff{qo, qlen, kmax, ids, pos, blk, mko} = of
  (b, +L) = r
  qe.b(q, L, of, w, mk, cl, Array.get(U32, b, (qo + (q * 4) + 1 : U32)))

# encode question q

def qe.one(+q: U32, +of: QOff, w: Array<F32>, b: Array<U32>, mk: Array<F32>, cl: Array<F32>) -> (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>):
  QOff{qo, qlen, kmax, ids, pos, blk, mko} = of
  qe.a(q, of, w, mk, cl, Array.get(U32, b, (qo + (q * 4) + 2 : U32)))

def qes.seq(k: Nat, +q: U32, +of: QOff, r: (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>)) -> (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>):
  match k:
    case 0n:
      r
    case 1n+p:
      ((w, b), (mk, cl)) = r
      qes.seq(p, (q + 1 : U32), of, qe.one(q, of, w, b, mk, cl))

def opts.join(ra: (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>), rb: (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>))
  -> (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>):
  ((wa, ba), (ma, ca)) = ra
  ((wb, bb), (mb, cb)) = rb
  ((Array.join(F32, wa, wb), Array.join(U32, ba, bb)), (Array.join(F32, ma, mb), Array.join(F32, ca, cb)))

# parallel fork over question ranges [lo, lo+cnt)

def qes.go(d: Nat, fw: Array<F32> & Array<F32>, fb: Array<U32> & Array<U32>, fm: Array<F32> & Array<F32>, fc: Array<F32> & Array<F32>,
           +lo: U32, +cnt: U32, +of: QOff) -> (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>):
  match d:
    case 0n:
      (wa, wb) = fw
      (ba, bb) = fb
      (ma, mb) = fm
      (ca, cb) = fc
      qes.seq(U32.to_nat(cnt), lo, of, ((Array.join(F32, wa, wb), Array.join(U32, ba, bb)), (Array.join(F32, ma, mb), Array.join(F32, ca, cb))))
    case 1n++p:
      (wa, wb) = fw
      (ba, bb) = fb
      (ma, mb) = fm
      (ca, cb) = fc
      +h = U32.shr(cnt)
      ra rb = qes.go(p, Array.fork(F32, wa), Array.fork(U32, ba), Array.fork(F32, ma), Array.fork(F32, ca), lo, h, of) qes.go(p, Array.fork(F32, wb), Array.fork(U32, bb), Array.fork(F32, mb), Array.fork(F32, cb), (lo + h : U32), U32.sub(cnt, h), of)
      opts.join(ra, rb)

# all questions -> ((w, b), (mk: nopt x d, cl: nq x d))

def qencode(w: Array<F32>, b: Array<U32>, +nq: U32, +nopt: U32, +of: QOff) -> (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>):
  +d = L.d()
  qes.go(U32.log2(nq), Array.fork(F32, w), Array.fork(U32, b), Array.fork(F32, K.zeros(K.depth_for((nopt * d : U32)))), Array.fork(F32, K.zeros(K.depth_for((nq * d : U32)))),
         0, nq, of)

# ----------------------------------------------------------- questions
# qoff[q] = first option index of question q (prefix sums of the option counts)

def qoff.go(k: Nat, +q: U32, +qo: U32, +acc: U32, qoff: Array<U32>, r: Array<U32> & U32) -> Array<U32> & Array<U32>:
  match k:
    case 0n:
      (b, kq) = r
      (b, Array.set(U32, qoff, q, acc))
    case 1n+p:
      (b, kq) = r
      +qn = (q + 1 : U32)
      qoff.go(p, qn, qo, (acc + kq : U32), Array.set(U32, qoff, q, acc), Array.get(U32, b, (qo + (qn * 4) + 1 : U32)))

def qoffs(b: Array<U32>, +nq: U32, +qo: U32) -> Array<U32> & Array<U32>:
  qoff.go(U32.to_nat(U32.sub(nq, 1)), 0, qo, 0, Array.new(U32, K.depth_for(nq), 0), Array.get(U32, b, (qo + 1 : U32)))

# S row 0 += cl row o / K ; S row 1+k <- mk row o (+ level_emb[k] when ordinal)

def set.level(ordinal: Bool, +k: U32, +d: U32, w: Array<F32>, S: Array<F32>) -> Array<F32> & Array<F32>:
  match ordinal:
    case True{}:
      K.axpy(w, S, (L.level_emb() + (k * d) : U32), ((k + 1) * d : U32), d)
    case False{}:
      (w, S)

def set.row.d(mk: Array<F32>, r: Array<F32> & Array<F32>) -> (Array<F32> & Array<F32>) & Array<F32>:
  (w, S) = r
  ((mk, w), S)

def set.row.c(+k: U32, +d: U32, +ordinal: Bool, w: Array<F32>, r: Array<F32> & Array<F32>) -> (Array<F32> & Array<F32>) & Array<F32>:
  (mk, S) = r
  set.row.d(mk, set.level(ordinal, k, d, w, S))

def set.row.e(cl: Array<F32>, r: (Array<F32> & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  ((mk, w), S) = r
  (((mk, w), S), cl)

def set.row.b(+k: U32, +d: U32, +ordinal: Bool, w: Array<F32>, mk: Array<F32>, +o: U32, r: Array<F32> & Array<F32>, +invK: F32) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (cl, S) = r
  set.row.e(cl, set.row.c(k, d, ordinal, w, K.copy(mk, S, (o * d : U32), ((k + 1) * d : U32), d)))

def set.row(+k: U32, +d: U32, +ordinal: Bool, +o: U32, +invK: F32, mk: Array<F32>, cl: Array<F32>, w: Array<F32>, S: Array<F32>)
  -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  set.row.b(k, d, ordinal, w, mk, o, (cl, S), invK)

def set.rows.go(n: Nat, +k: U32, +d: U32, +ordinal: Bool, +o: U32, +invK: F32, r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>)
  -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  match n:
    case 0n:
      r
    case 1n+p:
      (((mk, w), S), cl) = r
      set.rows.go(p, (k + 1 : U32), d, ordinal, (o + 1 : U32), invK, set.row(k, d, ordinal, o, invK, mk, cl, w, S))

# S (Ks x d): question token in row 0, options in rows 1..K

def set.qtype2(mk: Array<F32>, cl: Array<F32>, r: Array<F32> & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (w, S) = r
  (((mk, w), S), cl)

# ---- one question: read its offsets, build the set, decide, store the confidence logit

def set.qtype.c(mk: Array<F32>, w: Array<F32>, +qtype: U32, +d: U32, r: Array<F32> & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (cl, S) = r
  set.qtype2(mk, cl, K.axpy(w, S, (L.qtype_emb() + (qtype * d) : U32), 0, d))

# S row 0 <- cl row q + qtype_emb[qtype]

def set.qtype(+d: U32, +qtype: U32, +q: U32, r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (((mk, w), S), cl) = r
  set.qtype.c(mk, w, qtype, d, K.axpy(cl, S, (q * d : U32), 0, d))

def build_set(mk: Array<F32>, cl: Array<F32>, w: Array<F32>, +o0: U32, +Kq: U32, +qtype: U32, +q: U32) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  +d = L.d()
  set.qtype(d, qtype, q, set.rows.go(U32.to_nat(Kq), 0, d, Bool.not(U32.is_zero(qtype)), o0, (1.0 / U32.to_f32(Kq) : F32),
    (((mk, w), K.zeros(K.depth_for(((Kq + 1) * d : U32)))), cl)))

def quest.f(b: Array<U32>, mk: Array<F32>, cl: Array<F32>, qoff: Array<U32>, w: Array<F32>, kv: Array<F32>, root: Array<F32>, r: Array<F32> & Array<F32>) -> GSt & Array<U32>:
  (sc, out) = r
  (GSt{w, b, kv, mk, cl, root, out}, qoff)

def quest.e(+q: U32, +Kq: U32, +nopt: U32, b: Array<U32>, mk: Array<F32>, cl: Array<F32>, qoff: Array<U32>, r: F.QOut) -> GSt & Array<U32>:
  F.QOut{st, out, root} = r
  F.QSt{w, kv, s, h, qkv, att, qq, xo, mlp, act, sc} = st
  quest.f(b, mk, cl, qoff, w, kv, root, K.copy(sc, out, Kq, (nopt + q : U32), 1))

def quest.d(+q: U32, +o0: U32, +Kq: U32, +qtype: U32, +nopt: U32, +nq: U32, +Mlen: U32, b: Array<U32>, kv: Array<F32>, root: Array<F32>, out: Array<F32>,
            qoff: Array<U32>, r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> GSt & Array<U32>:
  (((mk, w), S), cl) = r
  quest.e(q, Kq, nopt, b, mk, cl, qoff, F.decide(Kq, Mlen, qtype, o0, (nopt + nq + o0 : U32), F.qst.new(w, kv, S, (Kq + 1 : U32), Mlen), out, root))

def quest.c(+q: U32, +o0: U32, +Kq: U32, +nopt: U32, +nq: U32, +Mlen: U32, w: Array<F32>, kv: Array<F32>, mk: Array<F32>, cl: Array<F32>,
            root: Array<F32>, out: Array<F32>, qoff: Array<U32>, r: Array<U32> & U32) -> GSt & Array<U32>:
  (b, +qtype) = r
  quest.d(q, o0, Kq, qtype, nopt, nq, Mlen, b, kv, root, out, qoff, build_set(mk, cl, w, o0, Kq, qtype, q))

def quest.b(+q: U32, +o0: U32, +nopt: U32, +nq: U32, +Mlen: U32, +qo: U32, w: Array<F32>, kv: Array<F32>, mk: Array<F32>, cl: Array<F32>,
            root: Array<F32>, out: Array<F32>, qoff: Array<U32>, r: Array<U32> & U32) -> GSt & Array<U32>:
  (b, Kq) = r
  quest.c(q, o0, Kq, nopt, nq, Mlen, w, kv, mk, cl, root, out, qoff, Array.get(U32, b, (qo + (q * 4) : U32)))

def quest.a(+q: U32, +nopt: U32, +nq: U32, +Mlen: U32, +qo: U32, w: Array<F32>, b: Array<U32>, kv: Array<F32>, mk: Array<F32>, cl: Array<F32>,
            root: Array<F32>, out: Array<F32>, r: Array<U32> & U32) -> GSt & Array<U32>:
  (qoff, o0) = r
  quest.b(q, o0, nopt, nq, Mlen, qo, w, kv, mk, cl, root, out, qoff, Array.get(U32, b, (qo + (q * 4) + 1 : U32)))

def quest(+q: U32, +nopt: U32, +nq: U32, +Mlen: U32, +qo: U32, st: GSt, qoff: Array<U32>) -> GSt & Array<U32>:
  GSt{w, b, kv, mk, cl, root, out} = st
  quest.a(q, nopt, nq, Mlen, qo, w, b, kv, mk, cl, root, out, Array.get(U32, qoff, q))

def quests.seq(k: Nat, +q: U32, +nopt: U32, +nq: U32, +Mlen: U32, +qo: U32, r: GSt & Array<U32>) -> GSt & Array<U32>:
  match k:
    case 0n:
      r
    case 1n+p:
      (st, qoff) = r
      quests.seq(p, (q + 1 : U32), nopt, nq, Mlen, qo, quest(q, nopt, nq, Mlen, qo, st, qoff))

def quests.join(ra: GSt & Array<U32>, rb: GSt & Array<U32>) -> GSt & Array<U32>:
  (GSt{wa, ba, kva, mka, cla, roota, outa}, qa) = ra
  (GSt{wb, bb, kvb, mkb, clb, rootb, outb}, qb) = rb
  (GSt{Array.join(F32, wa, wb), Array.join(U32, ba, bb), Array.join(F32, kva, kvb), Array.join(F32, mka, mkb), Array.join(F32, cla, clb),
       Array.join(F32, roota, rootb), Array.join(F32, outa, outb)}, Array.join(U32, qa, qb))

def gst.fork.g(wa: Array<F32>, wb: Array<F32>, ba: Array<U32>, bb: Array<U32>, ka: Array<F32>, kb: Array<F32>, ma: Array<F32>, mb: Array<F32>, ca: Array<F32>, cb: Array<F32>, ra: Array<F32>, rb: Array<F32>, fo: Array<F32> & Array<F32>) -> GSt & GSt:
  (oa, ob) = fo
  (GSt{wa, ba, ka, ma, ca, ra, oa}, GSt{wb, bb, kb, mb, cb, rb, ob})

# parallel fork over question ranges; qoff is forked too

def gst.fork.f(wa: Array<F32>, wb: Array<F32>, ba: Array<U32>, bb: Array<U32>, ka: Array<F32>, kb: Array<F32>, ma: Array<F32>, mb: Array<F32>, ca: Array<F32>, cb: Array<F32>, fr: Array<F32> & Array<F32>, out: Array<F32>) -> GSt & GSt:
  (ra, rb) = fr
  gst.fork.g(wa, wb, ba, bb, ka, kb, ma, mb, ca, cb, ra, rb, Array.fork(F32, out))

def gst.fork.e(wa: Array<F32>, wb: Array<F32>, ba: Array<U32>, bb: Array<U32>, ka: Array<F32>, kb: Array<F32>, ma: Array<F32>, mb: Array<F32>, fc: Array<F32> & Array<F32>, root: Array<F32>, out: Array<F32>) -> GSt & GSt:
  (ca, cb) = fc
  gst.fork.f(wa, wb, ba, bb, ka, kb, ma, mb, ca, cb, Array.fork(F32, root), out)

def gst.fork.d(wa: Array<F32>, wb: Array<F32>, ba: Array<U32>, bb: Array<U32>, ka: Array<F32>, kb: Array<F32>, fm: Array<F32> & Array<F32>, cl: Array<F32>, root: Array<F32>, out: Array<F32>) -> GSt & GSt:
  (ma, mb) = fm
  gst.fork.e(wa, wb, ba, bb, ka, kb, ma, mb, Array.fork(F32, cl), root, out)

def gst.fork.c(wa: Array<F32>, wb: Array<F32>, ba: Array<U32>, bb: Array<U32>, fk: Array<F32> & Array<F32>, mk: Array<F32>, cl: Array<F32>, root: Array<F32>, out: Array<F32>) -> GSt & GSt:
  (ka, kb) = fk
  gst.fork.d(wa, wb, ba, bb, ka, kb, Array.fork(F32, mk), cl, root, out)

def gst.fork.b(wa: Array<F32>, wb: Array<F32>, fb: Array<U32> & Array<U32>, kv: Array<F32>, mk: Array<F32>, cl: Array<F32>, root: Array<F32>, out: Array<F32>) -> GSt & GSt:
  (ba, bb) = fb
  gst.fork.c(wa, wb, ba, bb, Array.fork(F32, kv), mk, cl, root, out)

def gst.fork.a(fw: Array<F32> & Array<F32>, b: Array<U32>, kv: Array<F32>, mk: Array<F32>, cl: Array<F32>, root: Array<F32>, out: Array<F32>) -> GSt & GSt:
  (wa, wb) = fw
  gst.fork.b(wa, wb, Array.fork(U32, b), kv, mk, cl, root, out)

def gst.fork(st: GSt) -> GSt & GSt:
  GSt{w, b, kv, mk, cl, root, out} = st
  gst.fork.a(Array.fork(F32, w), b, kv, mk, cl, root, out)

def quests.merge(sa: GSt, sb: GSt) -> GSt:
  GSt{wa, ba, kva, mka, cla, roota, outa} = sa
  GSt{wb, bb, kvb, mkb, clb, rootb, outb} = sb
  GSt{Array.join(F32, wa, wb), Array.join(U32, ba, bb), Array.join(F32, kva, kvb), Array.join(F32, mka, mkb), Array.join(F32, cla, clb), Array.join(F32, roota, rootb), Array.join(F32, outa, outb)}

def quests.go(d: Nat, fs: GSt & GSt, fq: Array<U32> & Array<U32>, +lo: U32, +cnt: U32, +nopt: U32, +nq: U32, +Mlen: U32, +qo: U32) -> GSt & Array<U32>:
  match d:
    case 0n:
      (sa, sb) = fs
      (qa, qb) = fq
      quests.seq(U32.to_nat(cnt), lo, nopt, nq, Mlen, qo, (quests.merge(sa, sb), Array.join(U32, qa, qb)))
    case 1n++p:
      (sa, sb) = fs
      (qa, qb) = fq
      +h = U32.shr(cnt)
      ra rb = quests.go(p, gst.fork(sa), Array.fork(U32, qa), lo, h, nopt, nq, Mlen, qo) quests.go(p, gst.fork(sb), Array.fork(U32, qb), (lo + h : U32), U32.sub(cnt, h), nopt, nq, Mlen, qo)
      quests.join(ra, rb)

def questions(st: GSt, qoff: Array<U32>, +nq: U32, +nopt: U32, +Mlen: U32, +qo: U32) -> GSt & Array<U32>:
  quests.go(U32.log2(nq), gst.fork(st), Array.fork(U32, qoff), 0, nq, nopt, nq, Mlen, qo)

# ---------------------------------------------------------------- run
# total tokens over the chunks (sum of b[5 .. 5+nch))

def tt.go(k: Nat, +i: U32, +acc: U32, r: Array<U32> & U32) -> Array<U32> & U32:
  match k:
    case 0n:
      (b, n) = r
      (b, (acc + n : U32))
    case 1n+p:
      (b, n) = r
      +j = (i + 1 : U32)
      tt.go(p, j, (acc + n : U32), Array.get(U32, b, (off.lens() + j : U32)))

def tok_total(b: Array<U32>, +nch: U32) -> Array<U32> & U32:
  tt.go(U32.to_nat(U32.sub(nch, 1)), 0, 0, Array.get(U32, b, off.lens()))

def run.g(r: GSt & Array<U32>) -> Array<F32>:
  (st, qoff) = r
  GSt{w, b, kv, mk, cl, root, out} = st
  out

# after question encoding: ((w, b), (mk, cl)); kv, root, Mlen known

def run.f2(+nq: U32, +nopt: U32, +Mlen: U32, +qo: U32, kv: Array<F32>, root: Array<F32>, w: Array<F32>, mk: Array<F32>, cl: Array<F32>, r: Array<U32> & Array<U32>) -> Array<F32>:
  (b, qoff) = r
  run.g(questions(GSt{w, b, kv, mk, cl, root, K.zeros(K.depth_for(((nopt * 2) + nq + 128 : U32)))}, qoff, nq, nopt, Mlen, qo))

def run.f(+nq: U32, +nopt: U32, +Mlen: U32, +qo: U32, kv: Array<F32>, root: Array<F32>, r: (Array<F32> & Array<U32>) & (Array<F32> & Array<F32>)) -> Array<F32>:
  ((w, b), (mk, cl)) = r
  run.f2(nq, nopt, Mlen, qo, kv, root, w, mk, cl, qoffs(b, nq, qo))

# after kvs: ((mem, w), kv); mem is no longer needed

def run.e(+nch: U32, +clen: U32, +nq: U32, +nopt: U32, +qlen: U32, +kmax: U32, +Mlen: U32, b: Array<U32>, root: Array<F32>, r: (Array<F32> & Array<F32>) & Array<F32>) -> Array<F32>:
  ((mem, w), kv) = r
  +of = {QOff{off.q(nch), qlen, kmax, off.qids(nch, nq, clen), off.qpos(nch, nq, clen, qlen), off.qblk(nch, nq, clen, qlen), off.qmk(nch, nq, clen, qlen)} : QOff}
  run.f(nq, nopt, Mlen, off.q(nch), kv, root, qencode(w, b, nq, nopt, of))

# after tree: (((mem, w), root), Mlen)

def run.d(+nch: U32, +clen: U32, +nq: U32, +nopt: U32, +qlen: U32, +kmax: U32, b: Array<U32>, r: ((Array<F32> & Array<F32>) & Array<F32>) & U32) -> Array<F32>:
  (((mem, w), root), +Mlen) = r
  run.e(nch, clen, nq, nopt, qlen, kmax, Mlen, b, root, kvs(mem, w, Mlen))

# after mem.fill: MSt

def run.c(+nch: U32, +clen: U32, +nq: U32, +nopt: U32, +qlen: U32, +kmax: U32, +T: U32, b: Array<U32>, st: MSt) -> Array<F32>:
  MSt{mem, lv, w, row, c} = st
  run.d(nch, clen, nq, nopt, qlen, kmax, b, tree(lv, mem, w, nch, T))

# after tok_total: (b, T) with hs in hand

def run.b(+nch: U32, +clen: U32, +nq: U32, +nopt: U32, +qlen: U32, +kmax: U32, w: Array<F32>, hs: Hs, r: Array<U32> & U32) -> Array<F32>:
  (b, +T) = r
  +d = L.d()
  run.c(nch, clen, nq, nopt, qlen, kmax, T, b, mem.fill(hs, MSt{K.zeros(K.depth_for(((T + nch) * d : U32))), K.zeros(K.depth_for((nch * d : U32))), w, 0, 0}))

# after chunks: ((w, b), hs)

def run.a(+nch: U32, +clen: U32, +nq: U32, +nopt: U32, +qlen: U32, +kmax: U32, r: (Array<F32> & Array<U32>) & Hs) -> Array<F32>:
  ((w, b), hs) = r
  run.b(nch, clen, nq, nopt, qlen, kmax, w, hs, tok_total(b, nch))

def run(w: Array<F32>, b: Array<U32>, +nch: U32, +clen: U32, +nq: U32, +qlen: U32, +kmax: U32, +nopt: U32) -> Array<F32>:
  run.a(nch, clen, nq, nopt, qlen, kmax, chunks(w, b, nch, clen, off.cids(nch, nq)))

# ------------------------------------------------------------- header
# [n_chunks, chunk_len_max, n_questions, q_len_max, k_max, n_options]

def h6(w: Array<F32>, +nch: U32, +clen: U32, +nq: U32, +qlen: U32, +kmax: U32, r: Array<U32> & U32) -> Array<F32>:
  (b, nopt) = r
  run(w, b, nch, clen, nq, qlen, kmax, nopt)

def h5(w: Array<F32>, +nch: U32, +clen: U32, +nq: U32, +qlen: U32, r: Array<U32> & U32) -> Array<F32>:
  (b, kmax) = r
  h6(w, nch, clen, nq, qlen, kmax, Array.get(U32, b, 5))

def h4(w: Array<F32>, +nch: U32, +clen: U32, +nq: U32, r: Array<U32> & U32) -> Array<F32>:
  (b, qlen) = r
  h5(w, nch, clen, nq, qlen, Array.get(U32, b, 4))

def h3(w: Array<F32>, +nch: U32, +clen: U32, r: Array<U32> & U32) -> Array<F32>:
  (b, nq) = r
  h4(w, nch, clen, nq, Array.get(U32, b, 3))

def h2(w: Array<F32>, +nch: U32, r: Array<U32> & U32) -> Array<F32>:
  (b, clen) = r
  h3(w, nch, clen, Array.get(U32, b, 2))

def h1(w: Array<F32>, r: Array<U32> & U32) -> Array<F32>:
  (b, nch) = r
  h2(w, nch, Array.get(U32, b, 1))

# Groot 1 forward: weights + request blob -> [probs(n_opt) ; conf logits(n_q) ; logits(n_opt)]

def forward(w: Array<F32>, b: Array<U32>) -> Array<F32>:
  h1(w, Array.get(U32, b, 0))
