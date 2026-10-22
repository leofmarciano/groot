# Groot 1 — numeric kernels on flat F32 arrays (row-major), pure Bend.
#
# Conventions
# - Every tensor is a flat Array<F32> plus explicit offsets/dims (U32).
# - Arrays are affine: a kernel receives the arrays it reads/writes and hands
#   them back beside its result. Loops are structural recursion on a Nat
#   counter that comes FIRST in the parameter list (termination check), with
#   U32 indices carried alongside. The result of an Array.get is threaded into
#   the next iteration as a parameter (Bend matches only on parameters).
# - Weights follow PyTorch's nn.Linear layout: W[N, K] row-major, y = x W^T.
import Base

# ---------------------------------------------------------------- scalars

def pickf(c: Bool, a: F32, b: F32) -> F32:
  match c:
    case True{}:
      a
    case False{}:
      b

def picku(c: Bool, a: U32, b: U32) -> U32:
  match c:
    case True{}:
      a
    case False{}:
      b

def pickarr(c: Bool, a: Array<F32>, b: Array<F32>) -> Array<F32>:
  match c:
    case True{}:
      a
    case False{}:
      b

# erf via Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7)

def erf(+x: F32) -> F32:
  +s = pickf((x < 0.0 : F32), F32.neg(1.0), 1.0)
  +ax = F32.abs(x)
  +t = (1.0 / (1.0 + (0.3275911 * ax)) : F32)
  +p5 = (1.061405429 * t : F32)
  +p4 = ((p5 - 1.453152027) * t : F32)
  +p3 = ((p4 + 1.421413741) * t : F32)
  +p2 = ((p3 - 0.284496736) * t : F32)
  +p1 = ((p2 + 0.254829592) * t : F32)
  (s * (1.0 - (p1 * F32.exp(F32.neg((ax * ax : F32))))) : F32)

# exact GELU (matches torch.nn.functional.gelu default)

def gelu(+x: F32) -> F32:
  (0.5 * x * (1.0 + erf((x * 0.7071067811865476 : F32))) : F32)

def zeros(+d: Nat) -> Array<F32>:
  Array.new(F32, d, 0.0)

# smallest depth d with 2^d >= n  (n >= 1)

def depth_for(+n: U32) -> Nat:
  Nat.add(U32.log2(U32.sub(U32.max(n, 2), 1)), 1n)

# ------------------------------------------------------------ set / fill

def fill.go(n: Nat, a: Array<F32>, +i: U32, +v: F32) -> Array<F32>:
  match n:
    case 0n:
      a
    case 1n+p:
      fill.go(p, Array.set(F32, a, i, v), (i + 1 : U32), v)

def fill(a: Array<F32>, +i: U32, +n: U32, +v: F32) -> Array<F32>:
  fill.go(U32.to_nat(n), a, i, v)

# y[yo + i] = x[xo + i] for i < n

def copy.go(n: Nat, +xi: U32, +yi: U32, y: Array<F32>, r: Array<F32> & F32) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      (x, v) = r
      (x, Array.set(F32, y, yi, v))
    case 1n+p:
      (x, v) = r
      +xj = (xi + 1 : U32)
      copy.go(p, xj, (yi + 1 : U32), Array.set(F32, y, yi, v), Array.get(F32, x, xj))

def copy(x: Array<F32>, y: Array<F32>, +xo: U32, +yo: U32, +n: U32) -> Array<F32> & Array<F32>:
  copy.go(U32.to_nat(U32.sub(n, 1)), xo, yo, y, Array.get(F32, x, xo))

# y[yo + i] = y[yo + i] + x[xo + i]   (residual add)

def axpy.step(+yi: U32, +xv: F32, r: Array<F32> & F32) -> Array<F32>:
  (y, yv) = r
  Array.set(F32, y, yi, (yv + xv : F32))

def axpy.go(n: Nat, +xi: U32, +yi: U32, y: Array<F32>, r: Array<F32> & F32) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      (x, xv) = r
      (x, axpy.step(yi, xv, Array.get(F32, y, yi)))
    case 1n+p:
      (x, xv) = r
      +xj = (xi + 1 : U32)
      axpy.go(p, xj, (yi + 1 : U32), axpy.step(yi, xv, Array.get(F32, y, yi)), Array.get(F32, x, xj))

def axpy(x: Array<F32>, y: Array<F32>, +xo: U32, +yo: U32, +n: U32) -> Array<F32> & Array<F32>:
  axpy.go(U32.to_nat(U32.sub(n, 1)), xo, yo, y, Array.get(F32, x, xo))

# y[yo + i] = y[yo + i] * s

def scale.step(+yi: U32, +s: F32, r: Array<F32> & F32) -> Array<F32>:
  (y, v) = r
  Array.set(F32, y, yi, (v * s : F32))

def scale.go(n: Nat, +yi: U32, +s: F32, y: Array<F32>) -> Array<F32>:
  match n:
    case 0n:
      y
    case 1n+p:
      scale.go(p, (yi + 1 : U32), s, scale.step(yi, s, Array.get(F32, y, yi)))

def scale(y: Array<F32>, +yo: U32, +n: U32, +s: F32) -> Array<F32>:
  scale.go(U32.to_nat(n), yo, s, y)

# ----------------------------------------------------------------- linear
# dot over k elements: sum x[xi..] * w[wi..]

def dot.go(k: Nat, +xi: U32, +wi: U32, acc: F32, rx: Array<F32> & F32, rw: Array<F32> & F32) -> (Array<F32> & Array<F32>) & F32:
  match k:
    case 0n:
      (x, xv) = rx
      (w, wv) = rw
      ((x, w), (acc + (xv * wv) : F32))
    case 1n+p:
      (x, xv) = rx
      (w, wv) = rw
      +xj = (xi + 1 : U32)
      +wj = (wi + 1 : U32)
      dot.go(p, xj, wj, (acc + (xv * wv) : F32), Array.get(F32, x, xj), Array.get(F32, w, wj))

def dot(x: Array<F32>, w: Array<F32>, +xi: U32, +wi: U32, +k: U32) -> (Array<F32> & Array<F32>) & F32:
  dot.go(U32.to_nat(U32.sub(k, 1)), xi, wi, 0.0, Array.get(F32, x, xi), Array.get(F32, w, wi))

# one output row m: y[yo + n] = dot(x row, w row n) + bias  for n < N

def lin.row.go(n: Nat, +xrow: U32, +wo: U32, +bo: U32, +k: U32, +yi: U32, +ni: U32, y: Array<F32>, r: (Array<F32> & Array<F32>) & F32)
  -> (Array<F32> & Array<F32>) & Array<F32>:
  match n:
    case 0n:
      ((x, w), v) = r
      ((x, w), Array.set(F32, y, yi, v))
    case 1n+p:
      ((x, w), v) = r
      +nj = (ni + 1 : U32)
      lin.row.go(p, xrow, wo, bo, k, (yi + 1 : U32), nj, Array.set(F32, y, yi, v), dot(x, w, xrow, (wo + (nj * k) : U32), k))

def lin.row(x: Array<F32>, w: Array<F32>, y: Array<F32>, +xrow: U32, +wo: U32, +bo: U32, +k: U32, +n: U32, +yrow: U32)
  -> (Array<F32> & Array<F32>) & Array<F32>:
  lin.row.go(U32.to_nat(U32.sub(n, 1)), xrow, wo, bo, k, yrow, 0, y, dot(x, w, xrow, wo, k))

def lin.rows.go(m: Nat, +xo: U32, +wo: U32, +k: U32, +n: U32, +yo: U32, r: (Array<F32> & Array<F32>) & Array<F32>)
  -> (Array<F32> & Array<F32>) & Array<F32>:
  match m:
    case 0n:
      r
    case 1n+p:
      ((x, w), y) = r
      lin.rows.go(p, (xo + k : U32), wo, k, n, (yo + n : U32), lin.row(x, w, y, xo, wo, 0, k, n, yo))

# y[yo + m*N + n] = sum_k x[xo + m*K + k] * w[wo + n*K + k]      (M rows)

def linear(x: Array<F32>, w: Array<F32>, y: Array<F32>, +M: U32, +K: U32, +N: U32, +xo: U32, +wo: U32, +yo: U32)
  -> (Array<F32> & Array<F32>) & Array<F32>:
  lin.rows.go(U32.to_nat(M), xo, wo, K, N, yo, ((x, w), y))

# bias add over M rows: y[yo + m*N + n] += w[bo + n]

def bias.row.go(n: Nat, +bi: U32, +yi: U32, y: Array<F32>, r: Array<F32> & F32) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      (w, bv) = r
      (w, axpy.step(yi, bv, Array.get(F32, y, yi)))
    case 1n+p:
      (w, bv) = r
      +bj = (bi + 1 : U32)
      bias.row.go(p, bj, (yi + 1 : U32), axpy.step(yi, bv, Array.get(F32, y, yi)), Array.get(F32, w, bj))

def bias.rows.go(m: Nat, +bo: U32, +n: U32, +yo: U32, r: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  match m:
    case 0n:
      r
    case 1n+p:
      (w, y) = r
      bias.rows.go(p, bo, n, (yo + n : U32), bias.row.go(U32.to_nat(U32.sub(n, 1)), bo, yo, y, Array.get(F32, w, bo)))

def add_bias(w: Array<F32>, y: Array<F32>, +M: U32, +N: U32, +bo: U32, +yo: U32) -> Array<F32> & Array<F32>:
  bias.rows.go(U32.to_nat(M), bo, N, yo, (w, y))

# parallel linear over row halves: forked handles read x/w and write disjoint rows of y

def plin.join(ra: (Array<F32> & Array<F32>) & Array<F32>, rb: (Array<F32> & Array<F32>) & Array<F32>)
  -> (Array<F32> & Array<F32>) & Array<F32>:
  ((xa, wa), ya) = ra
  ((xb, wb), yb) = rb
  ((Array.join(F32, xa, xb), Array.join(F32, wa, wb)), Array.join(F32, ya, yb))

def plin.go(d: Nat, fx: Array<F32> & Array<F32>, fw: Array<F32> & Array<F32>, fy: Array<F32> & Array<F32>,
            +M: U32, +K: U32, +N: U32, +xo: U32, +wo: U32, +yo: U32)
  -> (Array<F32> & Array<F32>) & Array<F32>:
  match d:
    case 0n:
      (xa, xb) = fx
      (wa, wb) = fw
      (ya, yb) = fy
      linear(Array.join(F32, xa, xb), Array.join(F32, wa, wb), Array.join(F32, ya, yb), M, K, N, xo, wo, yo)
    case 1n++p:
      (xa, xb) = fx
      (wa, wb) = fw
      (ya, yb) = fy
      +h = U32.shr(M)
      +rest = U32.sub(M, h)
      ra rb = plin.go(p, Array.fork(F32, xa), Array.fork(F32, wa), Array.fork(F32, ya), h, K, N, xo, wo, yo) plin.go(p, Array.fork(F32, xb), Array.fork(F32, wb), Array.fork(F32, yb), rest, K, N, (xo + (h * K) : U32), wo, (yo + (h * N) : U32))
      plin.join(ra, rb)

# depth so that leaves hold about 8 rows

def plin.depth(+M: U32) -> Nat:
  U32.log2(U32.max(1, U32.shrn(M, 3n)))

def plinear(x: Array<F32>, w: Array<F32>, y: Array<F32>, +M: U32, +K: U32, +N: U32, +xo: U32, +wo: U32, +yo: U32)
  -> (Array<F32> & Array<F32>) & Array<F32>:
  plin.go(plin.depth(M), Array.fork(F32, x), Array.fork(F32, w), Array.fork(F32, y), M, K, N, xo, wo, yo)

# -------------------------------------------------------------- layernorm

def ln.sum.go(n: Nat, +xi: U32, acc: F32, acc2: F32, r: Array<F32> & F32) -> (Array<F32> & F32) & F32:
  match n:
    case 0n:
      (x, +v) = r
      ((x, (acc + v : F32)), (acc2 + (v * v) : F32))
    case 1n+p:
      (x, +v) = r
      +xj = (xi + 1 : U32)
      ln.sum.go(p, xj, (acc + v : F32), (acc2 + (v * v) : F32), Array.get(F32, x, xj))

def ln.beta(w: Array<F32>, +bi: U32, has: Bool) -> Array<F32> & F32:
  match has:
    case True{}:
      Array.get(F32, w, bi)
    case False{}:
      (w, 0.0)

def ln.gb.fin(+g: F32, rb: Array<F32> & F32) -> (Array<F32> & F32) & F32:
  (w, beta) = rb
  ((w, g), beta)

# read gamma (already in rg) then beta -> ((w, gamma), beta)

def ln.gb(+bi: U32, has: Bool, rg: Array<F32> & F32) -> (Array<F32> & F32) & F32:
  (w, g) = rg
  ln.gb.fin(g, ln.beta(w, bi, has))

def ln.apply.go(n: Nat, +xi: U32, +gi: U32, +bi: U32, +has: Bool, +yi: U32, +mean: F32, +inv: F32, y: Array<F32>,
                rx: Array<F32> & F32, rgb: (Array<F32> & F32) & F32) -> (Array<F32> & Array<F32>) & Array<F32>:
  match n:
    case 0n:
      (x, xv) = rx
      ((w, g), beta) = rgb
      ((x, w), Array.set(F32, y, yi, ((((xv - mean) * inv) * g) + beta : F32)))
    case 1n+p:
      (x, xv) = rx
      ((w, g), beta) = rgb
      +xj = (xi + 1 : U32)
      +gj = (gi + 1 : U32)
      +bj = (bi + 1 : U32)
      ln.apply.go(p, xj, gj, bj, has, (yi + 1 : U32), mean, inv, Array.set(F32, y, yi, ((((xv - mean) * inv) * g) + beta : F32)),
        Array.get(F32, x, xj), ln.gb(bj, has, Array.get(F32, w, gj)))

def ln.row.fin(+d: U32, +xo: U32, +go: U32, +bo: U32, +has: Bool, +yo: U32, +eps: F32, w: Array<F32>, y: Array<F32>,
               r: (Array<F32> & F32) & F32) -> (Array<F32> & Array<F32>) & Array<F32>:
  ((x, s1), s2) = r
  +nf = U32.to_f32(d)
  +mean = (s1 / nf : F32)
  +var = ((s2 / nf) - (mean * mean) : F32)
  +inv = (1.0 / F32.sqrt((var + eps : F32)) : F32)
  ln.apply.go(U32.to_nat(U32.sub(d, 1)), xo, go, bo, has, yo, mean, inv, y, Array.get(F32, x, xo), ln.gb(bo, has, Array.get(F32, w, go)))

def ln.rows.go(m: Nat, +d: U32, +xo: U32, +go: U32, +bo: U32, +has: Bool, +yo: U32, +eps: F32,
               r: (Array<F32> & Array<F32>) & Array<F32>) -> (Array<F32> & Array<F32>) & Array<F32>:
  match m:
    case 0n:
      r
    case 1n+p:
      ((x, w), y) = r
      ln.rows.go(p, d, (xo + d : U32), go, bo, has, (yo + d : U32), eps,
        ln.row.fin(d, xo, go, bo, has, yo, eps, w, y, ln.sum.go(U32.to_nat(U32.sub(d, 1)), xo, 0.0, 0.0, Array.get(F32, x, xo))))

# LayerNorm over M rows of width d: gamma at w[go..], beta at w[bo..] when has_beta

def layernorm(x: Array<F32>, w: Array<F32>, y: Array<F32>, +M: U32, +d: U32, +xo: U32, +go: U32, +bo: U32, +has_beta: Bool, +yo: U32, +eps: F32)
  -> (Array<F32> & Array<F32>) & Array<F32>:
  ln.rows.go(U32.to_nat(M), d, xo, go, bo, has_beta, yo, eps, ((x, w), y))

# ------------------------------------------------------------------ GeGLU
# out[oo + m*I + i] = gelu(h[ho + m*2I + i]) * h[ho + m*2I + I + i]

def geglu.ag.fin(+a: F32, rg: Array<F32> & F32) -> (Array<F32> & F32) & F32:
  (h, g) = rg
  ((h, a), g)

# read a (already in ra) then the gate g at gi -> ((h, a), g)

def geglu.ag(+gi: U32, ra: Array<F32> & F32) -> (Array<F32> & F32) & F32:
  (h, a) = ra
  geglu.ag.fin(a, Array.get(F32, h, gi))

def geglu.go(n: Nat, +ai: U32, +gi: U32, +oi: U32, out: Array<F32>, rag: (Array<F32> & F32) & F32) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      ((h, a), g) = rag
      (h, Array.set(F32, out, oi, (gelu(a) * g : F32)))
    case 1n+p:
      ((h, a), g) = rag
      +aj = (ai + 1 : U32)
      +gj = (gi + 1 : U32)
      geglu.go(p, aj, gj, (oi + 1 : U32), Array.set(F32, out, oi, (gelu(a) * g : F32)), geglu.ag(gj, Array.get(F32, h, aj)))

def geglu.rows.go(m: Nat, +I: U32, +ho: U32, +oo: U32, r: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  match m:
    case 0n:
      r
    case 1n+p:
      (h, out) = r
      geglu.rows.go(p, I, (ho + (I * 2) : U32), (oo + I : U32), geglu.go(U32.to_nat(U32.sub(I, 1)), ho, (ho + I : U32), oo, out, geglu.ag((ho + I : U32), Array.get(F32, h, ho))))

def geglu(h: Array<F32>, out: Array<F32>, +M: U32, +I: U32, +ho: U32, +oo: U32) -> Array<F32> & Array<F32>:
  geglu.rows.go(U32.to_nat(M), I, ho, oo, (h, out))

# ------------------------------------------------------------------- RoPE
# rotate one (a, b) pair in place: x[ia] = a cos - b sin ; x[ib] = b cos + a sin

def rope.pair.fin(+ia: U32, +ib: U32, +c: F32, +s: F32, +a: F32, r: Array<F32> & F32) -> Array<F32>:
  (x, +b) = r
  Array.set(F32, Array.set(F32, x, ia, ((a * c) - (b * s) : F32)), ib, ((b * c) + (a * s) : F32))

def rope.pair(+ia: U32, +ib: U32, +c: F32, +s: F32, r: Array<F32> & F32) -> Array<F32>:
  (x, a) = r
  rope.pair.fin(ia, ib, c, s, a, Array.get(F32, x, ib))

# all pairs j < half of one head: angle = pos * theta^(-2j/Dh)

def rope.head.go(n: Nat, +j: U32, +base: U32, +half: U32, +posf: F32, +theta: F32, +dh: F32, x: Array<F32>) -> Array<F32>:
  match n:
    case 0n:
      x
    case 1n+p:
      +inv = F32.pow(theta, F32.neg(((U32.to_f32(j) * 2.0) / dh : F32)))
      +ang = (posf * inv : F32)
      rope.head.go(p, (j + 1 : U32), base, half, posf, theta, dh, rope.pair((base + j : U32), (base + j + half : U32), F32.cos(ang), F32.sin(ang), Array.get(F32, x, (base + j : U32))))

def rope.heads.go(n: Nat, +base: U32, +dh_u: U32, +posf: F32, +theta: F32, x: Array<F32>) -> Array<F32>:
  match n:
    case 0n:
      x
    case 1n+p:
      rope.heads.go(p, (base + dh_u : U32), dh_u, posf, theta, rope.head.go(U32.to_nat(U32.shr(dh_u)), 0, base, U32.shr(dh_u), posf, theta, U32.to_f32(dh_u), x))

# apply RoPE to `H` heads of width Dh for L tokens whose rows start at xo with stride (row width);
# position of token i is posf(i) (a template so it compiles away)

def rope.rows.go(~posf: U32 -> U32, n: Nat, +i: U32, +xo: U32, +stride: U32, +H: U32, +dh: U32, +theta: F32, x: Array<F32>) -> Array<F32>:
  match n:
    case 0n:
      x
    case 1n+p:
      rope.rows.go(~posf, p, (i + 1 : U32), (xo + stride : U32), stride, H, dh, theta, rope.heads.go(U32.to_nat(H), xo, dh, U32.to_f32(posf(i)), theta, x))

def rope(~posf: U32 -> U32, x: Array<F32>, +L: U32, +xo: U32, +stride: U32, +H: U32, +dh: U32, +theta: F32) -> Array<F32>:
  rope.rows.go(~posf, U32.to_nat(L), 0, xo, stride, H, dh, theta, x)

# ------------------------------------------------------------- attention
# dot of two rows of the SAME array: sum x[ai + c] * x[bi + c]

def sdot.b.fin(+a: F32, rb: Array<F32> & F32) -> (Array<F32> & F32) & F32:
  (x, b) = rb
  ((x, a), b)

def sdot.b(+bi: U32, ra: Array<F32> & F32) -> (Array<F32> & F32) & F32:
  (x, a) = ra
  sdot.b.fin(a, Array.get(F32, x, bi))

def sdot.go(k: Nat, +ai: U32, +bi: U32, acc: F32, rab: (Array<F32> & F32) & F32) -> Array<F32> & F32:
  match k:
    case 0n:
      ((x, a), b) = rab
      (x, (acc + (a * b) : F32))
    case 1n+p:
      ((x, a), b) = rab
      +aj = (ai + 1 : U32)
      +bj = (bi + 1 : U32)
      sdot.go(p, aj, bj, (acc + (a * b) : F32), sdot.b(bj, Array.get(F32, x, aj)))

def sdot(x: Array<F32>, +ai: U32, +bi: U32, +k: U32) -> Array<F32> & F32:
  sdot.go(U32.to_nat(U32.sub(k, 1)), ai, bi, 0.0, sdot.b(bi, Array.get(F32, x, ai)))

def sdot.scl(+scl: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  (x, v) = r
  (x, (v * scl : F32))

# score of query row qi against key row kj: scaled dot, or -1e9 when the key is outside the window

def att.score(x: Array<F32>, +qi: U32, +kj: U32, +dh: U32, +scl: F32, masked: Bool) -> Array<F32> & F32:
  match masked:
    case True{}:
      (x, F32.neg(1000000000.0))
    case False{}:
      sdot.scl(scl, sdot(x, qi, kj, dh))

def att.masked(+win: U32, +qpos: U32, +j: U32) -> Bool:
  Bool.and(U32.is_gt(win, 0), U32.is_gt(picku(U32.is_lt(j, qpos), U32.sub(qpos, j), U32.sub(j, qpos)), win))

# scores of query row qi against keys j < L into s[j]  (r carries the pending score of key j)

def att.sc.go(n: Nat, +j: U32, +qi: U32, +ko: U32, +stride: U32, +dh: U32, +scl: F32, +qpos: U32, +win: U32, s: Array<F32>,
              r: Array<F32> & F32) -> (Array<F32> & Array<F32>):
  match n:
    case 0n:
      (x, v) = r
      (x, Array.set(F32, s, j, v))
    case 1n+p:
      (x, v) = r
      +jn = (j + 1 : U32)
      att.sc.go(p, jn, qi, ko, stride, dh, scl, qpos, win, Array.set(F32, s, j, v),
        att.score(x, qi, (ko + (jn * stride) : U32), dh, scl, att.masked(win, qpos, jn)))

def att.max.go(n: Nat, +j: U32, +m: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  match n:
    case 0n:
      (s, v) = r
      (s, F32.max(m, v))
    case 1n+p:
      (s, v) = r
      +jn = (j + 1 : U32)
      att.max.go(p, jn, F32.max(m, v), Array.get(F32, s, jn))

def att.exp.step(+j: U32, +m: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  (s, v) = r
  +e = F32.exp((v - m : F32))
  (Array.set(F32, s, j, e), e)

def att.exp.go(n: Nat, +j: U32, +m: F32, +acc: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  match n:
    case 0n:
      (s, e) = r
      (s, (acc + e : F32))
    case 1n+p:
      (s, e) = r
      +jn = (j + 1 : U32)
      att.exp.go(p, jn, m, (acc + e : F32), att.exp.step(jn, m, Array.get(F32, s, jn)))

def att.norm.fin(+L: U32, r: Array<F32> & F32) -> Array<F32>:
  (s, total) = r
  scale(s, 0, L, (1.0 / total : F32))

def att.norm.max(+L: U32, r: Array<F32> & F32) -> Array<F32>:
  (s, +m) = r
  att.norm.fin(L, att.exp.go(U32.to_nat(U32.sub(L, 1)), 0, m, 0.0, att.exp.step(0, m, Array.get(F32, s, 0))))

# softmax in place over s[0..L)

def softmax_row(s: Array<F32>, +L: U32) -> Array<F32>:
  att.norm.max(L, att.max.go(U32.to_nat(U32.sub(L, 1)), 0, F32.neg(1000000000000.0), Array.get(F32, s, 0)))

# out[oi + c] += p * x[vj + c] for c < dh

def att.acc.step(+oi: U32, +p: F32, out: Array<F32>, r: Array<F32> & F32) -> Array<F32> & Array<F32>:
  (x, v) = r
  (x, axpy.step(oi, (p * v : F32), Array.get(F32, out, oi)))

def att.acc.go(n: Nat, +c: U32, +vj: U32, +oi: U32, +p: F32, r: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      r
    case 1n+q:
      (x, out) = r
      att.acc.go(q, (c + 1 : U32), vj, oi, p, att.acc.step((oi + c : U32), p, out, Array.get(F32, x, (vj + c : U32))))

# out row += sum_j s[j] * v_j     (rxo carries (x, out); rs carries the pending read of s[j])

def att.val.go(n: Nat, +j: U32, +vo: U32, +stride: U32, +dh: U32, +oi: U32, rxo: Array<F32> & Array<F32>, rs: Array<F32> & F32)
  -> (Array<F32> & Array<F32>) & Array<F32>:
  match n:
    case 0n:
      (x, out) = rxo
      (s, p) = rs
      (att.acc.go(U32.to_nat(dh), 0, vo, oi, p, (x, out)), s)
    case 1n+q:
      (x, out) = rxo
      (s, p) = rs
      +jn = (j + 1 : U32)
      att.val.go(q, jn, (vo + stride : U32), stride, dh, oi, att.acc.go(U32.to_nat(dh), 0, vo, oi, p, (x, out)), Array.get(F32, s, jn))

# repack ((x, out), s) with the weights handle -> (((x, w), out), s)

def att.row.pack(w: Array<F32>, r: (Array<F32> & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  ((x, out), s) = r
  (((x, w), out), s)

def att.row.fin(+L: U32, +vo: U32, +stride: U32, +dh: U32, +oi: U32, w: Array<F32>, out: Array<F32>, r: Array<F32> & Array<F32>)
  -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  (x, s) = r
  att.row.pack(w, att.val.go(U32.to_nat(U32.sub(L, 1)), 0, vo, stride, dh, oi, (x, out), Array.get(F32, softmax_row(s, L), 0)))

# one query row i of one head (ostride = width of an output row = H*dh)

def att.row(x: Array<F32>, w: Array<F32>, out: Array<F32>, s: Array<F32>, +L: U32, +i: U32, +qo: U32, +ko: U32, +vo: U32,
            +stride: U32, +ostride: U32, +dh: U32, +oo: U32, +win: U32, +scl: F32) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  +qi = (qo + (i * stride) : U32)
  att.row.fin(L, vo, stride, dh, (oo + (i * ostride) : U32), w, out,
    att.sc.go(U32.to_nat(U32.sub(L, 1)), 0, qi, ko, stride, dh, scl, i, win, s, att.score(x, qi, ko, dh, scl, att.masked(win, i, 0))))

def att.rows.go(n: Nat, +i: U32, +L: U32, +qo: U32, +ko: U32, +vo: U32, +stride: U32, +ostride: U32, +dh: U32, +oo: U32, +win: U32, +scl: F32,
                r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  match n:
    case 0n:
      r
    case 1n+p:
      (((x, w), out), s) = r
      att.rows.go(p, (i + 1 : U32), L, qo, ko, vo, stride, ostride, dh, oo, win, scl, att.row(x, w, out, s, L, i, qo, ko, vo, stride, ostride, dh, oo, win, scl))

# all heads: qkv rows are [q(H*dh) | k(H*dh) | v(H*dh)] with row stride 3*H*dh ; out rows of width H*dh

def att.heads.go(n: Nat, +h: U32, +L: U32, +qkvo: U32, +H: U32, +dh: U32, +oo: U32, +win: U32, +scl: F32,
                 r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  match n:
    case 0n:
      r
    case 1n+p:
      +hd = (H * dh : U32)
      +stride = (hd * 3 : U32)
      +qo = (qkvo + (h * dh) : U32)
      att.heads.go(p, (h + 1 : U32), L, qkvo, H, dh, oo, win, scl,
        att.rows.go(U32.to_nat(L), 0, L, qo, (qo + hd : U32), (qo + (hd * 2) : U32), stride, hd, dh, (oo + (h * dh) : U32), win, scl, r))

# self-attention over L tokens: qkv (L x 3Hdh at qkvo) -> out (L x Hdh at oo); window 0 = full attention.
# `s` is a scratch array with >= L slots; `w` (weights handle) passes through untouched.

def attention(qkv: Array<F32>, w: Array<F32>, out: Array<F32>, s: Array<F32>, +L: U32, +qkvo: U32, +H: U32, +dh: U32, +oo: U32, +win: U32)
  -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  +scl = (1.0 / F32.sqrt(U32.to_f32(dh)) : F32)
  att.heads.go(U32.to_nat(H), 0, L, qkvo, H, dh, oo, win, scl, (((qkv, w), fill(out, oo, (L * H * dh : U32), 0.0)), s))

# --------------------------------------------------------- cross-attention
# q rows (stride qs) attend over kv rows (stride kvs, key at +0, value at +vo_in) of another array.

def xatt.sc.go(n: Nat, +j: U32, +qi: U32, +ko: U32, +kvs: U32, +dh: U32, +scl: F32, s: Array<F32>, r: (Array<F32> & Array<F32>) & F32)
  -> (Array<F32> & Array<F32>) & Array<F32>:
  match n:
    case 0n:
      ((q, kv), v) = r
      ((q, kv), Array.set(F32, s, j, (v * scl : F32)))
    case 1n+p:
      ((q, kv), v) = r
      +jn = (j + 1 : U32)
      xatt.sc.go(p, jn, qi, ko, kvs, dh, scl, Array.set(F32, s, j, (v * scl : F32)), dot(q, kv, qi, (ko + (jn * kvs) : U32), dh))

def xatt.val.go(n: Nat, +j: U32, +vo: U32, +kvs: U32, +dh: U32, +oi: U32, rxo: Array<F32> & Array<F32>, rs: Array<F32> & F32)
  -> (Array<F32> & Array<F32>) & Array<F32>:
  match n:
    case 0n:
      (kv, out) = rxo
      (s, p) = rs
      (att.acc.go(U32.to_nat(dh), 0, vo, oi, p, (kv, out)), s)
    case 1n+q:
      (kv, out) = rxo
      (s, p) = rs
      +jn = (j + 1 : U32)
      xatt.val.go(q, jn, (vo + kvs : U32), kvs, dh, oi, att.acc.go(U32.to_nat(dh), 0, vo, oi, p, (kv, out)), Array.get(F32, s, jn))

def xatt.row.pack(q: Array<F32>, r: (Array<F32> & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  ((kv, out), s) = r
  (((q, kv), out), s)

def xatt.row.fin(+Lk: U32, +vo: U32, +kvs: U32, +dh: U32, +oi: U32, out: Array<F32>, r: (Array<F32> & Array<F32>) & Array<F32>)
  -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  ((q, kv), s) = r
  xatt.row.pack(q, xatt.val.go(U32.to_nat(U32.sub(Lk, 1)), 0, vo, kvs, dh, oi, (kv, out), Array.get(F32, softmax_row(s, Lk), 0)))

# one query row i of one head: q row at qo + i*qs (+ head offset already in qo); keys at ko + j*kvs; values at ko + vofs + j*kvs

def xatt.row(q: Array<F32>, kv: Array<F32>, out: Array<F32>, s: Array<F32>, +Lk: U32, +i: U32, +qo: U32, +qs: U32, +ko: U32, +vofs: U32,
             +kvs: U32, +dh: U32, +oo: U32, +os: U32, +scl: F32) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  +qi = (qo + (i * qs) : U32)
  xatt.row.fin(Lk, (ko + vofs : U32), kvs, dh, (oo + (i * os) : U32), out,
    xatt.sc.go(U32.to_nat(U32.sub(Lk, 1)), 0, qi, ko, kvs, dh, scl, s, dot(q, kv, qi, ko, dh)))

def xatt.rows.go(n: Nat, +i: U32, +Lk: U32, +qo: U32, +qs: U32, +ko: U32, +vofs: U32, +kvs: U32, +dh: U32, +oo: U32, +os: U32, +scl: F32,
                 r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  match n:
    case 0n:
      r
    case 1n+p:
      (((q, kv), out), s) = r
      xatt.rows.go(p, (i + 1 : U32), Lk, qo, qs, ko, vofs, kvs, dh, oo, os, scl, xatt.row(q, kv, out, s, Lk, i, qo, qs, ko, vofs, kvs, dh, oo, os, scl))

def xatt.heads.go(n: Nat, +h: U32, +Lq: U32, +Lk: U32, +H: U32, +dh: U32, +kvo: U32, +scl: F32,
                  r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  match n:
    case 0n:
      r
    case 1n+p:
      +hd = (H * dh : U32)
      +ho = (h * dh : U32)
      xatt.heads.go(p, (h + 1 : U32), Lq, Lk, H, dh, kvo, scl,
        xatt.rows.go(U32.to_nat(Lq), 0, Lk, ho, hd, (kvo + ho : U32), hd, (hd * 2 : U32), dh, ho, hd, scl, r))

# cross-attention: q (Lq x Hdh) over kv rows starting at kvo (Lk x 2Hdh: [k | v]) -> out (Lq x Hdh); s is scratch (>= Lk)

def xattention(q: Array<F32>, kv: Array<F32>, out: Array<F32>, s: Array<F32>, +Lq: U32, +Lk: U32, +H: U32, +dh: U32, +kvo: U32)
  -> ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>:
  +scl = (1.0 / F32.sqrt(U32.to_f32(dh)) : F32)
  xatt.heads.go(U32.to_nat(H), 0, Lq, Lk, H, dh, kvo, scl, (((q, kv), fill(out, 0, (Lq * H * dh : U32), 0.0)), s))

# ------------------------------------------------------------ elementwise
# in-place GELU over a[o .. o+n)

def gelu.step(+i: U32, r: Array<F32> & F32) -> Array<F32>:
  (a, v) = r
  Array.set(F32, a, i, gelu(v))

def gelu.go(n: Nat, +i: U32, a: Array<F32>) -> Array<F32>:
  match n:
    case 0n:
      a
    case 1n+p:
      gelu.go(p, (i + 1 : U32), gelu.step(i, Array.get(F32, a, i)))

def gelu_rows(a: Array<F32>, +o: U32, +n: U32) -> Array<F32>:
  gelu.go(U32.to_nat(n), o, a)

# y[yo + i] = y[yo + i] + s * x[xo + i]

def axpys.step(+yi: U32, +s: F32, +xv: F32, r: Array<F32> & F32) -> Array<F32>:
  (y, yv) = r
  Array.set(F32, y, yi, (yv + (s * xv) : F32))

def axpys.go(n: Nat, +xi: U32, +yi: U32, +s: F32, y: Array<F32>, r: Array<F32> & F32) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      (x, xv) = r
      (x, axpys.step(yi, s, xv, Array.get(F32, y, yi)))
    case 1n+p:
      (x, xv) = r
      +xj = (xi + 1 : U32)
      axpys.go(p, xj, (yi + 1 : U32), s, axpys.step(yi, s, xv, Array.get(F32, y, yi)), Array.get(F32, x, xj))

def axpys(x: Array<F32>, y: Array<F32>, +xo: U32, +yo: U32, +n: U32, +s: F32) -> Array<F32> & Array<F32>:
  axpys.go(U32.to_nat(U32.sub(n, 1)), xo, yo, s, y, Array.get(F32, x, xo))

# sum of x[xo..xo+n)

def vsum.go(n: Nat, +xi: U32, acc: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  match n:
    case 0n:
      (x, v) = r
      (x, (acc + v : F32))
    case 1n+p:
      (x, v) = r
      +xj = (xi + 1 : U32)
      vsum.go(p, xj, (acc + v : F32), Array.get(F32, x, xj))

def vsum(x: Array<F32>, +xo: U32, +n: U32) -> Array<F32> & F32:
  vsum.go(U32.to_nat(U32.sub(n, 1)), xo, 0.0, Array.get(F32, x, xo))

# ------------------------------------------- RoPE with positions from an array
# position of row i is pa[po + i] (U32); used by question sequences whose option
# blocks share a base position

def rope.p.rows.go(n: Nat, +i: U32, +xo: U32, +stride: U32, +H: U32, +dh: U32, +theta: F32, +po: U32, x: Array<F32>, r: Array<U32> & U32)
  -> Array<F32> & Array<U32>:
  match n:
    case 0n:
      (pa, pos) = r
      (rope.heads.go(U32.to_nat(H), xo, dh, U32.to_f32(pos), theta, x), pa)
    case 1n+p:
      (pa, pos) = r
      +j = (i + 1 : U32)
      rope.p.rows.go(p, j, (xo + stride : U32), stride, H, dh, theta, po, rope.heads.go(U32.to_nat(H), xo, dh, U32.to_f32(pos), theta, x),
        Array.get(U32, pa, (po + j : U32)))

def rope_p(x: Array<F32>, pa: Array<U32>, +L: U32, +xo: U32, +stride: U32, +H: U32, +dh: U32, +theta: F32, +po: U32) -> Array<F32> & Array<U32>:
  rope.p.rows.go(U32.to_nat(U32.sub(L, 1)), 0, xo, stride, H, dh, theta, po, x, Array.get(U32, pa, po))

# ------------------------------------------------ block-restricted attention
# key j is visible to query i iff blk[j] == 0 (prefix) or blk[j] == blk[i] (same option block)

def att.bm.set(blocked: Bool, +j: U32, s: Array<F32>) -> Array<F32>:
  match blocked:
    case True{}:
      Array.set(F32, s, j, F32.neg(1000000000.0))
    case False{}:
      s

def att.bm.go(n: Nat, +j: U32, +bo: U32, +bi: U32, s: Array<F32>, r: Array<U32> & U32) -> Array<F32> & Array<U32>:
  match n:
    case 0n:
      (blk, +bj) = r
      (att.bm.set(Bool.and(U32.is_ne(bj, 0), U32.is_ne(bj, bi)), j, s), blk)
    case 1n+p:
      (blk, +bj) = r
      +jn = (j + 1 : U32)
      att.bm.go(p, jn, bo, bi, att.bm.set(Bool.and(U32.is_ne(bj, 0), U32.is_ne(bj, bi)), j, s), Array.get(U32, blk, (bo + jn : U32)))

def att.row.b.fin(blk: Array<U32>, r: ((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) -> (((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) & Array<U32>:
  (r, blk)

def att.row.b.mask(+L: U32, +vo: U32, +stride: U32, +dh: U32, +oi: U32, x: Array<F32>, w: Array<F32>, out: Array<F32>, r: Array<F32> & Array<U32>)
  -> (((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) & Array<U32>:
  (s, blk) = r
  att.row.b.fin(blk, att.row.fin(L, vo, stride, dh, oi, w, out, (x, s)))

def att.row.b.bi(+L: U32, +vo: U32, +stride: U32, +dh: U32, +oi: U32, +bo: U32, x: Array<F32>, w: Array<F32>, out: Array<F32>, s: Array<F32>, r: Array<U32> & U32)
  -> (((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) & Array<U32>:
  (blk, +bi) = r
  att.row.b.mask(L, vo, stride, dh, oi, x, w, out, att.bm.go(U32.to_nat(U32.sub(L, 1)), 0, bo, bi, s, Array.get(U32, blk, bo)))

def att.row.b.sc(+L: U32, +i: U32, +vo: U32, +stride: U32, +dh: U32, +oi: U32, +bo: U32, w: Array<F32>, out: Array<F32>, blk: Array<U32>, r: Array<F32> & Array<F32>)
  -> (((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) & Array<U32>:
  (x, s) = r
  att.row.b.bi(L, vo, stride, dh, oi, bo, x, w, out, s, Array.get(U32, blk, (bo + i : U32)))

# one query row i of one head, block mask read from blk[bo ..]

def att.row.b(x: Array<F32>, w: Array<F32>, out: Array<F32>, s: Array<F32>, blk: Array<U32>, +L: U32, +i: U32, +qo: U32, +ko: U32, +vo: U32,
              +stride: U32, +ostride: U32, +dh: U32, +oo: U32, +win: U32, +scl: F32, +bo: U32)
  -> (((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) & Array<U32>:
  +qi = (qo + (i * stride) : U32)
  att.row.b.sc(L, i, vo, stride, dh, (oo + (i * ostride) : U32), bo, w, out, blk,
    att.sc.go(U32.to_nat(U32.sub(L, 1)), 0, qi, ko, stride, dh, scl, i, win, s, att.score(x, qi, ko, dh, scl, att.masked(win, i, 0))))

def att.rows.b.go(n: Nat, +i: U32, +L: U32, +qo: U32, +ko: U32, +vo: U32, +stride: U32, +ostride: U32, +dh: U32, +oo: U32, +win: U32, +scl: F32, +bo: U32,
                  r: (((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) & Array<U32>) -> (((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) & Array<U32>:
  match n:
    case 0n:
      r
    case 1n+p:
      ((((x, w), out), s), blk) = r
      att.rows.b.go(p, (i + 1 : U32), L, qo, ko, vo, stride, ostride, dh, oo, win, scl, bo, att.row.b(x, w, out, s, blk, L, i, qo, ko, vo, stride, ostride, dh, oo, win, scl, bo))

def att.heads.b.go(n: Nat, +h: U32, +L: U32, +qkvo: U32, +H: U32, +dh: U32, +oo: U32, +win: U32, +scl: F32, +bo: U32,
                   r: (((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) & Array<U32>) -> (((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) & Array<U32>:
  match n:
    case 0n:
      r
    case 1n+p:
      +hd = (H * dh : U32)
      +stride = (hd * 3 : U32)
      +qo = (qkvo + (h * dh) : U32)
      att.heads.b.go(p, (h + 1 : U32), L, qkvo, H, dh, oo, win, scl, bo,
        att.rows.b.go(U32.to_nat(L), 0, L, qo, (qo + hd : U32), (qo + (hd * 2) : U32), stride, hd, dh, (oo + (h * dh) : U32), win, scl, bo, r))

# self-attention with a block mask read from blk[bo .. bo+L)

def attention_b(qkv: Array<F32>, w: Array<F32>, out: Array<F32>, s: Array<F32>, blk: Array<U32>, +L: U32, +qkvo: U32, +H: U32, +dh: U32, +oo: U32, +win: U32, +bo: U32)
  -> (((Array<F32> & Array<F32>) & Array<F32>) & Array<F32>) & Array<U32>:
  +scl = (1.0 / F32.sqrt(U32.to_f32(dh)) : F32)
  att.heads.b.go(U32.to_nat(H), 0, L, qkvo, H, dh, oo, win, scl, bo, ((((qkv, w), fill(out, oo, (L * H * dh : U32), 0.0)), s), blk))
