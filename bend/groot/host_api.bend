# Pure entry points for JavaScript hosts (bend host_api.bend -o host_api.mjs).
# Arrays cross zero-copy: Float32Array <-> Array<F32>, Uint32Array <-> Array<U32>.
import Base
import ./kernels.bend as K
import ./encoder.bend as E
import ./groot.bend as G

def third(r: (Array<F32> & Array<U32>) & Array<F32>) -> Array<F32>:
  ((w, ids), h) = r
  h

# hidden states (n x d) of the token ids at ids[ido..ido+n)
def encode(w: Array<F32>, ids: Array<U32>, ido: U32, n: U32) -> Array<F32>:
  third(E.encode(w, ids, ido, n))

# full forward: weights + u32 request blob -> [probs(nopt) ; conf logits(nq) ; raw logits(nopt)]
def run_blob(w: Array<F32>, b: Array<U32>) -> Array<F32>:
  G.forward(w, b)
