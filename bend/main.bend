# Groot 1 — native command-line engine (CPU threads / GPU via Bend's runtime).
#
#   GROOT_WEIGHTS=model.gbw GROOT_WOFF=<data offset> GROOT_WDEPTH=<log2 floats> \
#   GROOT_REQ=req.bin GROOT_RDEPTH=<log2 words> GROOT_NOUT=<values> ./groot +RTS --threads 8
#
# Prints the output vector [probs ; confidence logits ; logits], one value per line.
import Base
import ./groot/groot.bend as G

def Blob.load_f32(path: String, offset: U32, depth: U32) -> IO(Array<F32>):
  import "./effs/blob_load_f32.c"
  import "./effs/blob_load_f32.js"

def Blob.load_u32(path: String, offset: U32, depth: U32) -> IO(Array<U32>):
  import "./effs/blob_load_u32.c"
  import "./effs/blob_load_u32.js"

def env_u32(name: String) -> IO(U32):
  do IO<U32>:
    s : String <- IO.try(String, IO.get_env(name))
    return Maybe.default(&2, U32, U32.read(s), 0)

def show.go(k: Nat, +i: U32, r: Array<F32> & F32) -> IO(Unit):
  match k:
    case 0n:
      (a, v) = r
      IO.print(F32.show(v))
    case 1n+p:
      (a, v) = r
      +j = (i + 1 : U32)
      do IO<Unit>:
        IO.print(F32.show(v))
        show.go(p, j, Array.get(F32, a, j))

def show(out: Array<F32>, +n: U32) -> IO(Unit):
  show.go(U32.to_nat(U32.sub(n, 1)), 0, Array.get(F32, out, 0))

def run(wpath: String, +woff: U32, +wdepth: U32, bpath: String, +bdepth: U32, +nout: U32) -> IO(Unit):
  do IO<Unit>:
    w : Array<F32> <- Blob.load_f32(wpath, woff, wdepth)
    b : Array<U32> <- Blob.load_u32(bpath, 0, bdepth)
    t0 : Nat <- IO.now()
    out : Array<F32> = G.forward(w, b)
    show(out, nout)
    t1 : Nat <- IO.now()
    IO.print_err("groot forward ms = " ++ Nat.show(Nat.sub(t1, t0)))

def main() -> IO(Unit):
  do IO<Unit>:
    wpath : String <- IO.try(String, IO.get_env("GROOT_WEIGHTS"))
    woff : U32 <- env_u32("GROOT_WOFF")
    wdepth : U32 <- env_u32("GROOT_WDEPTH")
    bpath : String <- IO.try(String, IO.get_env("GROOT_REQ"))
    bdepth : U32 <- env_u32("GROOT_RDEPTH")
    nout : U32 <- env_u32("GROOT_NOUT")
    run(wpath, woff, wdepth, bpath, bdepth, nout)
