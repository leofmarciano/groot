// Full-model parity: Bend forward vs the PyTorch reference on tests/fixtures/parity.
// Regenerate the fixture with `python scripts/make_parity_fixture.py`, then
// `bend host_api.bend -o host_api.mjs && node parity.mjs`.
import fs from "fs";
import path from "path";
import url from "url";
import M from "./host_api.mjs";

const here = path.dirname(url.fileURLToPath(import.meta.url));
const CK = path.join(here, "../../tests/fixtures/parity");
const meta = JSON.parse(fs.readFileSync(path.join(CK, "meta.json")));

const wbuf = fs.readFileSync(path.join(CK, "model.gbw"));
const w = new Float32Array(wbuf.buffer, wbuf.byteOffset + meta.data_offset, (wbuf.length - meta.data_offset) >> 2);

const bbuf = fs.readFileSync(path.join(CK, "req.bin"));
const b = new Uint32Array(bbuf.buffer, bbuf.byteOffset, bbuf.length >> 2);

const rbuf = fs.readFileSync(path.join(CK, "ref.npy"));
const hl = rbuf.readUInt16LE(8);
const ref = new Float32Array(rbuf.buffer, rbuf.byteOffset + 10 + hl, meta.ref_len);

const t0 = performance.now();
const out = M.run_blob(w, b);
const dt = performance.now() - t0;

let md = 0, mr = 0, at = -1;
for (let i = 0; i < ref.length; i++) {
  const d = Math.abs(out[i] - ref[i]);
  if (d > md) { md = d; at = i; }
  mr = Math.max(mr, Math.abs(ref[i]));
}
console.log(`forward parity: max|diff| = ${md.toExponential(2)} at ${at}  max|ref| = ${mr.toFixed(3)}  ms = ${dt.toFixed(1)}  ${md < 2e-3 ? "OK" : "FAIL"}`);
console.log("probs:", Array.from(out.slice(0, meta.n_options), (x) => x.toFixed(4)).join(" "));
console.log("conf :", Array.from(out.slice(meta.n_options, meta.n_options + meta.n_questions), (x) => x.toFixed(4)).join(" "));
process.exit(md < 2e-3 ? 0 : 1);
