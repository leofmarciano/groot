function blob_load_f32(path, off, depth) {
  const fs = require("fs");
  const n = 1 << depth;
  const buf = Buffer.alloc(n * 4);
  const fd = fs.openSync(path, "r");
  fs.readSync(fd, buf, 0, n * 4, off);
  fs.closeSync(fd);
  return new Float32Array(buf.buffer, buf.byteOffset, n);
}
io_eff(CID(Blob.load_f32), blob_load_f32);
