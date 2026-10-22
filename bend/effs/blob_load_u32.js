function blob_load_u32(path, off, depth) {
  const fs = require("fs");
  const n = 1 << depth;
  const buf = Buffer.alloc(n * 4);
  const fd = fs.openSync(path, "r");
  fs.readSync(fd, buf, 0, n * 4, off);
  fs.closeSync(fd);
  return new Uint32Array(buf.buffer, buf.byteOffset, n);
}
io_eff(CID(Blob.load_u32), blob_load_u32);
