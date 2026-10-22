#include <stdio.h>
#include <stdlib.h>
// Blob.load_u32(path: String, offset_bytes: U32, depth: U32) -> IO(Array<U32>)
// Reads 4 * 2^depth little-endian words at offset_bytes into a fresh array block (zero-padded past EOF).
// Uses Bend 2.0.x runtime internals (blk_new / blk_ptr / blk_loc); rebuild on every Bend upgrade.
Term blob_load_u32_run(Env e, Term* f, IoWork* w) {
  u64 len = 0;
  char* path = io_cstr(e, f[0], &len);
  u32 off = (u32)f[1];
  u32 depth = (u32)f[2];
  Term zero = 0;
  Term arr = blk_new(e, false, depth, 0, 1, &zero);
  FILE* fp = fopen(path, "rb");
  free(path);
  if (fp) {
    fseek(fp, off, SEEK_SET);
    DEV u32a* dst = blk_ptr(e.mem, blk_loc(e.mem, arr), 0);
    size_t got = fread((void*)dst, 4, 1ull << depth, fp);
    (void)got;
    fclose(fp);
  }
  return arr;
}
static void __attribute__((constructor)) blob_load_u32_use(void) {
  io_eff(CID(Blob.load_u32), blob_load_u32_run);
}
