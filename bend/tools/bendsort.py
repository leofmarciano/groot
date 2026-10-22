#!/usr/bin/env python3
"""Topologically sort the defs of a Bend file so every def is declared after the defs it calls.

Bend resolves calls only to defs declared above (no forward references, no mutual recursion),
so the natural top-down writing order must be flipped. Types and imports stay at the top in
their original order; defs are emitted dependencies-first with the original order as tiebreak.
Usage: bendsort.py FILE.bend [--check]
"""
import re
import sys
from pathlib import Path

DEF_RE = re.compile(r"^(?:@unsafe\s*\n)?def\s+([A-Za-z_][\w.]*)\s*[\(?]", re.M)


def split_blocks(src: str):
    lines = src.split("\n")
    blocks, cur, kind = [], [], None
    for line in lines:
        starts_def = line.startswith("def ") or line.startswith("@unsafe")
        starts_other = line.startswith("type ") or line.startswith("import ") or line.startswith("law ")
        if (starts_def or starts_other) and cur and not (kind == "def" and line.startswith("def ") and cur and cur[-1].startswith("@unsafe")):
            blocks.append((kind, "\n".join(cur)))
            cur = []
        if starts_def:
            kind = "def"
        elif starts_other:
            kind = "other"
        elif not cur:
            kind = kind or "other"
        cur.append(line)
    if cur:
        blocks.append((kind, "\n".join(cur)))
    return blocks


def def_name(block: str):
    m = DEF_RE.search(block)
    return m.group(1) if m else None


def main():
    path = Path(sys.argv[1])
    check = "--check" in sys.argv
    src = path.read_text()
    blocks = split_blocks(src)
    defs = [(def_name(b), i, b) for i, (k, b) in enumerate(blocks) if k == "def"]
    others = [(i, b) for i, (k, b) in enumerate(blocks) if k != "def"]
    names = [n for n, _, _ in defs if n]
    body_of = {n: b for n, _, b in defs}
    order_of = {n: i for n, i, _ in defs}
    deps = {}
    for n in names:
        body = body_of[n].split("\n", 1)[1] if "\n" in body_of[n] else ""
        found = set()
        for m in names:
            if m == n:
                continue
            if re.search(r"(?<![\w.])" + re.escape(m) + r"(?=\s*\()", body) or re.search(r"~" + re.escape(m) + r"\b", body):
                found.add(m)
        deps[n] = found
    emitted, out, temp = set(), [], set()

    def visit(n, stack):
        if n in emitted:
            return
        if n in temp:
            raise SystemExit(f"cycle (mutual recursion) through: {' -> '.join(stack + [n])}")
        temp.add(n)
        for m in sorted(deps[n], key=lambda x: order_of[x]):
            visit(m, stack + [n])
        temp.discard(n)
        emitted.add(n)
        out.append(body_of[n])

    for n in names:
        visit(n, [])
    header = "\n".join(b for _, b in others).rstrip("\n")
    result = header + "\n\n" + "\n".join(x.rstrip("\n") + "\n" for x in out)
    if check:
        changed = result != src
        print("needs reorder" if changed else "already sorted")
        return
    path.write_text(result)
    print(f"sorted {len(names)} defs in {path}")


if __name__ == "__main__":
    main()
