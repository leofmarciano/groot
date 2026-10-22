"""Native Bend engine parity: compile bend/main.bend, run it on the parity fixture, compare with PyTorch."""
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "parity"
BEND = shutil.which("bend") or str(Path.home() / ".bend" / "bin" / "bend")


@pytest.mark.skipif(not Path(BEND).exists() or not (FIXTURE / "model.gbw").exists(), reason="bend toolchain or fixture missing")
def test_native_forward_parity(tmp_path):
    from groot1.export import read_gbw_header

    binary = tmp_path / "groot_native"
    build = subprocess.run([BEND, str(ROOT / "bend" / "main.bend"), "-o", str(binary)], capture_output=True, text=True,
                           timeout=600, env={**os.environ, "BEND_NO_TELEMETRY": "1"})
    assert binary.exists(), build.stdout + build.stderr
    header = read_gbw_header(FIXTURE / "model.gbw")
    meta = json.loads((FIXTURE / "meta.json").read_text())
    total = sum(t["numel"] for t in header["tensors"])
    words = np.fromfile(FIXTURE / "req.bin", dtype="<u4").size
    env = {
        **os.environ,
        "GROOT_WEIGHTS": str(FIXTURE / "model.gbw"),
        "GROOT_WOFF": str(meta["data_offset"]),
        "GROOT_WDEPTH": str(math.ceil(math.log2(total))),
        "GROOT_REQ": str(FIXTURE / "req.bin"),
        "GROOT_RDEPTH": str(math.ceil(math.log2(words))),
        "GROOT_NOUT": str(meta["ref_len"]),
    }
    run = subprocess.run([str(binary), "+RTS", "--threads", "4"], capture_output=True, text=True, timeout=300, env=env)
    assert run.returncode == 0, run.stderr
    out = np.array([float(x) for x in run.stdout.split()])
    ref = np.load(FIXTURE / "ref.npy")
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < 2e-3, np.abs(out - ref).max()
