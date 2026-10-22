import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PARITY = ROOT / "bend" / "groot" / "parity.mjs"
FIXTURE = ROOT / "tests" / "fixtures" / "parity"


@pytest.mark.skipif(
    shutil.which("node") is None or not PARITY.exists() or not (FIXTURE / "model.gbw").exists(),
    reason="node, bend engine or parity fixture missing",
)
def test_bend_forward_parity():
    proc = subprocess.run(["node", str(PARITY)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
