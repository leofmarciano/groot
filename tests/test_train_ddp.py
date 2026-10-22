"""2-process CPU DDP smoke test (gloo): no hang across an epoch boundary, and resume works."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run(tmp: Path, *extra: str, port: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "--master_port", port, "-m", "groot1.train",
           "--tiny", "--train", str(tmp / "data" / "train.jsonl"), "--val", str(tmp / "data" / "val.jsonl"), "--out", str(tmp / "run"),
           "--batch-size", "4", "--epochs", "2", "--log-every", "1", "--eval-every", "100", "--eval-rows", "4", "--final-eval-rows", "8",
           "--save-every", "3", "--warmup", "1", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
                          "HOME": str(Path.home()), "TOKENIZERS_PARALLELISM": "false"}, cwd=ROOT)


@pytest.mark.slow
def test_ddp_epoch_boundary_and_resume(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tiny_data.py"), "--out", str(tmp_path / "data"), "--n", "37"], check=True,
                   capture_output=True)
    # 37 rows, world 2, batch 4 -> 4 batches/rank/epoch; 6 steps crosses into epoch 1
    first = _run(tmp_path, "--max-steps", "3", port="29531")
    assert first.returncode == 0, first.stdout[-2000:] + first.stderr[-2000:]
    state = json.loads((tmp_path / "run" / "ckpt" / "trainer_state.json").read_text())
    assert state["step"] == 3
    second = _run(tmp_path, "--max-steps", "6", "--resume", port="29532")
    assert second.returncode == 0, second.stdout[-2000:] + second.stderr[-2000:]
    assert '"resumed"' in second.stdout
    logs = [json.loads(l) for l in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    steps = [r["step"] for r in logs if "step" in r and "loss" in r]
    assert max(steps) == 6 and any(r.get("epoch") == 1 for r in logs if "loss" in r)
    assert (tmp_path / "run" / "final" / "model.safetensors").exists()
