import numpy as np
import pytest
import torch
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "parity"
CKPT = FIXTURE / "ckpt"

pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="parity fixture missing")


@pytest.fixture(scope="module")
def model_and_batch():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_parity_fixture import PAD, build_request
    from groot1.model import GrootDecisionModel, collate
    from groot1.serialize import EncodedQuestion, EncodedState

    model = GrootDecisionModel.from_pretrained(CKPT)
    chunks, eqs = build_request(np.random.default_rng(0), 1024)
    state = EncodedState(chunks=chunks, tokens=0)
    batch = collate([state], eqs, PAD, q_state=[0] * len(eqs))
    return model, batch, [q.n_options for q in eqs]


def test_forward_shapes_and_probabilities(model_and_batch):
    model, batch, ks = model_and_batch
    with torch.inference_mode():
        out = model(batch)
    assert out["probabilities"].shape[0] == len(ks)
    assert out["confidence_logit"].shape == (len(ks),)
    for qi, k in enumerate(ks):
        p = out["probabilities"][qi, :k]
        assert p.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert torch.isfinite(out["confidence_logit"][qi])


def test_outputs_match_reference_npy(model_and_batch):
    model, batch, ks = model_and_batch
    ref = np.load(FIXTURE / "ref.npy")
    nopt, nq = sum(ks), len(ks)
    with torch.inference_mode():
        out = model(batch)
    probs = np.concatenate([out["probabilities"][q, :k].numpy() for q, k in enumerate(ks)])
    conf = out["confidence_logit"].numpy()
    logits = np.concatenate([out["logits"][q, :k].numpy() for q, k in enumerate(ks)])
    got = np.concatenate([probs, conf, logits])
    assert np.abs(got - ref).max() < 1e-5
