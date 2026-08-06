import numpy as np
import torch

from qecdec.circuits import CircuitSpec
from qecdec.models import MLPDecoder
from qecdec.sampling import sample_detection_events
from qecdec.train import mwpm_predict, nn_predict, train_model


def test_sampling_deterministic_and_shaped():
    spec = CircuitSpec(code="repetition", distance=5, rounds=5, p=0.02)
    a = sample_detection_events(spec, 500, seed=3)
    b = sample_detection_events(spec, 500, seed=3)
    assert a.events.shape[0] == 500 and a.events.shape == b.events.shape
    assert np.array_equal(a.events, b.events)
    assert set(np.unique(a.events)) <= {0, 1}


def test_mlp_learns_repetition_code_cpu():
    spec = CircuitSpec(code="repetition", distance=3, rounds=3, p=0.03)
    torch.manual_seed(0)
    train = sample_detection_events(spec, 60_000, seed=1)
    test = sample_detection_events(spec, 20_000, seed=2)
    device = torch.device("cpu")
    model = MLPDecoder(train.events.shape[1], hidden=(128, 64)).to(device)
    train_model(
        model, train.events, train.flips,
        epochs=8, batch_size=1024, lr=3e-3, device=device,
    )
    nn_rate = float((nn_predict(model, test.events, device) != test.flips).mean())
    mwpm_rate = float((mwpm_predict(spec, test.events) != test.flips).mean())
    # Depth matters: a (128, 64) MLP lands at/below MWPM on the d=3 rep code
    # (a single 64-unit layer plateaus ~2.3x worse — see Phase 1 notes).
    assert nn_rate < 1.15 * mwpm_rate


def test_cnn_grid_mapping():
    from qecdec.circuits import make_circuit
    from qecdec.train import build_model

    for d in (5, 7, 11):
        spec = CircuitSpec(code="repetition", distance=d, rounds=d, p=0.02)
        n = make_circuit(spec).num_detectors
        assert n == (d + 1) * (d - 1)
        model = build_model({"model": {"arch": "cnn", "channels": 8, "depth": 1}}, spec, n)
        out = model(torch.zeros(4, n))
        assert out.shape == (4,)
