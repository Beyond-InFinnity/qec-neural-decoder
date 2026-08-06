import numpy as np
import pymatching
import pytest

from qecdec.circuits import CircuitSpec, make_circuit


@pytest.mark.parametrize("code", ["repetition", "surface"])
@pytest.mark.parametrize("noise", ["circuit", "phenomenological"])
def test_noiseless_circuit_has_no_events(code, noise):
    spec = CircuitSpec(code=code, distance=5, rounds=5, p=0.0, noise=noise)
    circuit = make_circuit(spec)
    events, observables = circuit.compile_detector_sampler(seed=0).sample(
        1000, separate_observables=True
    )
    assert not events.any()
    assert not observables.any()


@pytest.mark.parametrize("code", ["repetition", "surface"])
def test_single_logical_observable(code):
    circuit = make_circuit(CircuitSpec(code=code, distance=3, rounds=3, p=0.001))
    assert circuit.num_observables == 1
    assert circuit.num_detectors > 0


def test_mwpm_trivial_syndrome_predicts_no_flip():
    circuit = make_circuit(CircuitSpec(code="surface", distance=3, rounds=3, p=0.001))
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    events = np.zeros((10, dem.num_detectors), dtype=np.uint8)
    assert not matcher.decode_batch(events).any()


def test_mwpm_beats_coin_flip_at_moderate_noise():
    spec = CircuitSpec(code="surface", distance=3, rounds=3, p=0.005)
    circuit = make_circuit(spec)
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    events, observables = circuit.compile_detector_sampler(seed=42).sample(
        5000, separate_observables=True
    )
    predictions = matcher.decode_batch(events)
    error_rate = np.mean(predictions[:, 0] != observables[:, 0])
    assert error_rate < 0.05  # d=3 at p=0.005 should be well under threshold


def test_spec_validation():
    with pytest.raises(ValueError):
        CircuitSpec(code="steane", distance=3, rounds=3, p=0.01)
    with pytest.raises(ValueError):
        CircuitSpec(code="surface", distance=4, rounds=4, p=0.01)
    with pytest.raises(ValueError):
        CircuitSpec(code="surface", distance=3, rounds=3, p=1.5)
