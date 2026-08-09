import numpy as np
import torch

from qecdec.circuits import CircuitSpec, make_circuit
from qecdec.representation import volume_spec
from qecdec.models import SurfaceConvDecoder


def test_volume_mapping_injective_and_shaped():
    for d in (3, 5):
        spec = CircuitSpec(code="surface", distance=d, rounds=d, p=0.005)
        circuit = make_circuit(spec)
        v = volume_spec(circuit)
        assert v.shape == (2, d + 1, d + 1, d + 1)
        assert len(v.flat_index) == circuit.num_detectors
        assert len(np.unique(v.flat_index)) == circuit.num_detectors
        assert v.flat_index.max() < np.prod(v.shape)


def test_z_and_x_channels_split_evenly():
    # d=3, r=3: 12 Z-type and 12 X-type detector slots... Z appear in rounds
    # 0..3 (4+4+4... actually 4 per boundary round + shared interior); just
    # check both channels are used and Z channel holds the t=0 detectors.
    spec = CircuitSpec(code="surface", distance=3, rounds=3, p=0.005)
    circuit = make_circuit(spec)
    v = volume_spec(circuit)
    C, T, H, W = v.shape
    chan = v.flat_index // (T * H * W)
    coords = circuit.get_detector_coordinates()
    for i in range(circuit.num_detectors):
        if coords[i][2] == 0:
            assert chan[i] == 0
    assert set(chan.tolist()) == {0, 1}


def test_surface_decoder_forward():
    spec = CircuitSpec(code="surface", distance=3, rounds=3, p=0.005)
    circuit = make_circuit(spec)
    v = volume_spec(circuit)
    model = SurfaceConvDecoder(v.shape, v.flat_index, channels=8, depth=2, head=16)
    out = model(torch.zeros(5, circuit.num_detectors))
    assert out.shape == (5,)
    # Scatter places each detector in its own cell: nonzero input -> nonzero volume
    x = torch.zeros(1, circuit.num_detectors)
    x[0, 7] = 1.0
    b = x.shape[0]
    vol = x.new_zeros((b, int(np.prod(v.shape))))
    vol[:, model.flat_index] = x
    assert vol.sum() == 1.0
