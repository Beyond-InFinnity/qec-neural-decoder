"""Map Stim's flat detector vector onto a (C, T, H, W) space-time volume.

Stim's generated rotated surface code places detectors at even coordinates on
a (2d+1)^2 grid with t = 0..rounds. Halving gives a (d+1)x(d+1) spatial grid.
Channel 0 holds Z-type checks (the ones present at t=0 in a memory_z
experiment), channel 1 the X-type checks. Cells with no detector stay zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim


@dataclass(frozen=True)
class VolumeSpec:
    shape: tuple[int, int, int, int]  # (C, T, H, W)
    flat_index: np.ndarray  # (num_detectors,) linear index into the volume


def volume_spec(circuit: stim.Circuit) -> VolumeSpec:
    coords = circuit.get_detector_coordinates()
    n = circuit.num_detectors
    if set(coords) != set(range(n)):
        raise ValueError("expected coordinates for every detector")
    xs = np.array([coords[i][0] for i in range(n)])
    ys = np.array([coords[i][1] for i in range(n)])
    ts = np.array([coords[i][2] for i in range(n)])
    if (xs % 2).any() or (ys % 2).any():
        raise ValueError("expected even spatial coordinates (rotated surface code)")
    gx = (xs // 2).astype(np.int64)
    gy = (ys // 2).astype(np.int64)
    gt = ts.astype(np.int64)

    z_cells = {(int(x), int(y)) for x, y, t in zip(gx, gy, gt) if t == 0}
    chan = np.array([0 if (int(x), int(y)) in z_cells else 1 for x, y in zip(gx, gy)])

    C, T, H, W = 2, int(gt.max()) + 1, int(gy.max()) + 1, int(gx.max()) + 1
    flat = ((chan * T + gt) * H + gy) * W + gx
    if len(np.unique(flat)) != n:
        raise ValueError("detector -> cell mapping is not injective")
    return VolumeSpec(shape=(C, T, H, W), flat_index=flat)
