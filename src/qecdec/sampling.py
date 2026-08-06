"""Stim sampling → numpy arrays for training and paired evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuits import CircuitSpec, make_circuit


@dataclass
class SampleSet:
    """Detection events (shots × detectors) and logical observable flips (shots,)."""

    events: np.ndarray
    flips: np.ndarray

    def __post_init__(self) -> None:
        assert self.events.ndim == 2 and self.flips.ndim == 1
        assert self.events.shape[0] == self.flips.shape[0]


def sample_detection_events(spec: CircuitSpec, shots: int, seed: int) -> SampleSet:
    circuit = make_circuit(spec)
    events, observables = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )
    return SampleSet(
        events=events.astype(np.uint8),
        flips=observables[:, 0].astype(np.uint8),
    )
