"""Circuit generation for standard QEC memory experiments via Stim."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import stim

CODE_TASKS = {
    "repetition": "repetition_code:memory",
    "surface": "surface_code:rotated_memory_z",
}

NOISE_MODELS = ("circuit", "phenomenological")


@dataclass(frozen=True)
class CircuitSpec:
    """Fully specifies a memory-experiment circuit.

    noise="circuit" is SD6-style circuit-level depolarizing noise: every
    Clifford is followed by depolarization at rate p, resets/measurements flip
    at rate p, and data qubits depolarize each round. noise="phenomenological"
    applies only data depolarization and measurement flips.
    """

    code: str
    distance: int
    rounds: int
    p: float
    noise: str = "circuit"

    def __post_init__(self) -> None:
        if self.code not in CODE_TASKS:
            raise ValueError(f"unknown code {self.code!r}; options: {sorted(CODE_TASKS)}")
        if self.noise not in NOISE_MODELS:
            raise ValueError(f"unknown noise model {self.noise!r}; options: {NOISE_MODELS}")
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("distance must be an odd integer >= 3")
        if not (0 <= self.p < 1):
            raise ValueError("p must be in [0, 1)")

    def to_metadata(self) -> dict:
        return asdict(self)


def make_circuit(spec: CircuitSpec) -> stim.Circuit:
    noise_kwargs: dict[str, float] = {
        "before_round_data_depolarization": spec.p,
        "before_measure_flip_probability": spec.p,
    }
    if spec.noise == "circuit":
        noise_kwargs["after_clifford_depolarization"] = spec.p
        noise_kwargs["after_reset_flip_probability"] = spec.p
    return stim.Circuit.generated(
        CODE_TASKS[spec.code],
        distance=spec.distance,
        rounds=spec.rounds,
        **noise_kwargs,
    )
