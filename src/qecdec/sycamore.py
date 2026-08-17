"""Ingestion for Google's Willow surface-code memory dataset (Zenodo 13273331).

Layout per experiment directory (<root>/<placement>/<basis>/r<rounds>/):
    circuit_ideal.stim, circuit_noisy_si1000.stim
    detection_events.b8          50k shots, packed detector bits
    obs_flips_actual.b8          real logical outcome per shot
    metadata.json
    decoding_results/<decoder>/  obs_flips_predicted.b8 + error_model.dem
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim


@dataclass(frozen=True)
class SycamoreExperiment:
    path: Path
    placement: str
    basis: str
    rounds: int

    @property
    def name(self) -> str:
        return f"{self.placement}/{self.basis}/r{self.rounds}"


def discover(root: Path) -> list[SycamoreExperiment]:
    out = []
    for r_dir in sorted(root.glob("*/*/r*")):
        if not (r_dir / "detection_events.b8").exists():
            continue
        out.append(
            SycamoreExperiment(
                path=r_dir,
                placement=r_dir.parts[-3],
                basis=r_dir.parts[-2],
                rounds=int(r_dir.name[1:]),
            )
        )
    return out


def load_circuit(exp: SycamoreExperiment, noisy: bool = True) -> stim.Circuit:
    name = "circuit_noisy_si1000.stim" if noisy else "circuit_ideal.stim"
    return stim.Circuit.from_file(exp.path / name)


def load_shots(exp: SycamoreExperiment) -> tuple[np.ndarray, np.ndarray]:
    """Returns (detection_events uint8 [shots, detectors], actual flips [shots])."""
    circuit = load_circuit(exp, noisy=False)
    events = stim.read_shot_data_file(
        path=str(exp.path / "detection_events.b8"),
        format="b8",
        num_detectors=circuit.num_detectors,
    ).astype(np.uint8)
    flips = stim.read_shot_data_file(
        path=str(exp.path / "obs_flips_actual.b8"),
        format="b8",
        num_observables=1,
        num_detectors=0,
    ).astype(np.uint8)[:, 0]
    if events.shape[0] != flips.shape[0]:
        raise ValueError(f"{exp.name}: {events.shape[0]} event shots vs {flips.shape[0]} flips")
    return events, flips


def google_decoders(exp: SycamoreExperiment) -> list[str]:
    d = exp.path / "decoding_results"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def load_google_predictions(exp: SycamoreExperiment, decoder: str) -> np.ndarray:
    return stim.read_shot_data_file(
        path=str(exp.path / "decoding_results" / decoder / "obs_flips_predicted.b8"),
        format="b8",
        num_observables=1,
        num_detectors=0,
    ).astype(np.uint8)[:, 0]


def load_fitted_dem(exp: SycamoreExperiment, decoder: str) -> stim.DetectorErrorModel:
    return stim.DetectorErrorModel.from_file(
        exp.path / "decoding_results" / decoder / "error_model.dem"
    )
