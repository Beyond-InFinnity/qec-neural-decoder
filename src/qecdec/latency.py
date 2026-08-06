"""PyMatching decode-latency baseline: batched throughput and single-shot latency.

Usage:
    python -m qecdec.latency --config experiments/phase0_latency.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pymatching

from .circuits import CircuitSpec, make_circuit


def bench_spec(spec: CircuitSpec, *, shots: int, single_shot_n: int, seed: int) -> dict:
    circuit = make_circuit(spec)
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=seed)
    events, _ = sampler.sample(shots, separate_observables=True)

    # Batched throughput (amortized, the offline-analysis number).
    matcher.decode_batch(events[:100])  # warm up
    t0 = time.perf_counter()
    matcher.decode_batch(events)
    batch_s = time.perf_counter() - t0

    # Single-shot latency (the real-time-feasibility number).
    times = []
    for i in range(min(single_shot_n, shots)):
        t0 = time.perf_counter()
        matcher.decode(events[i])
        times.append(time.perf_counter() - t0)
    times.sort()

    return {
        **spec.to_metadata(),
        "num_detectors": circuit.num_detectors,
        "shots": shots,
        "batch_us_per_shot": 1e6 * batch_s / shots,
        "single_us_median": 1e6 * statistics.median(times),
        "single_us_p99": 1e6 * times[int(0.99 * (len(times) - 1))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())

    rng = np.random.default_rng(config.get("seed", 0))
    rows = []
    for d in config["distances"]:
        spec = CircuitSpec(
            code=config["code"],
            distance=d,
            rounds=d if config.get("rounds", "d") == "d" else int(config["rounds"]),
            p=config["p"],
            noise=config.get("noise", "circuit"),
        )
        rows.append(
            bench_spec(
                spec,
                shots=config.get("shots", 10_000),
                single_shot_n=config.get("single_shot_n", 2_000),
                seed=int(rng.integers(2**31)),
            )
        )
        print(rows[-1])

    out = args.out or args.config.with_name(args.config.stem + ".results.json")
    out.write_text(json.dumps({"config": config, "results": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
