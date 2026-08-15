"""Decoder latency benchmark: NN checkpoint vs PyMatching vs belief-matching.

Usage:
    python -m qecdec.latency_nn --config experiments/phase2_surface_cnn3d.json \
        --device cuda:0 --out experiments/phase3_latency.json

For each cell: batched throughput (us/shot at several batch sizes), single-shot
latency (median/p99 over N decodes, cuda-synchronized), for the NN in fp32 and
fp16, plus the classical decoders on the same circuits. Accuracy of every timed
variant is verified on the same shots so speed numbers can't silently trade
correctness.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pymatching
import torch
from beliefmatching import BeliefMatching

from .circuits import CircuitSpec, make_circuit
from .collect import git_sha
from .sampling import sample_detection_events
from .train import build_model


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def bench_nn(model, events: np.ndarray, flips: np.ndarray, device, *, dtype,
             batch_sizes=(256, 2048, 8192), single_n=2000) -> dict:
    model = model.to(device=device, dtype=dtype).eval()
    x_all = torch.from_numpy(events).to(device)
    out: dict = {"dtype": str(dtype).split(".")[-1]}

    correct = []
    for start in range(0, len(events), 8192):
        xb = x_all[start : start + 8192].to(dtype)
        correct.append(((model(xb) > 0).to(torch.uint8)).cpu().numpy())
    out["logical_error_rate"] = float((np.concatenate(correct) != flips).mean())

    for bs in batch_sizes:
        xb = x_all[:bs].to(dtype)
        for _ in range(3):
            model(xb)
        _sync(device)
        t0 = time.perf_counter()
        iters = max(1, 50_000 // bs)
        for _ in range(iters):
            model(xb)
        _sync(device)
        out[f"batch{bs}_us_per_shot"] = 1e6 * (time.perf_counter() - t0) / (iters * bs)

    x1 = x_all[:1].to(dtype)
    for _ in range(10):
        model(x1)
    _sync(device)
    times = []
    for i in range(single_n):
        xi = x_all[i : i + 1].to(dtype)
        t0 = time.perf_counter()
        model(xi)
        _sync(device)
        times.append(time.perf_counter() - t0)
    times.sort()
    out["single_us_median"] = 1e6 * statistics.median(times)
    out["single_us_p99"] = 1e6 * times[int(0.99 * (len(times) - 1))]
    return out


def bench_classical(decoder, events: np.ndarray, flips: np.ndarray, *, single_n=2000) -> dict:
    decoder.decode_batch(events[:100])
    t0 = time.perf_counter()
    pred = decoder.decode_batch(events)
    batch_s = time.perf_counter() - t0
    times = []
    for i in range(single_n):
        t0 = time.perf_counter()
        decoder.decode(events[i])
        times.append(time.perf_counter() - t0)
    times.sort()
    return {
        "logical_error_rate": float((pred[:, 0].astype(np.uint8) != flips).mean()),
        "batch_us_per_shot": 1e6 * batch_s / len(events),
        "single_us_median": 1e6 * statistics.median(times),
        "single_us_p99": 1e6 * times[int(0.99 * (len(times) - 1))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    device = torch.device(args.device)
    ckpt_dir = Path(config.get("ckpt_dir", "experiments/models"))
    seed = int(config.get("seed", 0))

    cells = config.get("cells") or [[d, p] for d in config["distances"] for p in config["ps"]]
    rows = []
    for d, p in cells:
        rounds = d if config.get("rounds", "d") == "d" else int(config["rounds"])
        spec = CircuitSpec(code=config["code"], distance=d, rounds=rounds, p=p,
                           noise=config.get("noise", "circuit"))
        test = sample_detection_events(spec, args.shots, seed=seed + 90)
        dem = make_circuit(spec).detector_error_model(decompose_errors=True)
        model = build_model(config, spec, test.events.shape[1])
        ckpt = ckpt_dir / f"{config.get('name', 'run')}_d{d}_p{p}.pt"
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        params = sum(t.numel() for t in model.parameters())

        row = {**spec.to_metadata(), "params": params, "shots": args.shots, "decoders": {}}
        row["decoders"]["nn_fp32"] = bench_nn(model, test.events, test.flips, device, dtype=torch.float32)
        row["decoders"]["nn_fp16"] = bench_nn(model, test.events, test.flips, device, dtype=torch.float16)
        row["decoders"]["mwpm"] = bench_classical(
            pymatching.Matching.from_detector_error_model(dem), test.events, test.flips)
        bm = BeliefMatching(dem, max_bp_iters=20)
        bm_events = test.events[:20_000]
        t0 = time.perf_counter()
        bm_pred = bm.decode_batch(bm_events)
        row["decoders"]["beliefmatching"] = {
            "logical_error_rate": float((bm_pred[:, 0].astype(np.uint8) != test.flips[:20_000]).mean()),
            "batch_us_per_shot": 1e6 * (time.perf_counter() - t0) / len(bm_events),
        }
        print(json.dumps(row, indent=None), flush=True)
        rows.append(row)

    args.out.write_text(json.dumps(
        {"config_name": config.get("name"), "device": str(device),
         "git_sha": git_sha(args.config.resolve().parent.parent), "results": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
