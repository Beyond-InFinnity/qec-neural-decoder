"""Real-time-path benchmarks: int8-weight accuracy + CUDA-graph single-shot latency.

Usage:
    python -m qecdec.latency_rt --config experiments/ladder_c16d2.json \
        --device cuda:0 --out experiments/ladder_c16d2.rt.json

Two questions per checkpoint:
1. Does int8 weight quantization (symmetric, per-output-channel) preserve p_L?
   (Weight-only quantization; activations stay fp16 — the standard first rung
   of a deployment pipeline.)
2. How much of the ~400 us single-shot latency is kernel-launch overhead?
   CUDA-graph capture replays the whole forward as one pre-recorded unit.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .circuits import CircuitSpec, make_circuit
from .sampling import sample_detection_events
from .train import build_model


def quantize_weights_int8(model: nn.Module) -> nn.Module:
    """Symmetric per-output-channel int8 fake-quant of conv/linear weights."""
    model = copy.deepcopy(model)
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            w = m.weight.data
            dims = tuple(range(1, w.dim()))
            scale = w.abs().amax(dim=dims, keepdim=True).clamp(min=1e-8) / 127.0
            m.weight.data = (w / scale).round().clamp(-127, 127) * scale
    return model


@torch.no_grad()
def error_rate(model, events, flips, device, dtype) -> float:
    model = model.to(device=device, dtype=dtype).eval()
    preds = []
    for start in range(0, len(events), 8192):
        x = torch.from_numpy(events[start : start + 8192]).to(device=device, dtype=dtype)
        preds.append((model(x) > 0).to(torch.uint8).cpu().numpy())
    return float((np.concatenate(preds) != flips).mean())


@torch.no_grad()
def cuda_graph_single_shot(model, events, device, *, n=2000) -> dict:
    model = model.to(device=device, dtype=torch.float16).eval()
    num_det = events.shape[1]
    static_in = torch.zeros(1, num_det, device=device, dtype=torch.float16)

    s = torch.cuda.Stream(device)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        for _ in range(5):
            model(static_in)
    torch.cuda.current_stream(device).wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = model(static_in)

    xs = torch.from_numpy(events[:n]).to(device=device, dtype=torch.float16)
    for i in range(10):
        static_in.copy_(xs[i : i + 1])
        graph.replay()
    torch.cuda.synchronize(device)
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        static_in.copy_(xs[i : i + 1])
        graph.replay()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    times.sort()
    return {
        "graph_single_us_median": 1e6 * statistics.median(times),
        "graph_single_us_p99": 1e6 * times[int(0.99 * (n - 1))],
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
        model = build_model(config, spec, make_circuit(spec).num_detectors)
        ckpt = ckpt_dir / f"{config.get('name', 'run')}_d{d}_p{p}.pt"
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

        row = {**spec.to_metadata(), "name": config.get("name"),
               "params": sum(t.numel() for t in model.parameters()),
               "p_L_fp16": error_rate(model, test.events, test.flips, device, torch.float16),
               "p_L_int8w_fp16": error_rate(quantize_weights_int8(model), test.events,
                                            test.flips, device, torch.float16),
               **cuda_graph_single_shot(model, test.events, device)}
        print(json.dumps(row), flush=True)
        rows.append(row)

    args.out.write_text(json.dumps({"device": str(device), "results": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
