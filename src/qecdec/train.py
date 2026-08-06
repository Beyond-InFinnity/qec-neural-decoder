"""Config-driven NN-decoder training with paired MWPM evaluation.

Usage:
    python -m qecdec.train --config experiments/phase1_rep_mlp.json [--device cuda:0]

For every (distance, p) pair in the config: sample a train/test split, train
the model, decode the SAME test shots with PyMatching, and record paired
results. Artifact rows carry both raw counts and the paired disagreements
(nn_only_wrong / mwpm_only_wrong) so significance can be assessed downstream.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pymatching
import torch
from torch import nn

from .circuits import CircuitSpec, make_circuit
from .collect import git_sha
from .models import ConvDecoder, MLPDecoder
from .sampling import sample_detection_events


def build_model(config: dict, spec: CircuitSpec, num_detectors: int) -> nn.Module:
    mcfg = config["model"]
    arch = mcfg.get("arch", "mlp")
    if arch == "mlp":
        return MLPDecoder(
            num_detectors=num_detectors,
            hidden=tuple(mcfg.get("hidden", [256, 256])),
            dropout=mcfg.get("dropout", 0.0),
        )
    if arch == "cnn":
        if spec.code != "repetition":
            raise NotImplementedError("CNN grid mapping is repetition-code-only for now")
        grid = (spec.rounds + 1, spec.distance - 1)
        if grid[0] * grid[1] != num_detectors:
            raise ValueError(f"grid {grid} != {num_detectors} detectors")
        return ConvDecoder(
            grid=grid,
            channels=mcfg.get("channels", 64),
            depth=mcfg.get("depth", 4),
            head=mcfg.get("head", 128),
        )
    raise ValueError(f"unknown arch {arch!r}")


def mwpm_predict(spec: CircuitSpec, events: np.ndarray) -> np.ndarray:
    dem = make_circuit(spec).detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    return matcher.decode_batch(events)[:, 0].astype(np.uint8)


def train_model(
    model: nn.Module,
    events: np.ndarray,
    flips: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    cosine: bool = False,
    log_prefix: str = "",
) -> None:
    # Keep the dataset as uint8 on-device (8x smaller than float32 — a 10M-shot
    # d=11 set is ~1.2 GB) and cast per batch.
    x = torch.from_numpy(events).to(device=device)
    y = torch.from_numpy(flips).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        if cosine
        else None
    )
    loss_fn = nn.BCEWithLogitsLoss()
    n = x.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x[idx].float()), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * idx.shape[0]
        if sched is not None:
            sched.step()
        print(f"{log_prefix}epoch {epoch + 1}/{epochs} loss {total / n:.5f}", flush=True)


@torch.no_grad()
def nn_predict(model: nn.Module, events: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, events.shape[0], 65536):
        x = torch.from_numpy(events[start : start + 65536]).to(
            device=device, dtype=torch.float32
        )
        preds.append((model(x) > 0).to(torch.uint8).cpu().numpy())
    return np.concatenate(preds)


def run_one(config: dict, d: int, p: float, device: torch.device) -> dict:
    spec = CircuitSpec(
        code=config["code"],
        distance=d,
        rounds=d if config.get("rounds", "d") == "d" else int(config["rounds"]),
        p=p,
        noise=config.get("noise", "circuit"),
    )
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    label = f"[d={d} p={p}] "
    t0 = time.perf_counter()
    train = sample_detection_events(spec, config["train_shots"], seed=seed + 1)
    test = sample_detection_events(spec, config["test_shots"], seed=seed + 2)

    model = build_model(config, spec, train.events.shape[1]).to(device)
    train_model(
        model,
        train.events,
        train.flips,
        epochs=config.get("epochs", 10),
        batch_size=config.get("batch_size", 1024),
        lr=config.get("lr", 1e-3),
        device=device,
        cosine=config.get("cosine", False),
        log_prefix=label,
    )

    nn_pred = nn_predict(model, test.events, device)
    mwpm_pred = mwpm_predict(spec, test.events)
    nn_wrong = nn_pred != test.flips
    mwpm_wrong = mwpm_pred != test.flips
    row = {
        **spec.to_metadata(),
        "train_shots": config["train_shots"],
        "test_shots": config["test_shots"],
        "nn_errors": int(nn_wrong.sum()),
        "mwpm_errors": int(mwpm_wrong.sum()),
        "nn_only_wrong": int((nn_wrong & ~mwpm_wrong).sum()),
        "mwpm_only_wrong": int((~nn_wrong & mwpm_wrong).sum()),
        "nn_logical_error_rate": float(nn_wrong.mean()),
        "mwpm_logical_error_rate": float(mwpm_wrong.mean()),
        "params": sum(t.numel() for t in model.parameters()),
        "seconds": time.perf_counter() - t0,
    }
    print(f"{label}p_L nn={row['nn_logical_error_rate']:.2e} "
          f"mwpm={row['mwpm_logical_error_rate']:.2e}", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    device = torch.device(args.device)

    rows = [
        run_one(config, d, p, device)
        for d in config["distances"]
        for p in config["ps"]
    ]
    out = args.out or args.config.with_name(args.config.stem + ".results.json")
    out.write_text(
        json.dumps(
            {
                "config": config,
                "git_sha": git_sha(args.config.resolve().parent.parent),
                "device": str(device),
                "results": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
