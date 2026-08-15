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
from .models import ConvDecoder, MLPDecoder, SurfaceConvDecoder
from .representation import volume_spec
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
            dilations=mcfg.get("dilations"),
        )
    if arch == "cnn3d":
        if spec.code != "surface":
            raise NotImplementedError("cnn3d expects the surface code")
        vspec = volume_spec(make_circuit(spec))
        if len(vspec.flat_index) != num_detectors:
            raise ValueError("volume mapping does not cover all detectors")
        return SurfaceConvDecoder(
            volume_shape=vspec.shape,
            flat_index=vspec.flat_index,
            channels=mcfg.get("channels", 48),
            depth=mcfg.get("depth", 4),
            head=mcfg.get("head", 128),
            groupnorm=mcfg.get("groupnorm", True),
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
    grad_clip: float | None = None,
    warmup_steps: int = 0,
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
    step = 0
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch_size):
            # Linear warmup: guards against collapse-to-base-rate from unlucky
            # early batch sequences (observed as a data-seed-dependent freeze;
            # see docs/phase2 stability notes).
            if warmup_steps and step < warmup_steps:
                for g in opt.param_groups:
                    g["lr"] = lr * (step + 1) / warmup_steps
            step += 1
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x[idx].float()), y[idx])
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += loss.item() * idx.shape[0]
        if sched is not None:
            sched.step()
        print(f"{log_prefix}epoch {epoch + 1}/{epochs} loss {total / n:.5f}", flush=True)


@torch.no_grad()
def nn_predict(
    model: nn.Module, events: np.ndarray, device: torch.device, chunk: int = 8192
) -> np.ndarray:
    # Chunk size bounds eval activation memory (Conv3d at 64k shots OOMs 8 GB).
    model.eval()
    preds = []
    for start in range(0, events.shape[0], chunk):
        x = torch.from_numpy(events[start : start + chunk]).to(
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
        grad_clip=config.get("grad_clip"),
        warmup_steps=config.get("warmup_steps", 0),
        log_prefix=label,
    )

    # Persist weights immediately: a crash during eval must not cost the
    # training run. Checkpoints are gitignored (*.pt).
    ckpt_dir = Path(config.get("ckpt_dir", "experiments/models"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"{config.get('name', 'run')}_d{d}_p{p}.pt")

    # "eval_ps" evaluates the p-trained model across other noise rates
    # (train-where-errors-are-plentiful, test-where-they're-rare).
    rows = []
    for eval_p in config.get("eval_ps") or [p]:
        eval_spec = CircuitSpec(
            code=spec.code, distance=d, rounds=spec.rounds, p=eval_p, noise=spec.noise
        )
        etest = (
            test
            if eval_p == p
            else sample_detection_events(eval_spec, config["test_shots"], seed=seed + 3)
        )
        nn_wrong = nn_predict(model, etest.events, device) != etest.flips
        mwpm_wrong = mwpm_predict(eval_spec, etest.events) != etest.flips
        row = {
            **spec.to_metadata(),
            "eval_p": eval_p,
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
        print(f"{label}eval_p={eval_p} p_L nn={row['nn_logical_error_rate']:.2e} "
              f"mwpm={row['mwpm_logical_error_rate']:.2e}", flush=True)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    device = torch.device(args.device)

    # "cells": explicit [d, p] pairs override the distances x ps cross-product
    # (for targeted retrains).
    cells = config.get("cells") or [
        [d, p] for d in config["distances"] for p in config["ps"]
    ]
    rows = [row for d, p in cells for row in run_one(config, d, p, device)]
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
