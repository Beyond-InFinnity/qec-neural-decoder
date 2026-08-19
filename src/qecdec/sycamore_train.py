"""Pretrain on Google's fitted DEM, fine-tune on real shots, paired evaluation.

Usage:
    python -m qecdec.sycamore_train --config experiments/sycamore_d3_r13.json --device cuda:1

Protocol (docs/phase3b-sycamore-plan.md):
1. Sample synthetic detection events from the experiment's fitted DEM;
   pretrain the volume CNN on them.
2. Split the 50k real shots (seeded permutation): fine-tune on train split,
   never touching the held-out test split.
3. Evaluate pretrained-only and fine-tuned models on held-out real shots,
   alongside plain MWPM and every recorded Google decoder, all on the same
   shots (Google predictions are subset to the same held-out indices).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pymatching
import torch

from .collect import git_sha
from .models import SurfaceConvDecoder
from .representation import volume_spec
from .sycamore import (SycamoreExperiment, discover, google_decoders,
                       load_circuit, load_fitted_dem, load_google_predictions,
                       load_shots)
from .train import nn_predict, train_model


def find_experiment(root: Path, name: str) -> SycamoreExperiment:
    for exp in discover(root):
        if exp.name == name:
            return exp
    raise ValueError(f"experiment {name!r} not found under {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    device = torch.device(args.device)
    seed = int(config["seed"])
    torch.manual_seed(seed)

    exp = find_experiment(Path(config["root"]), config["experiment"])
    circuit = load_circuit(exp, noisy=False)
    vspec = volume_spec(circuit)
    dem = load_fitted_dem(exp, config["dem_source"])
    print(f"{exp.name}: {circuit.num_detectors} detectors -> volume {vspec.shape}", flush=True)

    sampler = dem.compile_sampler(seed=seed + 1)
    dets, obs, _ = sampler.sample(config["pretrain_shots"])
    dets = dets.astype(np.uint8)
    obs = obs[:, 0].astype(np.uint8)
    print(f"pretrain set: {dets.shape}, flip rate {obs.mean():.4f}", flush=True)

    mcfg = config["model"]
    model = SurfaceConvDecoder(
        volume_shape=vspec.shape, flat_index=vspec.flat_index,
        channels=mcfg.get("channels", 32), depth=mcfg.get("depth", 3),
        head=mcfg.get("head", 128), groupnorm=mcfg.get("groupnorm", True),
    ).to(device)
    def resample(epoch: int):
        s = dem.compile_sampler(seed=seed + 100 + epoch)
        d2, o2, _ = s.sample(config["pretrain_shots"])
        return d2.astype(np.uint8), o2[:, 0].astype(np.uint8)

    train_model(
        model, dets, obs,
        epochs=config["pretrain_epochs"], batch_size=config.get("batch_size", 4096),
        lr=config.get("lr", 1e-3), device=device, cosine=True,
        grad_clip=1.0, warmup_steps=config.get("warmup_steps", 500),
        data_refresh=resample if config.get("resample_per_epoch") else None,
        log_prefix="[pretrain] ",
    )
    ckpt_dir = Path("experiments/models")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"{config['name']}_pretrained.pt")

    events, flips = load_shots(exp)
    rng = np.random.default_rng(seed + 2)
    perm = rng.permutation(events.shape[0])
    n_test = int(config.get("test_frac", 0.2) * events.shape[0])
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    results: dict = {"experiment": exp.name, "volume_shape": list(vspec.shape),
                     "detectors": int(circuit.num_detectors),
                     "test_shots": int(n_test),
                     "params": sum(t.numel() for t in model.parameters())}

    def rate(pred: np.ndarray) -> float:
        return float((pred != flips[test_idx]).mean())

    results["nn_pretrained"] = rate(nn_predict(model, events[test_idx], device))

    train_model(
        model, events[train_idx], flips[train_idx],
        epochs=config["finetune_epochs"], batch_size=config.get("finetune_batch", 512),
        lr=config.get("finetune_lr", 1e-4), device=device, cosine=True,
        grad_clip=1.0, warmup_steps=100, log_prefix="[finetune] ",
    )
    torch.save(model.state_dict(), ckpt_dir / f"{config['name']}_finetuned.pt")
    nn_ft = nn_predict(model, events[test_idx], device)
    results["nn_finetuned"] = rate(nn_ft)

    mwpm = pymatching.Matching.from_detector_error_model(dem)
    results["mwpm_on_fitted_dem"] = rate(
        mwpm.decode_batch(events[test_idx])[:, 0].astype(np.uint8))
    for dec in google_decoders(exp):
        gp = load_google_predictions(exp, dec)[test_idx]
        results[f"google:{dec}"] = rate(gp)
        wrong_nn = nn_ft != flips[test_idx]
        wrong_g = gp != flips[test_idx]
        results[f"paired_vs_{dec}"] = {
            "nn_only_wrong": int((wrong_nn & ~wrong_g).sum()),
            "google_only_wrong": int((~wrong_nn & wrong_g).sum()),
        }

    out = args.out or args.config.with_name(args.config.stem + ".results.json")
    out.write_text(json.dumps(
        {"config": config, "git_sha": git_sha(Path.cwd()), "results": results}, indent=2))
    print(json.dumps({k: v for k, v in results.items() if not k.startswith("paired")},
                     indent=2), flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
