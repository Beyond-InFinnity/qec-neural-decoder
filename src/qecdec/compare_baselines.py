"""Three-way paired comparison: NN checkpoint vs plain MWPM vs belief-matching.

Usage:
    python -m qecdec.compare_baselines --config experiments/phase2_surface_cnn3d.json \
        [--device cuda:1] [--out ...]

Regenerates each cell's test sets from the config seeds (identical to the
training run's evaluation shots), loads the saved checkpoint, and decodes the
same shots with all three decoders. Belief-matching (BP on the full DEM +
matching; Higgott et al.) is the strong classical baseline that recovers
correlated-error information plain MWPM discards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pymatching
import torch
from beliefmatching import BeliefMatching

from .circuits import CircuitSpec, make_circuit
from .collect import git_sha
from .sampling import sample_detection_events
from .train import build_model, nn_predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-bp-iters", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    device = torch.device(args.device)
    ckpt_dir = Path(config.get("ckpt_dir", "experiments/models"))
    seed = int(config.get("seed", 0))

    cells = config.get("cells") or [
        [d, p] for d in config["distances"] for p in config["ps"]
    ]
    rows = []
    for d, p in cells:
        rounds = d if config.get("rounds", "d") == "d" else int(config["rounds"])
        spec = CircuitSpec(code=config["code"], distance=d, rounds=rounds, p=p,
                           noise=config.get("noise", "circuit"))
        model = build_model(config, spec, make_circuit(spec).num_detectors).to(device)
        ckpt = ckpt_dir / f"{config.get('name', 'run')}_d{d}_p{p}.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

        for eval_p in config.get("eval_ps") or [p]:
            espec = CircuitSpec(code=spec.code, distance=d, rounds=rounds, p=eval_p,
                                noise=spec.noise)
            test = sample_detection_events(
                espec, config["test_shots"], seed=seed + (2 if eval_p == p else 3)
            )
            dem = make_circuit(espec).detector_error_model(decompose_errors=True)
            preds = {
                "nn": nn_predict(model, test.events, device),
                "mwpm": pymatching.Matching.from_detector_error_model(dem)
                .decode_batch(test.events)[:, 0].astype(np.uint8),
                "beliefmatching": BeliefMatching(dem, max_bp_iters=args.max_bp_iters)
                .decode_batch(test.events)[:, 0].astype(np.uint8),
            }
            wrong = {k: v != test.flips for k, v in preds.items()}
            row = {
                **spec.to_metadata(), "eval_p": eval_p, "test_shots": config["test_shots"],
                **{f"{k}_logical_error_rate": float(w.mean()) for k, w in wrong.items()},
                **{f"{k}_errors": int(w.sum()) for k, w in wrong.items()},
                "nn_only_wrong_vs_bm": int((wrong["nn"] & ~wrong["beliefmatching"]).sum()),
                "bm_only_wrong_vs_nn": int((~wrong["nn"] & wrong["beliefmatching"]).sum()),
            }
            print({k: v for k, v in row.items() if "rate" in k or k in ("distance", "eval_p")},
                  flush=True)
            rows.append(row)

    out = args.out or args.config.with_name(args.config.stem + ".baselines.json")
    out.write_text(json.dumps(
        {"config": config, "git_sha": git_sha(args.config.resolve().parent.parent),
         "max_bp_iters": args.max_bp_iters, "results": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
