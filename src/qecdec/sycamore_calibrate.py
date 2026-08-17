"""Calibration pass: our classical decoders on Google's real shots + fitted DEMs,
side-by-side with Google's own decoders' recorded predictions.

Usage:
    python -m qecdec.sycamore_calibrate --root data/sycamore/google_105Q_surface_code_d3_d5_d7 \
        [--placements d3_at_q4_5 d5_at_q4_5] --out experiments/sycamore_calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pymatching

from .sycamore import (discover, google_decoders, load_fitted_dem,
                       load_google_predictions, load_shots)

DEM_SOURCE = "correlated_matching_decoder_with_si1000_prior"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--placements", nargs="*", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for exp in discover(args.root):
        if args.placements and exp.placement not in args.placements:
            continue
        events, flips = load_shots(exp)
        row = {
            "experiment": exp.name,
            "shots": int(events.shape[0]),
            "detectors": int(events.shape[1]),
            "detection_fraction": float(events.mean()),
        }
        for dec in google_decoders(exp):
            pred = load_google_predictions(exp, dec)
            row[f"google:{dec}"] = float((pred != flips).mean())
        dem = load_fitted_dem(exp, DEM_SOURCE)
        ours = pymatching.Matching.from_detector_error_model(dem)
        row["ours:pymatching_on_fitted_dem"] = float(
            (ours.decode_batch(events)[:, 0].astype(np.uint8) != flips).mean()
        )
        print(json.dumps(row), flush=True)
        rows.append(row)

    args.out.write_text(json.dumps({"dem_source": DEM_SOURCE, "results": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
