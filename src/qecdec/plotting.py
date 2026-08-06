"""Threshold-curve figures from collect.py artifacts.

Usage:
    python -m qecdec.plotting --results experiments/phase0_surface.results.json \
        --out docs/figures/phase0_surface_threshold.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ordinal single-hue ramp (reference dataviz palette, sequential blue,
# light-surface ordinal bounds: no step lighter than 250).
RAMPS = {
    3: ["#86b6ef", "#2a78d6", "#104281"],
    5: ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def plot_threshold(artifact: dict, out: Path, *, direct_labels: bool = True) -> None:
    by_d: dict[int, list[dict]] = defaultdict(list)
    for row in artifact["results"]:
        by_d[row["distance"]].append(row)
    distances = sorted(by_d)
    ramp = RAMPS.get(len(distances)) or RAMPS[5][: len(distances)]

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for color, d in zip(ramp, distances):
        rows = sorted(by_d[d], key=lambda r: r["p"])
        ps = np.array([r["p"] for r in rows])
        pl = np.array([r["logical_error_rate"] for r in rows])
        shots = np.array([r["shots"] for r in rows])
        errors = np.array([r["errors"] for r in rows])
        err = np.sqrt(pl * (1 - pl) / shots)
        # Gaussian error bars are meaningless below ~10 observed errors; show
        # those points as open markers with no bar instead of a bogus interval.
        solid = errors >= 10
        ax.errorbar(
            ps[solid], pl[solid], yerr=err[solid], color=color, linewidth=2,
            marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1,
            capsize=2, label=f"d = {d}",
        )
        if (~solid).any():
            ax.plot(
                ps[~solid], pl[~solid], linestyle="none", marker="o", markersize=6,
                markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=1.5,
            )
            ax.plot(ps, pl, color=color, linewidth=2, zorder=1)
        if direct_labels and len(distances) <= 4:
            ax.annotate(
                f"d = {d}", (ps[-1], pl[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    cfg = artifact["config"]
    ax.set_xlabel("physical error rate p", color=INK)
    ax.set_ylabel("logical error rate per shot", color=INK)
    ax.set_title(
        f"{cfg['code']} code memory, {cfg['noise']}-level noise, MWPM (PyMatching)",
        color=INK, fontsize=11,
    )
    ax.grid(True, which="both", color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    if direct_labels and len(distances) <= 4:
        ax.margins(x=0.12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot_threshold(json.loads(args.results.read_text()), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
