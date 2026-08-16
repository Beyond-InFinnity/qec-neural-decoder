"""Phase 1 convergence figure: p_L(NN)/p_L(MWPM) at p=0.02 across the
intervention sequence, repetition code, d = 7/9/11.

Usage: .venv/bin/python scripts/plot_phase1_arc.py
"""

import matplotlib.pyplot as plt

from qecdec.plotting import AXIS, GRID, INK, MUTED, SURFACE

# Ratios p_L(NN)/p_L(MWPM) at p=0.02 from committed artifacts:
# phase1_rep_mlp, phase1_rep_mlp_scaled, phase1_rep_cnn,
# phase1_rep_cnn_dilated(_fix), phase1_rep_cnn_transfer.
STAGES = ["MLP", "MLP,\n19$\\times$ compute", "CNN", "+ dilation\n(RF 29)",
          "+ train at\np = 0.05"]
SERIES = {  # d: (ratio per stage; None = not run at that stage)
    7: [3.20, 1.69, 1.27, 1.14, None],
    9: [13.56, 3.15, 2.35, 1.44, 1.04],
    11: [54.97, 17.83, 6.71, 1.88, 1.08],
}
RAMP = {7: "#86b6ef", 9: "#2a78d6", 11: "#104281"}

fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

x = range(len(STAGES))
for d, ys in SERIES.items():
    pts = [(xi, y) for xi, y in zip(x, ys) if y is not None]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=RAMP[d], linewidth=2,
            marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1,
            label=f"d = {d}")
    dy = {7: 0, 9: -9, 11: 7}[d]
    ax.annotate(f"d = {d}", pts[-1], xytext=(8, dy), textcoords="offset points",
                va="center", fontsize=9, color=INK)

ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
ax.annotate("MWPM parity", (0.02, 1.0), xytext=(0, 4), textcoords="offset points",
            fontsize=8, color=MUTED)

ax.set_yscale("log")
ax.set_xticks(list(x), STAGES, fontsize=8.5)
ax.set_ylabel("$p_L(\\mathrm{NN})\\,/\\,p_L(\\mathrm{MWPM})$  at  $p = 0.02$",
              color=INK)
ax.set_title("Repetition-code decoder: each intervention isolates one variable",
             color=INK, fontsize=11)
ax.grid(True, which="both", axis="y", color=GRID, linewidth=0.6)
ax.tick_params(colors=MUTED, labelcolor=INK)
for spine in ax.spines.values():
    spine.set_color(AXIS)
ax.margins(x=0.14)
fig.tight_layout()
fig.savefig("docs/figures/phase1_convergence.png", facecolor=SURFACE)
print("wrote docs/figures/phase1_convergence.png")
