"""Phase 3A Pareto frontier: accuracy vs throughput at d=5, p=0.008.

Usage: .venv/bin/python scripts/plot_pareto.py
"""

import glob
import json

import matplotlib.pyplot as plt

from qecdec.plotting import AXIS, GRID, INK, MUTED, SURFACE

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

rungs = []
for f in sorted(glob.glob("experiments/ladder_c*.latency.json")):
    d = json.load(open(f))["results"][0]
    name = f.split("/")[-1].replace(".latency.json", "").replace("ladder_", "")
    rungs.append((d["decoders"]["nn_fp16"]["batch8192_us_per_shot"],
                  d["decoders"]["nn_fp16"]["logical_error_rate"], name, d["params"]))
v1 = json.load(open("experiments/phase3_latency_v1.json"))["results"][1]["decoders"]
rungs.append((v1["nn_fp16"]["batch8192_us_per_shot"],
              v1["nn_fp16"]["logical_error_rate"], "48d4 (v1)", 1516769))
rungs.sort()

fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

xs = [r[0] for r in rungs]
ys = [r[1] for r in rungs]
ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=6,
        markeredgecolor=SURFACE, markeredgewidth=1, label="NN (fp16, batched)")
for x, y, name, params in rungs:
    ax.annotate(f"{name}\n{params/1e3:.0f}k", (x, y), xytext=(6, 6),
                textcoords="offset points", fontsize=7.5, color=INK)

ax.plot([v1["mwpm"]["batch_us_per_shot"]], [v1["mwpm"]["logical_error_rate"]],
        marker="s", markersize=8, color=ORANGE, linestyle="none", label="MWPM (PyMatching)")
ax.annotate("MWPM", (v1["mwpm"]["batch_us_per_shot"], v1["mwpm"]["logical_error_rate"]),
            xytext=(6, -12), textcoords="offset points", fontsize=8, color=INK)
ax.plot([v1["beliefmatching"]["batch_us_per_shot"]],
        [v1["beliefmatching"]["logical_error_rate"]],
        marker="D", markersize=8, color=AQUA, linestyle="none", label="belief-matching")
ax.annotate("belief-matching", (v1["beliefmatching"]["batch_us_per_shot"],
            v1["beliefmatching"]["logical_error_rate"]),
            xytext=(-8, 10), textcoords="offset points", fontsize=8, color=INK, ha="right")

ax.axvline(1.0, color=MUTED, linewidth=1, linestyle="--")
ax.annotate("1 µs syndrome cycle", (1.0, 0.037), xytext=(4, 0),
            textcoords="offset points", fontsize=7.5, color=MUTED, rotation=90, va="bottom")

ax.set_xscale("log")
ax.set_xlabel("decode cost, µs/shot (batched, RTX 3070)", color=INK)
ax.set_ylabel("logical error rate  (d=5, p=0.008)", color=INK)
ax.set_title("Accuracy-throughput frontier: learned vs classical decoders",
             color=INK, fontsize=11)
ax.grid(True, which="both", color=GRID, linewidth=0.6)
ax.tick_params(colors=MUTED, labelcolor=INK)
for spine in ax.spines.values():
    spine.set_color(AXIS)
ax.legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig("docs/figures/phase3_pareto_d5.png", facecolor=SURFACE)
print("wrote docs/figures/phase3_pareto_d5.png")
