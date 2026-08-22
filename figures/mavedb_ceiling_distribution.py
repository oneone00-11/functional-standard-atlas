"""Figure 3 — attenuation ceilings across human MaveDB deposits.

Human deposits only. The designed-stability protease platform is excluded: its
published interval is a curve-fitting CI on dG rather than a reproducibility
interval, and its targets are de novo mini-proteins rather than human variants.
"""
from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
S1, S2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.edgecolor": GRID, "axes.linewidth": .8})

d = pd.read_csv(REPO / "results/mavedb_ceilings_v1.tsv", sep="\t")
hd = d[(d.stratum == "human") & (d.status == "ok")].dropna(subset=["ceiling"])
h = hd.ceiling.sort_values()
# deposits whose error column is an explicit SE, reconcilable replicates or a CI —
# i.e. excluding the majority form, a deposited SD used as if it were an SE
sub = hd[hd.method != "sd_as_se"].ceiling.sort_values()
n = len(h)
med = float(h.median())
b90 = float((h < 0.90).mean())
b45 = float((h < 0.45).mean())

fig, ax = plt.subplots(figsize=(7.6, 4.6))
y = np.arange(1, n + 1) / n
ax.step(h, y, where="post", color=S1, lw=2, zorder=4, label=f"all deposits (n = {n})")
ax.fill_between(h, 0, y, step="post", color=S1, alpha=.10, zorder=2)
ys = np.arange(1, len(sub) + 1) / len(sub)
ax.step(sub, ys, where="post", color=S2, lw=2, zorder=5,
        label=f"explicit SE / replicates (n = {len(sub)})")

for x, lab, ha in ((0.45, "0.45\ncorrection lower bound", "left"), (0.90, "0.90", "right")):
    ax.axvline(x, color=INK2, lw=1.1, ls="--", zorder=3)
    ax.text(x + (0.012 if ha == "left" else -0.012), 0.34, lab, ha=ha, va="top",
            fontsize=8, color=INK2)

STUDY = [("BRCA1", 0.894), ("PALB2", 0.894), ("BARD1", 0.938)]
for i, (g, c) in enumerate(STUDY):
    yy = float((h < c).mean())
    ax.scatter([c], [yy], s=70, color=INK, zorder=7, edgecolors=SURF, linewidths=1.6)
    ax.annotate(g, (c, yy), textcoords="offset points",
                xytext=(9, -14 if g == "PALB2" else 6), fontsize=8.5, color=INK)

ax.text(.035, .95, f"all {n}: median {med:.3f}, {b90:.1%} below 0.90\n"
        f"excluding SD-as-SE ({len(sub)}): median {sub.median():.3f}, "
        f"{(sub < 0.90).mean():.1%} below 0.90\n{b45:.1%} below 0.45",
        transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.0)
ax.set_xlabel("Attenuation ceiling  √(reliability)", color=INK2)
ax.set_ylabel("Cumulative fraction of deposits", color=INK2)
ax.set_title("A third to a half of human MaveDB deposits cap correlation below 0.90",
             loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=10)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(.03, .80))
ax.grid(color=GRID, lw=.7, zorder=0); ax.set_axisbelow(True)
fig.text(.005, .005, "Human deposits only; the designed-protein protease-stability "
         "platform is excluded (see Methods). Black points mark the three assays in this study.", fontsize=7.6, color=INK2)
fig.tight_layout(rect=[0, .03, 1, 1])
for ext in ("png", "pdf"):
    fig.savefig(REPO / f"results/fig_mavedb_ceilings.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote fig_mavedb_ceilings  (n={n}, median {med:.3f}, <0.90 {b90:.1%}, <0.45 {b45:.1%})")
