"""Supplemental Figure S17 — what the ceiling does and does not track.

Human deposits only. Three candidate explanations for the spread in Figure 6:
experiment size, deposition year, and the kind of error column published.
"""
from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from pathlib import Path
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
S1 = "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.edgecolor": GRID, "axes.linewidth": .8})
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=.7, zorder=0); ax.set_axisbelow(True)

d = pd.read_csv(REPO / "results/mavedb_ceilings_v1.tsv", sep="\t")
m = pd.read_csv(REPO / "results/mavedb_metadata_v1.tsv", sep="\t")
d = d.merge(m[["urn", "published_date", "num_variants"]], on="urn", how="left")
h = d[(d.stratum == "human") & (d.status == "ok")].dropna(subset=["ceiling"]).copy()
h["year"] = pd.to_datetime(h.published_date, errors="coerce").dt.year

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

ax = axes[0]
x = h.dropna(subset=["num_variants"])
ax.scatter(x.num_variants, x.ceiling, s=26, color=S1, alpha=.5,
           linewidths=.8, edgecolors=SURF, zorder=3)
ax.set_xscale("log")
r = stats.spearmanr(np.log10(x.num_variants), x.ceiling)
ax.set_xlabel("Variants in the deposit  (log scale)", color=INK2)
ax.set_ylabel("Attenuation ceiling", color=INK2)
ax.set_title("A  Experiment size", loc="left", fontsize=10.5, fontweight="bold", color=INK)
ax.text(.97, .05, f"ρ = {r.statistic:+.2f}, P = {r.pvalue:.2g}\nn = {len(x)}",
        transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)
clean(ax)

ax = axes[1]
y = h.dropna(subset=["year"])
jitter = (np.random.default_rng(0).random(len(y)) - .5) * .55
ax.scatter(y.year + jitter, y.ceiling, s=26, color=S1, alpha=.5,
           linewidths=.8, edgecolors=SURF, zorder=3)
r2 = stats.spearmanr(y.year, y.ceiling)
ax.set_xlabel("Year of deposition", color=INK2)
ax.set_title("B  Deposition year", loc="left", fontsize=10.5, fontweight="bold", color=INK)
ax.text(.97, .05, f"ρ = {r2.statistic:+.2f}, P = {r2.pvalue:.2g}\nn = {len(y)}",
        transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)
clean(ax)

ax = axes[2]
order = [k for k in ("se", "ci", "sd_as_se", "replicates") if (h.method == k).sum() >= 3]
data = [h[h.method == k].ceiling.values for k in order]
bp = ax.boxplot(data, tick_labels=[f"{k}\n(n={len(v)})" for k, v in zip(order, data)],
                patch_artist=True, widths=.55, showfliers=False)
for b in bp["boxes"]:
    b.set_facecolor(S1); b.set_alpha(.35); b.set_edgecolor(S1)
for part in ("medians", "whiskers", "caps"):
    for a in bp[part]: a.set_color(S1); a.set_linewidth(1.4)
for i, v in enumerate(data, 1):
    ax.scatter(np.random.default_rng(i).normal(i, .05, len(v)), v, s=9,
               color=INK2, alpha=.28, zorder=4)
kw = stats.kruskal(*data) if len(data) > 1 else None
ax.set_xlabel("Error column the ceiling was computed from", color=INK2)
ax.set_title("C  Kind of error column", loc="left", fontsize=10.5, fontweight="bold", color=INK)
if kw: ax.text(.97, .05, f"Kruskal–Wallis P = {kw.pvalue:.2g}",
               transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)
clean(ax)

fig.suptitle("Experiment size and deposition year explain almost none of the spread",
             x=.006, ha="left", fontsize=12.5, fontweight="bold", color=INK)
fig.text(.006, .005, "Human deposits with a computable ceiling (n = 674). "
         "Deposition years are heavily unbalanced (most deposits are 2023 or later), "
         "so B has limited power to detect a trend. The size correlation in A is statistically detectable but weak (it accounts for about 2% of the variance).", fontsize=7.6, color=INK2)
fig.tight_layout(rect=[0, .04, 1, .93])
for ext in ("png", "pdf"):
    fig.savefig(REPO / f"results/fig_mavedb_ceiling_correlates.{ext}", dpi=200, bbox_inches="tight")
print(f"A size:  rho={r.statistic:+.3f} P={r.pvalue:.3g}")
print(f"B year:  rho={r2.statistic:+.3f} P={r2.pvalue:.3g}")
if kw: print(f"C type:  Kruskal-Wallis P={kw.pvalue:.3g}")
for k, v in zip(order, data): print(f"   {k:<12} n={len(v):>3}  median {np.median(v):.3f}")
