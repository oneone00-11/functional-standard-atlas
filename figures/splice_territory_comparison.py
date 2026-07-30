"""Fig 2 — splice-territory comparison across splice-aware models.

Pooled Spearman ρ (DL meta-analysis) by intronic offset bin for AlphaGenome,
SpliceAI and Pangolin, with per-gene ρ as jittered points. Reads only
results/eval_<model>.json (AGENTS.md rule 5); missing models are skipped with
a note so the figure works before/after the Pangolin full run.

Usage (PYTHONPATH=src):  python figures/splice_territory_comparison.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

MODELS = [  # (eval key, label, color, marker)
    ("alphagenome", "AlphaGenome", "#C0392B", "o"),
    ("spliceai_ds", "SpliceAI", "#2471A3", "s"),
    ("pangolin", "Pangolin", "#1E8449", "^"),
]
BINS = [
    ("splice_1_2", "±1–2 bp"),
    ("splice_3_10", "3–10 bp"),
    ("splice_11_50", "11–50 bp"),
    ("splice_deep", ">50 bp"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(BINS))
    plotted = []
    offsets = np.linspace(-0.22, 0.22, len(MODELS))

    for mi, (key, label, color, marker) in enumerate(MODELS):
        path = RESULTS / f"eval_{key}.json"
        if not path.exists():
            print(f"note: {path.name} missing — {label} skipped (run the scorer + evaluate first)")
            continue
        data = json.loads(path.read_text())
        pooled, lo, hi, per_gene = [], [], [], []
        for b, _ in BINS:
            st = data["strata"].get(b)
            if st and st.get("pooled"):
                p = st["pooled"]
                pooled.append(p["pooled_rho"])
                lo.append(p["rho_ci_lo"])
                hi.append(p["rho_ci_hi"])
                per_gene.append([(g["gene"], g["rho"]) for g in st["per_gene"] if g.get("rho") is not None])
            else:
                pooled.append(np.nan); lo.append(np.nan); hi.append(np.nan); per_gene.append([])
        pooled, lo, hi = map(np.asarray, (pooled, lo, hi))
        xi = x + offsets[mi]
        ax.errorbar(xi, pooled, yerr=[pooled - lo, hi - pooled], fmt=marker + "-",
                    color=color, ecolor=color, elinewidth=1, capsize=3, ms=7,
                    lw=1.8, label=label, zorder=3)
        rng = np.random.default_rng(7)
        for j, genes in enumerate(per_gene):
            if not genes:
                continue
            gy = [r for _g, r in genes]
            gx = xi[j] + rng.uniform(-0.045, 0.045, len(gy))
            ax.scatter(gx, gy, s=14, color=color, alpha=0.45, zorder=2, edgecolors="none")
        plotted.append(label)

    ax.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax.set_xticks(x, [b for _k, b in BINS])
    ax.set_xlabel("Intronic offset from splice site")
    ax.set_ylabel("Pooled Spearman ρ vs functional score")
    ax.set_title("Splice territory by offset — pooled ρ (lines, 95% CI) and per-gene ρ (points)")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = RESULTS / f"fig_splice_territory.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)
    print("models plotted:", ", ".join(plotted))


if __name__ == "__main__":
    main()
