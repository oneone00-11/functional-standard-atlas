"""Fig 1 — atlas overview.

Three panels, all numbers read from pipeline files (AGENTS.md rule 5):
  A) Variants per gene, stacked by region class (classify_region on hgvs_c).
  B) Functional score (functional_pathogenicity) distribution per gene.
  C) Score coverage (% of 64,178 variants) per model, from the atlas matrix.

Usage (PYTHONPATH=src):  python figures/atlas_overview.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from atlas.evaluate import classify_region

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FROZEN = REPO / "data" / "frozen" / "frozen-matrix-v1.parquet"
MATRIX = RESULTS / "score_matrix_atlas_v1.parquet"

BINS = [
    ("coding_or_utr", "Coding / UTR", "#2471A3"),
    ("splice_1_2", "Splice ±1–2", "#C0392B"),
    ("splice_3_10", "Splice 3–10 bp", "#E67E22"),
    ("splice_11_50", "Splice 11–50 bp", "#F1C40F"),
    ("splice_deep", "Splice >50 bp", "#7D3C98"),
]

MODELS = [  # (column, label) — order = heatmap order
    ("alphagenome", "AlphaGenome"),
    ("spliceai_ds", "SpliceAI"),
    ("pangolin_score", "Pangolin"),
    ("cadd", "CADD"),
    ("alphamissense", "AlphaMissense"),
    ("evo2", "Evo2-7B"),
    ("gpn_msa", "GPN-MSA"),
    ("nucleotide_transformer", "NT-v2-500M"),
    ("phylop100way", "phyloP-100way"),
    ("phastcons100way", "phastCons-100way"),
    ("gnomad_af_global", "gnomAD AF (global)"),
    ("gnomad_af_popmax", "gnomAD AF (popmax)"),
]


def main() -> None:
    frozen = pd.read_parquet(FROZEN)
    frozen["bin"] = frozen["hgvs_c"].map(classify_region)
    genes = sorted(frozen["gene"].unique())
    n_total = len(frozen)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})

    # --- Panel A: stacked variants per gene by region class -----------------
    ax = axes[0]
    bottom = np.zeros(len(genes))
    for key, label, color in BINS:
        vals = np.array([(frozen.loc[frozen["gene"] == g, "bin"] == key).sum()
                         for g in genes], dtype=float)
        ax.bar(genes, vals, bottom=bottom, color=color, label=label,
               width=0.72, edgecolor="white", linewidth=0.4)
        bottom += vals
    for i, tot in enumerate(bottom):
        ax.text(i, tot + 220, f"{int(tot):,}", ha="center", fontsize=7.5,
                color="#333333")
    ax.set_ylabel("Variants")
    ax.set_title(f"A  Atlas composition — {n_total:,} SGE variants", loc="left",
                 fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_ylim(0, bottom.max() * 1.24)

    # --- Panel B: functional score distribution per gene --------------------
    ax = axes[1]
    data = [frozen.loc[frozen["gene"] == g, "functional_pathogenicity"].dropna()
            for g in genes]
    bp = ax.boxplot(data, tick_labels=genes, showfliers=False, widths=0.62,
                    medianprops={"color": "#C0392B", "lw": 1.4},
                    boxprops={"color": "#2471A3"},
                    whiskerprops={"color": "#2471A3"},
                    capprops={"color": "#2471A3"})
    ax.axhline(0, color="#999999", lw=0.8, ls="--")
    ax.set_ylabel("Functional score (oriented: higher = more damaging)")
    ax.set_title("B  Functional score distribution", loc="left", fontsize=10,
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # --- Panel C: model coverage --------------------------------------------
    ax = axes[2]
    mat = pd.read_parquet(MATRIX, columns=[c for c, _ in MODELS])
    cov = [100.0 * mat[c].notna().mean() for c, _ in MODELS]
    labels = [lb for _c, lb in MODELS]
    ypos = np.arange(len(MODELS))[::-1]
    colors = ["#C0392B" if v > 99 else "#2471A3" if v > 70 else "#E67E22"
              if v > 50 else "#7D3C98" if v > 20 else "#999999" for v in cov]
    ax.barh(ypos, cov, color=colors, height=0.66, edgecolor="white")
    for y, v in zip(ypos, cov):
        ax.text(v + 1.2, y, f"{v:.0f}%", va="center", fontsize=7.5,
                color="#333333")
    ax.set_yticks(ypos, labels, fontsize=8)
    ax.set_xlabel("Variants scored (%)")
    ax.set_xlim(0, 118)
    ax.set_title("C  Score coverage by model", loc="left", fontsize=10,
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = RESULTS / f"fig_atlas_overview.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
