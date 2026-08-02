"""Fig 5 — scoring-definition reproducibility.

Two panels:
  A) AlphaGenome legacy definition (client v0.6.1, max|raw|, 1-Mb window) vs
     atlas standard (client 0.7.0, merged quantile, 16-kb window): rho from
     results/concordance_alphagenome.json.
  B) Nucleotide Transformer re-scored with the documented companion
     definition vs the legacy column: rho from
     results/concordance_nucleotide_transformer.json.

All statistics are read from the concordance JSONs (AGENTS.md rule 5);
scatter density from the shared-variant joins.

Usage (PYTHONPATH=src):  python figures/reproducibility.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
LEGACY = REPO / "data" / "raw" / "legacy_score_matrix_final.parquet"

PANELS = [  # (concordance json, new parquet, old col, new col, title, note)
    ("concordance_alphagenome.json", "alphagenome_scores.parquet",
     "alphagenome_splice", "alphagenome",
     "A  AlphaGenome: legacy vs atlas definition",
     "client v0.6.1 max|raw| 1-Mb vs 0.7.0 merged-quantile 16-kb"),
    ("concordance_nucleotide_transformer.json", "nt_scores.parquet",
     "nucleotide_transformer", "nucleotide_transformer",
     "B  NT-v2-500M: re-scored vs legacy column",
     "same documented definition (6000-bp window, masked 6-mer LLR)"),
]

KEY = ["chrom", "pos", "ref", "alt"]


def _norm_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["chrom"] = (df["chrom"].astype(str).str.replace("chr", "", regex=False))
    df["pos"] = df["pos"].astype("int64")
    return df


def main() -> None:
    legacy = _norm_key(pd.read_parquet(LEGACY))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0))

    for ax, (jname, newfile, oldcol, newcol, title, note) in zip(axes, PANELS):
        stats = json.loads((RESULTS / jname).read_text())
        new = _norm_key(pd.read_parquet(RESULTS / newfile,
                                        columns=KEY + [newcol]))
        old = legacy[KEY + [oldcol]].dropna()
        j = old.merge(new.dropna(), on=KEY,
                      suffixes=("_old", "_new"))
        ox = f"{oldcol}_old" if f"{oldcol}_old" in j.columns else oldcol
        ny = f"{newcol}_new" if f"{newcol}_new" in j.columns else newcol
        x, y = j[ox], j[ny]
        hb = ax.hexbin(x, y, gridsize=60, bins="log", cmap="viridis",
                       mincnt=1)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, color="#C0392B", lw=1.0, ls="--", alpha=0.8,
                label="identity")
        rho = stats["overall_rho"]
        n = stats["n_shared"]
        ax.text(0.03, 0.97,
                f"Spearman ρ = {rho:.4f}\nn = {n:,} shared variants",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"boxstyle": "round,pad=0.35", "fc": "white",
                      "ec": "#BBBBBB", "alpha": 0.95})
        ax.set_xlabel("Legacy score", fontsize=9)
        ax.set_ylabel("Atlas score", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=7,
                color="#666666", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)
        fig.colorbar(hb, ax=ax, label="log10(count)")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = RESULTS / f"fig_reproducibility.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
