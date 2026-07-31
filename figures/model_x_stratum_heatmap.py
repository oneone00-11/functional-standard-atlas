#!/usr/bin/env python
"""Main figure: model × stratum heatmap of pooled Spearman ρ.

Reads results/eval_<model>.json (produced by atlas.evaluate) and renders the
pooled DerSimonian–Laird correlation for every model × stratum combination.
Missing cells (model does not score that stratum, e.g. AlphaMissense ×
splice) are shown as "—".

Run from the repo root:
    .venv/bin/python figures/model_x_stratum_heatmap.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

MODELS = [
    ("alphagenome", "AlphaGenome"),
    ("spliceai_ds", "SpliceAI*"),
    ("pangolin_score", "Pangolin*"),
    ("cadd", "CADD"),
    ("alphamissense", "AlphaMissense*"),
    ("gpn_msa", "GPN-MSA"),
    ("phylop100way", "phyloP-100way"),
    ("phastcons100way", "phastCons-100way"),
    ("gnomad_af_global", "gnomAD AF (global)"),
    ("gnomad_af_popmax", "gnomAD AF (popmax)"),
]
STRATA = [
    ("all", "All"),
    ("coding_or_utr", "Coding / UTR"),
    ("splice_region", "Splice region (all)"),
    ("splice_1_2", "Splice ±1–2"),
    ("splice_3_10", "Splice 3–10 bp"),
    ("splice_11_50", "Splice 11–50 bp"),
    ("splice_deep", "Splice >50 bp"),
    ("clinvar_recorded", "ClinVar-recorded"),
]


def load_rhos() -> np.ndarray:
    grid = np.full((len(MODELS), len(STRATA)), np.nan)
    for i, (key, _label) in enumerate(MODELS):
        path = RESULTS / f"eval_{key}.json"
        data = json.loads(path.read_text())
        for j, (stratum, _slabel) in enumerate(STRATA):
            st = data["strata"].get(stratum)
            if st and st.get("pooled"):
                grid[i, j] = st["pooled"]["pooled_rho"]
    return grid


def main() -> None:
    grid = load_rhos()

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F2F2F2")
    masked = np.ma.masked_invalid(grid)
    vmax = 0.5
    im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(STRATA)), [s for _k, s in STRATA], rotation=30, ha="right")
    ax.set_yticks(range(len(MODELS)), [m for _k, m in MODELS])
    for i in range(len(MODELS)):
        for j in range(len(STRATA)):
            v = grid[i, j]
            if math.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", color="#AAAAAA", fontsize=9)
            else:
                color = "white" if abs(v) > 0.32 else "#222222"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(STRATA)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pooled Spearman ρ vs functional score (DL meta-analysis)")
    ax.set_title("Model performance by variant region — functional-standard-atlas core7")
    fig.text(
        0.01, 0.01,
        "* AlphaMissense scores missense SNVs only; SpliceAI scores SNVs only (indels not scored); “—” = model does not score that stratum.\n"
        "Pooled across 7 genes (DerSimonian–Laird, Fisher-z); cells use pooled_rho from results/eval_*.json.",
        fontsize=7.5, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = RESULTS / "fig_model_x_stratum_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
