"""Fig 3 — the saturation ceiling and offset decay.

Panel A: distribution of oriented functional scores per region bucket, from
the atlas score matrix — at splice ±1–2 the readout is near-deterministic
(saturated), leaving no variance for any model to rank.
Panel B: pooled Spearman ρ per offset bin for every scored model — all lines
converge to ~0 beyond 10 bp (offset decay).

Reads only results/score_matrix_atlas_v1.parquet + results/eval_*.json
(AGENTS.md rule 5); missing eval files are skipped with a note.

Usage (PYTHONPATH=src):  python figures/saturation_decay.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from atlas.evaluate import classify_region

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

BINS = ["splice_1_2", "splice_3_10", "splice_11_50", "splice_deep"]
BIN_LABELS = ["±1–2 bp", "3–10 bp", "11–50 bp", ">50 bp"]

MODELS = [  # (eval key, label, color, lw)
    ("alphagenome", "AlphaGenome", "#C0392B", 2.2),
    ("spliceai_ds", "SpliceAI", "#2471A3", 2.2),
    ("pangolin_score", "Pangolin", "#1E8449", 2.2),
    ("cadd", "CADD", "#7D3C98", 1.2),
    ("evo2", "Evo2-7B", "#D35400", 1.6),
    ("gpn_msa", "GPN-MSA", "#B9770E", 1.2),
    ("nucleotide_transformer", "NT-v2-500M", "#16A085", 1.2),
    ("phylop100way", "phyloP-100way", "#707B7C", 1.2),
    ("phastcons100way", "phastCons-100way", "#A6ACAF", 1.2),
    ("gnomad_af_global", "gnomAD AF (global)", "#D4AC0D", 1.0),
    ("gnomad_af_popmax", "gnomAD AF (popmax)", "#F0B27A", 1.0),
]


def panel_a(ax: plt.Axes) -> None:
    mat = pd.read_parquet(RESULTS / "score_matrix_atlas_v1.parquet",
                          columns=["gene", "hgvs_c", "functional_pathogenicity"])
    mat = mat.dropna(subset=["functional_pathogenicity"])
    mat["bin"] = mat["hgvs_c"].map(classify_region)
    # within-gene percentile rank: assays have incompatible raw scales, and
    # saturation = clustering at the damaging extreme with collapsed variance
    mat["pct"] = mat.groupby("gene")["functional_pathogenicity"].rank(pct=True)
    groups, labels = [], []
    for b, lab in zip(["coding_or_utr"] + BINS, ["Coding / UTR"] + BIN_LABELS):
        v = mat.loc[mat["bin"] == b, "pct"].to_numpy()
        if len(v):
            groups.append(v)
            labels.append(f"{lab}\n(n={len(v):,})")
    parts = ax.violinplot(groups, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#2F4B5C"); pc.set_alpha(0.55)
    parts["cmedians"].set_color("#C0392B")
    ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
    ax.set_ylabel("Within-assay percentile of functional score")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("A  Readout saturates at the damaging extreme (splice ±1–2)",
                 loc="left", fontsize=10)


def panel_b(ax: plt.Axes) -> None:
    x = np.arange(len(BINS))
    plotted = []
    for key, label, color, lw in MODELS:
        path = RESULTS / f"eval_{key}.json"
        if not path.exists():
            print(f"note: {path.name} missing — {label} skipped")
            continue
        data = json.loads(path.read_text())
        y = []
        for b in BINS:
            st = data["strata"].get(b)
            y.append(st["pooled"]["pooled_rho"] if st and st.get("pooled") else np.nan)
        ax.plot(x, y, marker="o", ms=4, color=color, lw=lw, label=label)
        plotted.append(label)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(x, BIN_LABELS)
    ax.set_xlabel("Intronic offset from splice site")
    ax.set_ylabel("Pooled Spearman ρ")
    ax.set_title("B  Every model decays to ~0 beyond 10 bp", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    print("models plotted:", ", ".join(plotted))


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    panel_a(ax1)
    panel_b(ax2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = RESULTS / f"fig_saturation_decay.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
