"""Main figures for the hardened manuscript.

Reads only committed pipeline outputs under results/ (AGENTS.md rules 5–6) and
writes PNG + PDF pairs. Produces:

  fig_territory_corrected   Fig 2 — pooled ρ (7 genes) beside the same grid
                                    expressed as a fraction of the attenuation
                                    ceiling (3 genes with an error model)
  fig_splice_head_to_head   Fig 3 — offset decay, and paired Δρ for the ordering
                                    claims the text makes
  fig_classification        Fig 4 — AUROC by territory and LR+ against ACMG
                                    evidence-strength bands
  fig_rna_readout           Fig 5 — the same predictors against a second readout
  fig_selection_strategies  Fig 6 — leave-one-gene-out performance of the
                                    selection strategies a reader could follow

Usage (PYTHONPATH=src):  python figures/hardening_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

LABEL = {
    "alphagenome": "AlphaGenome", "spliceai_ds": "SpliceAI",
    "pangolin_score": "Pangolin", "cadd": "CADD", "alphamissense": "AlphaMissense",
    "evo2": "Evo2-7B", "gpn_msa": "GPN-MSA", "nucleotide_transformer": "NT-v2-500M",
    "phylop100way": "phyloP-100way", "phastcons100way": "phastCons-100way",
    "gnomad_af_global": "gnomAD AF (global)", "gnomad_af_popmax": "gnomAD AF (popmax)",
    "ens_broad": "Rank-avg (broad panel)", "ens_splice": "Rank-avg (splice panel)",
    # dbNSFP meta-predictors
    "revel": "REVEL", "bayesdel_addaf": "BayesDel (AF)", "clinpred": "ClinPred",
    "metarnn": "MetaRNN", "primateai": "PrimateAI", "vest4": "VEST4", "esm1b": "ESM-1b",
}
META_MODELS = ["revel", "bayesdel_addaf", "clinpred", "metarnn",
               "primateai", "vest4", "esm1b"]
MODEL_ORDER = ["alphagenome", "spliceai_ds", "pangolin_score", "cadd", "alphamissense",
               "evo2", "gpn_msa", "nucleotide_transformer", "phylop100way",
               "phastcons100way", "gnomad_af_global", "gnomad_af_popmax"]
STRATUM_LABEL = {
    "all": "All", "coding_or_utr": "Coding /\nUTR", "splice_region": "Splice\nregion",
    "splice_1_2": "Splice\n±1–2", "splice_3_10": "Splice\n3–10 bp",
    "splice_11_50": "Splice\n11–50 bp", "splice_deep": "Splice\n>50 bp",
    "clinvar_recorded": "ClinVar\nrecorded", "clinvar_absent": "ClinVar\nabsent",
    "missense": "Missense", "synonymous": "Synony-\nmous", "nonsense": "Nonsense",
    "indel": "Indel",
}
STRATUM_ORDER = ["all", "coding_or_utr", "splice_region", "splice_1_2",
                 "splice_3_10", "splice_11_50", "splice_deep"]


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        p = RESULTS / f"{name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def _heat(ax, M, rows, cols, norm, cmap, fmt="{:.2f}", fontsize=8):
    ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", color="#999999", fontsize=8)
                continue
            shade = cmap(norm(v))
            lum = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                    color="white" if lum < 0.5 else "#222222")
    ax.set_xticks(np.arange(-0.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------------------
def fig_territory_corrected() -> None:
    ext = pd.read_csv(RESULTS / "eval_ext_v1.tsv", sep="\t")
    att = pd.read_csv(RESULTS / "attenuation_v1.tsv", sep="\t")
    rel = pd.read_csv(RESULTS / "reliability_v1.tsv", sep="\t")

    raw = (ext[ext["stratum"].isin(STRATUM_ORDER)]
           .pivot(index="model", columns="stratum", values="pooled_rho")
           .reindex(MODEL_ORDER)[STRATUM_ORDER])
    # a realisation is only shown where at least two genes carry an error model:
    # dividing a single-gene ρ by a small ceiling amplifies noise rather than
    # correcting for it (the >50 bp ceiling is 0.30, from BARD1 alone)
    grp = att.groupby(["model", "stratum"])["realisation"]
    real = grp.median().where(grp.count() >= 2).unstack()
    real = real.reindex(MODEL_ORDER).reindex(columns=STRATUM_ORDER)
    ceil = (rel[rel["status"] == "validated"]
            .sort_values("method", ascending=False)
            .drop_duplicates(["gene", "stratum"])
            .groupby("stratum")["ceiling"].median().reindex(STRATUM_ORDER))
    genes_per_stratum = (att.groupby("stratum")["gene"].nunique().reindex(STRATUM_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.4),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28})
    rows = [LABEL[m] for m in MODEL_ORDER]

    cols_a = [STRATUM_LABEL[s] for s in STRATUM_ORDER]
    norm_a = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    _heat(axes[0], raw.to_numpy(float), rows, cols_a, norm_a, plt.get_cmap("RdBu_r"))
    axes[0].set_title("A  Pooled Spearman ρ — as measured\n"
                      "7 genes, 64,178 variants", loc="left", fontsize=10)

    cols_b = [f"{STRATUM_LABEL[s]}\nceiling {ceil[s]:.2f}" if np.isfinite(ceil.get(s, np.nan))
              else f"{STRATUM_LABEL[s]}\n(no error model)" for s in STRATUM_ORDER]
    norm_b = TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)
    _heat(axes[1], real.to_numpy(float), rows, cols_b, norm_b, plt.get_cmap("PuOr_r"))
    axes[1].set_title("B  Realisation: ρ as a fraction of the attenuation ceiling\n"
                      "genes with a per-variant error model only", loc="left", fontsize=10)

    for ax, cb in zip(axes, (plt.cm.ScalarMappable(norm=norm_a, cmap="RdBu_r"),
                             plt.cm.ScalarMappable(norm=norm_b, cmap="PuOr_r"))):
        fig.colorbar(cb, ax=ax, fraction=0.026, pad=0.02)

    fig.text(0.5, -0.08,
             "Panel B is the same grid as panel A after dividing by what the assay can "
             "actually measure in that stratum. Ceiling = √(assay reliability), from "
             "replicate scores (BRCA1) or CI-validated per-variant standard errors "
             "(BARD1, PALB2).\n"
             "Cells are shown only where at least two of those genes contribute; the "
             ">50 bp column is therefore blank, its ceiling of 0.30 resting on BARD1 "
             "alone — that stratum is close to unmeasurable rather than merely hard.\n"
             "Note that the apparent collapse at ±1–2 in panel A largely disappears in "
             "panel B: the drop is in the measurement, not in the predictors.",
             ha="center", fontsize=8, color="#444444")
    save(fig, "fig_territory_corrected")


# --------------------------------------------------------------------------
def fig_splice_head_to_head() -> None:
    h2h = pd.read_csv(RESULTS / "head_to_head_v1.tsv", sep="\t")
    panels = [("A  Territory and model-class claims", h2h[h2h["stratum"] != "missense"]),
              ("B  Missense: AlphaMissense vs the meta-predictors in clinical use",
               h2h[h2h["stratum"] == "missense"])]
    heights = [max(len(d), 1) for _, d in panels]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 0.42 * sum(heights) + 2.4),
                             gridspec_kw={"height_ratios": heights, "hspace": 0.32})

    for ax, (title, d) in zip(np.atleast_1d(axes), panels):
        d = d.iloc[::-1].reset_index(drop=True)
        if d.empty:
            ax.axis("off")
            continue
        y = np.arange(len(d))
        sig = d["steiger_p"] < 0.05
        ax.errorbar(d["delta_rho"], y,
                    xerr=[d["delta_rho"] - d["boot_ci_lo"], d["boot_ci_hi"] - d["delta_rho"]],
                    fmt="none", ecolor="#B9C6CE", lw=1.4, capsize=3, zorder=1)
        ax.scatter(d["delta_rho"], y, c=np.where(sig, "#C0392B", "#7F8C8D"), s=48, zorder=2)
        ax.axvline(0, color="#555555", lw=1.0, ls="--")
        labels = [f"{LABEL.get(a, a)} vs {LABEL.get(b, b)}"
                  + ("" if st == "missense"
                     else f"\n{STRATUM_LABEL.get(st, st).replace(chr(10), ' ')}")
                  for a, b, st in zip(d["model_a"], d["model_b"], d["stratum"])]
        ax.set_yticks(y, labels, fontsize=8)
        for i, pv in enumerate(d["steiger_p"]):
            ax.text(0.995, i, f"p = {pv:.3f}" if pv >= 0.001 else "p < 0.001",
                    transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=7.5, color="#C0392B" if pv < 0.05 else "#7F8C8D")
        ax.set_title(title, loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.26, y=0.12)

    np.atleast_1d(axes)[-1].set_xlabel(
        "\u0394 pooled Spearman \u03c1 (model A \u2212 model B), paired on shared variants")
    fig.text(0.02, -0.02,
             "Points are \u0394\u03c1 with a gene-cluster bootstrap 95% CI; p from Steiger's test for "
             "dependent correlations, DerSimonian\u2013Laird pooled over genes. Red = resolvable at "
             "\u03b1 = 0.05.\n"
             "Two correlations measured on the same variants are dependent, so overlapping "
             "marginal CIs are not a test of their difference \u2014 in panel B every marginal "
             "interval overlaps AlphaMissense's, yet all eight differences resolve.",
             fontsize=8, color="#444444")
    save(fig, "fig_splice_head_to_head")


# --------------------------------------------------------------------------
def fig_classification() -> None:
    ce = pd.read_csv(RESULTS / "clinical_evidence_v1.tsv", sep="\t")
    strata = [s for s in ["all", "coding_or_utr", "splice_region", "splice_3_10"]
              if s in set(ce["stratum"])]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0),
                             gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.3})

    auc = (ce[ce["stratum"].isin(strata)]
           .pivot(index="model", columns="stratum", values="auroc")
           .reindex(MODEL_ORDER)[strata])
    norm = TwoSlopeNorm(vmin=0.5, vcenter=0.75, vmax=1.0)
    _heat(axes[0], auc.to_numpy(float), [LABEL[m] for m in MODEL_ORDER],
          [STRATUM_LABEL[s] for s in strata], norm, plt.get_cmap("YlGnBu"), "{:.3f}")
    axes[0].set_title("A  AUROC against the assays' own functional calls\n"
                      "BARD1, PALB2, RAD51C — 31,835 labelled variants",
                      loc="left", fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="YlGnBu"), ax=axes[0],
                 fraction=0.026, pad=0.02)

    ax = axes[1]
    bands = [(2.08, 4.33, "Supporting", "#F4E1C1"), (4.33, 18.7, "Moderate", "#E8C39E"),
             (18.7, 350.0, "Strong", "#D89A6A")]
    for lo, hi, name, col in bands:
        ax.axhspan(lo, hi, color=col, alpha=0.45, zorder=0)
        ax.text(0.995, np.sqrt(lo * hi), name, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8, color="#7A5230")

    sub = ce[ce["stratum"].isin(strata)].copy()
    xs = {s: i for i, s in enumerate(strata)}
    marks = {"cadd": "o", "alphamissense": "s", "alphagenome": "^", "spliceai_ds": "v",
             "pangolin_score": "D", "evo2": "P", "gpn_msa": "X", "phylop100way": "*"}
    for model, mk in marks.items():
        d = sub[sub["model"] == model].dropna(subset=["lr_at_spec95_median"])
        if d.empty:
            continue
        ax.plot([xs[s] for s in d["stratum"]], d["lr_at_spec95_median"],
                marker=mk, ms=7, lw=1.2, label=LABEL[model])
    ax.set_yscale("log")
    ax.set_ylim(1.5, 45)  # the Strong band starts at 18.7; nothing reaches it
    ax.set_xticks(range(len(strata)),
                  [STRATUM_LABEL[s].replace("\n", " ") for s in strata], fontsize=8)
    ax.set_ylabel("Positive likelihood ratio at 95% specificity")
    ax.set_title("B  Evidence strength — no territory reaches PP3_Strong",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.02, -0.06,
             "Labels: GMM posterior ≥0.9 / ≤0.1 (BARD1, PALB2) or the authors' functional "
             "class (RAD51C); ambiguous calls dropped. Bands are Tavtigian-point LR "
             "thresholds at prior 0.10 and are indicative, not a ClinGen calibration.\n"
             "phastCons and gnomAD AF are absent from panel B: their scores are too "
             "coarsely quantised for any threshold to reach 95% specificity.",
             fontsize=8, color="#444444")
    save(fig, "fig_classification")


# --------------------------------------------------------------------------
def fig_rna_readout() -> None:
    pg = pd.read_csv(RESULTS / "rna_readout_v1.tsv", sep="\t")
    import json
    prov = json.loads((RESULTS / "rna_readout_v1.json").read_text())
    pooled = {e["model"]: e for e in prov["pooled_by_model"]}
    agree = prov["pooled_rna_vs_fitness"]["rho"]

    models = [m for m in MODEL_ORDER
              if m in pooled and pooled[m]["vs_fitness"] and pooled[m]["vs_rna"]]
    models.sort(key=lambda m: -pooled[m]["vs_fitness"]["rho"])

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    y = np.arange(len(models))
    fit = [pooled[m]["vs_fitness"]["rho"] for m in models]
    rna = [pooled[m]["vs_rna"]["rho"] for m in models]
    for i, (f, r) in enumerate(zip(fit, rna)):
        ax.plot([r, f], [i, i], color="#C7CDD1", lw=2.2, zorder=1, solid_capstyle="round")
    ax.scatter(fit, y, s=58, color="#2471A3", zorder=3, label="vs cell-fitness readout")
    ax.scatter(rna, y, s=58, color="#C0392B", zorder=3, label="vs RNA-abundance readout")
    ax.axvline(agree, color="#1E8449", lw=1.6, ls="--", zorder=2)
    ax.text(agree, -0.75, f"the assay's own two readouts agree at ρ = {agree:.3f}",
            color="#1E8449", fontsize=8, ha="center", va="bottom")
    n_above = sum(f > agree for f in fit)

    ax.set_yticks(y, [LABEL[m] for m in models], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Pooled Spearman ρ (coding/UTR variants, 4 genes)")
    ax.set_title("Predictor performance is readout-specific", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#EEEEEE", zorder=0)
    fig.text(0.02, -0.05,
             f"BRCA1, VHL, BARD1 and PALB2 publish an RNA-level score alongside the "
             f"fitness score the atlas freezes ({prov['n_coding']:,} coding variants; the RNA "
             "readout does not cover splice-region variants).\n"
             "Every coding-oriented predictor loses most of its signal on the RNA axis "
             "while the splice-aware models gain, inverting the ranking; and the "
             f"{n_above} strongest predictors track the fitness readout more closely "
             "than the assay's own second readout does.",
             fontsize=8, color="#444444")
    save(fig, "fig_rna_readout")


# --------------------------------------------------------------------------
def fig_selection_strategies() -> None:
    logo = pd.read_csv(RESULTS / "ensemble_logo_v1.tsv", sep="\t")
    strategies = [("single_global_cadd", "Single model everywhere (CADD)", "#7F8C8D"),
                  ("territory_logo", "Best model per territory", "#2471A3"),
                  ("rank_average_broad", "Rank-average, broad panel", "#B9770E"),
                  ("rank_average_splice", "Rank-average, splice panel", "#C0392B"),
                  ("best_possible_oracle", "Oracle single model (not achievable)", "#BDC3C7")]
    strata = [s for s in STRATUM_ORDER if s in set(logo["stratum"])]

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    width = 0.16
    x = np.arange(len(strata))
    for k, (col, name, colour) in enumerate(strategies):
        med = [logo.loc[logo["stratum"] == s, col].median() for s in strata]
        off = (k - (len(strategies) - 1) / 2) * width
        style = dict(hatch="//", alpha=0.55) if col == "best_possible_oracle" else {}
        ax.bar(x + off, med, width, color=colour, label=name, edgecolor="white", **style)

    ax.axhline(0, color="#555555", lw=0.9)
    ax.set_xticks(x, [STRATUM_LABEL[s].replace("\n", " ") for s in strata], fontsize=8.5)
    ax.set_ylabel("Held-out-gene Spearman ρ (median of 7)")
    ax.set_title("What a reader would actually get: selection strategies scored "
                 "leave-one-gene-out", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, -0.06,
             "For each held-out gene the strategy is fixed using the other six genes only, "
             "then scored on the held-out gene; bars are medians over the seven folds.\n"
             "Averaging the three splice-aware models wins every splice stratum. Picking a "
             "single best model per territory fails at ±1–2, where the winner is unstable "
             "across genes — the stratum where the paper recommends using no predictor.",
             fontsize=8, color="#444444")
    save(fig, "fig_selection_strategies")


# --------------------------------------------------------------------------
def fig_definition_sweep() -> None:
    import json
    grid = pd.read_csv(RESULTS / "definition_sweep_performance_v1.tsv", sep="\t")
    chk = json.loads((RESULTS / "definition_sweep_v1.json").read_text())

    order = ["coding_or_utr", "splice_1_2", "splice_3_10", "splice_11_50", "splice_deep"]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.3})

    ax = axes[0]
    widths = sorted(grid["width"].unique())
    marks = dict(zip(widths, ["o", "s", "^", "D"]))
    cols = {"merged_quantile": "#C0392B", "merged_raw": "#2471A3",
            "max_abs_raw": "#1E8449", "max_quantile": "#B9770E"}
    for xi, st in enumerate(order):
        sub = grid[grid["stratum"] == st]
        for _, r in sub.iterrows():
            jitter = (widths.index(r["width"]) - 1.5) * 0.07
            ax.scatter(xi + jitter, r["pooled_rho"], s=34, zorder=3,
                       marker=marks[r["width"]], color=cols[r["definition"]],
                       edgecolor="white", linewidth=0.5)
        lo, hi = sub["pooled_rho"].min(), sub["pooled_rho"].max()
        ax.plot([xi, xi], [lo, hi], color="#BDC3C7", lw=6, alpha=0.45, zorder=1,
                solid_capstyle="round")
        ax.text(xi, hi + 0.03, f"spread\n{hi-lo:.2f}", ha="center", va="bottom",
                fontsize=7.5, color="#555555")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_ylim(top=grid["pooled_rho"].max() + 0.13)   # headroom for the spread labels
    ax.set_xticks(range(len(order)),
                  [STRATUM_LABEL[s].replace("\n", " ") for s in order], fontsize=8.5)
    ax.set_ylabel("Pooled Spearman ρ")
    ax.set_title("A  Every definition × window combination, within stratum",
                 loc="left", fontsize=10)
    h = [plt.Line2D([], [], marker=marks[w], ls="", color="#555555",
                    label=f"{w//1024} kb") for w in widths]
    h += [plt.Line2D([], [], marker="o", ls="", color=c, label=d)
          for d, c in cols.items()]
    ax.legend(handles=h, frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    sa = chk["score_agreement_by_region"]
    perf = chk["performance_on_shared_variants"]
    regions = [r for r in order if r in sa and r in perf]
    x = np.arange(len(regions))
    ax.bar(x - 0.2, [sa[r] for r in regions], 0.4, color="#7F8C8D",
           label="agreement between the two score columns")
    ax.bar(x + 0.2, [abs(perf[r]["difference"]) for r in regions], 0.4, color="#C0392B",
           label="| difference in benchmark performance |")
    for i, r in enumerate(regions):
        ax.text(i + 0.2, abs(perf[r]["difference"]) + 0.02,
                f"{abs(perf[r]['difference']):.3f}", ha="center", fontsize=7.5,
                color="#C0392B")
    ax.axhline(sa["overall"], color="#2471A3", ls="--", lw=1.4)
    ax.text(len(regions) - 0.5, sa["overall"] + 0.02,
            f"headline concordance ρ = {sa['overall']:.3f}", ha="right",
            fontsize=8, color="#2471A3")
    ax.set_xticks(x, [STRATUM_LABEL[r].replace("\n", " ") for r in regions], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("B  Score disagreement does not transfer to conclusions",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center left")
    ax.spines[["top", "right"]].set_visible(False)

    r = chk["replicate_legacy_definition"]
    fig.text(0.02, -0.10,
             "Panel A: 2,722 variants scored under four windows × four aggregations "
             "(one API call per variant × window supplies every aggregation). Grey bars "
             "span the 16 combinations.\n"
             "Panel B: the two published definitions disagree at ρ = "
             f"{sa['overall']:.3f} overall and as low as {min(v for k, v in sa.items() if k != 'overall'):.3f} "
             "in one region, yet their pooled ρ against the functional standard differs by "
             f"at most {chk['max_abs_performance_difference']:.3f} in any stratum.\n"
             "Replicating the legacy definition under the current client reproduces the "
             f"legacy column at ρ = {r['rho_vs_legacy_column']:.3f} (n = {r['n']:,}), so the "
             "client version is not the source of the disagreement.",
             fontsize=8, color="#444444")
    save(fig, "fig_definition_sweep")


def main() -> None:
    fig_territory_corrected()
    fig_splice_head_to_head()
    fig_classification()
    fig_rna_readout()
    fig_selection_strategies()
    fig_definition_sweep()


if __name__ == "__main__":
    main()
