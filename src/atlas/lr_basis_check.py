"""Positive likelihood ratios at 95% specificity on two bases.

`atlas.clinical_evidence` reads the positive likelihood ratio at the most sensitive
observed score threshold whose false-positive rate is at most 5%
(`lr_at_specificity`) and reports the median of the per-gene ratios. The companion
splice-region benchmark reads it on the empirical ROC interpolated to exactly 95%
specificity. The two can differ where a score's values are coarse, so this module
computes both for every model x territory cell of `results/clinical_evidence_v1.tsv`,
per gene and as the same cross-gene median, and stops if its scanned medians do not
reproduce that table.

Outputs
-------
results/lr_basis_check_v1.tsv        one row per model x territory
results/lr_basis_check_genes_v1.tsv  one row per model x territory x gene
results/lr_basis_check_v1.json       the figures Note S15 and the Results text quote
docs/lr-basis-check.md               the same comparison as a page

Usage (PYTHONPATH=src):  python -m atlas.lr_basis_check
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.clinical_evidence import (ACMG_TIERS, MIN_NEG, MIN_POS, MODELS, STRATA, acmg_tier,
                                     binarise, lr_at_specificity)
from atlas.evaluate import classify_region

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
PAGE = REPO / "docs" / "lr-basis-check.md"

SPEC = 0.95
STRONG = next(t for t, name in ACMG_TIERS if name == "Strong")

# The four ratios the Results text quotes, and the cells Fig. 6B plots
# (figures/hardening_figures.py, fig_classification: `marks` and `strata`).
QUOTED = (("alphagenome", "splice_region"), ("pangolin_score", "splice_region"),
          ("spliceai_ds", "splice_region"), ("cadd", "coding_or_utr"))
FIG6B_MODELS = ("cadd", "alphamissense", "alphagenome", "spliceai_ds", "pangolin_score",
                "evo2", "gpn_msa", "phylop100way")
FIG6B_STRATA = ("all", "coding_or_utr", "splice_region", "splice_3_10")

LABEL = {"alphagenome": "AlphaGenome", "spliceai_ds": "SpliceAI", "pangolin_score": "Pangolin",
         "cadd": "CADD", "alphamissense": "AlphaMissense", "evo2": "Evo2", "gpn_msa": "GPN-MSA",
         "nucleotide_transformer": "NT-v2-500M", "phylop100way": "phyloP-100way",
         "phastcons100way": "phastCons-100way", "gnomad_af_global": "gnomAD AF (global)",
         "gnomad_af_popmax": "gnomAD AF (popmax)", "revel": "REVEL", "bayesdel_addaf": "BayesDel addAF",
         "clinpred": "ClinPred", "metarnn": "MetaRNN", "primateai": "PrimateAI", "vest4": "VEST4",
         "esm1b": "ESM-1b"}
STRAT = {"all": "all", "coding_or_utr": "coding/UTR", "splice_region": "splice region",
         "splice_1_2": "splice ±1–2", "splice_3_10": "splice 3–10 bp",
         "splice_11_50": "splice 11–50 bp", "splice_deep": "deep intronic"}


def roc_points(y: np.ndarray, s: np.ndarray):
    """Empirical ROC at every distinct score value, thresholds decreasing, so FPR and
    TPR are non-decreasing."""
    neg, pos = np.sort(s[y == 0]), np.sort(s[y == 1])
    u = np.unique(s)[::-1]
    fpr = 1.0 - np.searchsorted(neg, u, side="left") / len(neg)
    tpr = 1.0 - np.searchsorted(pos, u, side="left") / len(pos)
    return u, fpr, tpr


def lr_interp_at_specificity(y: np.ndarray, s: np.ndarray, spec: float = SPEC) -> tuple[float, float]:
    """(LR+, sensitivity) at exactly `spec`: the companion benchmark's definition
    (variant-fm-benchmark, phase1/src/phase5_likelihood_ratios.py).

    Sensitivity is interpolated linearly between the two adjacent observed operating
    points that bracket FPR = 1 - spec and divided by exactly 1 - spec. When the top score
    value is already a tie group whose FPR exceeds 1 - spec, the segment starts at the
    empty call set (FPR = TPR = 0): the value is then read between calling nothing
    positive and calling that whole tie group positive, which no single threshold does.
    """
    target = 1.0 - spec
    _, fpr, tpr = roc_points(y, s)
    k = int(np.searchsorted(fpr, target, side="right"))      # fpr[k-1] <= target < fpr[k]
    fa, ta = (0.0, 0.0) if k == 0 else (fpr[k - 1], tpr[k - 1])
    if k >= len(fpr):
        return float(ta / target), float(ta)
    fb, tb = fpr[k], tpr[k]
    t = ta if fb == fa else ta + (tb - ta) * (target - fa) / (fb - fa)
    return float(t / target), float(t)


def compare(mat: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both bases per gene and per cell, selecting exactly as clinical_evidence.evaluate does."""
    df = mat.merge(labels, on=["variant_id", "gene"], how="inner")
    df["region"] = df["hgvs_c"].map(classify_region)
    cells, genes = [], []
    for model in MODELS:
        if model not in df.columns:
            continue
        for stratum in STRATA:
            if stratum == "all":
                sub = df
            elif stratum == "splice_region":
                sub = df[df["region"].str.startswith("splice_")]
            else:
                sub = df[df["region"] == stratum]
            sub = sub.dropna(subset=[model])
            rows = []
            for gene, g in sub.groupby("gene"):
                y, s = g["y"].to_numpy(), g[model].to_numpy()
                n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
                if n_pos < MIN_POS or n_neg < MIN_NEG:
                    continue
                lr_s, thr = lr_at_specificity(y, s, SPEC)
                spec_obs = float(1.0 - (s[y == 0] >= thr).mean()) if np.isfinite(thr) else float("nan")
                lr_i, sens_i = lr_interp_at_specificity(y, s, SPEC)
                rows.append({"model": model, "stratum": stratum, "gene": gene,
                             "n_abnormal": n_pos, "n_normal": n_neg, "threshold": thr,
                             "lr_scanned": lr_s, "specificity_realised": spec_obs,
                             "lr_interpolated": lr_i, "sensitivity_interpolated": sens_i})
            if len(rows) < 2:           # clinical_evidence reports a cell only across two genes
                continue
            pg = pd.DataFrame(rows)
            scan = float(pg["lr_scanned"].median())
            interp = float(pg["lr_interpolated"].median())
            cells.append({"model": model, "stratum": stratum, "k_genes": len(pg),
                          "genes_with_scanned_value": int(pg["lr_scanned"].notna().sum()),
                          "lr_scanned_median": scan, "lr_interpolated_median": interp,
                          "abs_change": abs(interp - scan) if math.isfinite(scan) else float("nan"),
                          "band_scanned": acmg_tier(scan), "band_interpolated": acmg_tier(interp),
                          "specificity_realised_min": float(pg["specificity_realised"].min()),
                          "specificity_realised_max": float(pg["specificity_realised"].max()),
                          "quoted_in_text": (model, stratum) in QUOTED,
                          "in_fig6b": model in FIG6B_MODELS and stratum in FIG6B_STRATA})
            genes.extend(rows)
    return pd.DataFrame(cells), pd.DataFrame(genes)


def check_reproduces(cells: pd.DataFrame, table: Path) -> None:
    """The scanned medians must be clinical_evidence's own; otherwise nothing here describes it."""
    want = pd.read_csv(table, sep="\t").set_index(["model", "stratum"])["lr_at_spec95_median"]
    got = cells.set_index(["model", "stratum"])["lr_scanned_median"]
    if set(want.index) != set(got.index):
        raise SystemExit(f"[lr_basis_check] cells differ from {table.name}: "
                         f"{sorted(set(want.index) ^ set(got.index))[:5]}")
    bad = [(k, want[k], got[k]) for k in want.index
           if not ((math.isnan(want[k]) and math.isnan(got[k])) or abs(want[k] - got[k]) <= 1e-9)]
    if bad:
        raise SystemExit(f"[lr_basis_check] scanned medians do not reproduce {table.name}: {bad[:5]}")


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    return o


def summary(cells: pd.DataFrame, genes: pd.DataFrame) -> dict:
    fin = cells[np.isfinite(cells["lr_scanned_median"])]
    order = {c: i for i, c in enumerate(QUOTED)}
    q = (cells[cells["quoted_in_text"]]
         .assign(_o=lambda t: [order[(m, s)] for m, s in zip(t["model"], t["stratum"])])
         .sort_values("_o").drop(columns="_o"))
    qg = genes[[(m, s) in QUOTED for m, s in zip(genes["model"], genes["stratum"])]]
    fig = fin[fin["in_fig6b"]]
    fg = genes[genes["model"].isin(FIG6B_MODELS) & genes["stratum"].isin(FIG6B_STRATA)
               & np.isfinite(genes["lr_scanned"])]
    top = fig.loc[fig["abs_change"].idxmax()]
    return _clean({
        "specificity": SPEC, "strong_threshold": STRONG,
        "cells": len(cells), "cells_with_scanned_value": len(fin),
        "max_scanned_median": fin["lr_scanned_median"].max(),
        "max_interpolated_median": cells["lr_interpolated_median"].max(),
        "medians_reaching_strong_scanned": int((fin["lr_scanned_median"] >= STRONG).sum()),
        "medians_reaching_strong_interpolated": int((cells["lr_interpolated_median"] >= STRONG).sum()),
        "quoted_cells": q[["model", "stratum", "lr_scanned_median", "lr_interpolated_median", "abs_change",
                           "band_scanned", "band_interpolated", "specificity_realised_min",
                           "specificity_realised_max"]].to_dict("records"),
        "quoted_specificity_realised_min": qg["specificity_realised"].min(),
        "quoted_specificity_realised_max": qg["specificity_realised"].max(),
        "quoted_max_abs_change": q["abs_change"].max(),
        "fig6b_cells": len(fig),
        "fig6b_band_changes": int((fig["band_scanned"] != fig["band_interpolated"]).sum()),
        "fig6b_max_abs_change": top["abs_change"],
        "fig6b_max_abs_change_cell": [top["model"], top["stratum"]],
        "fig6b_specificity_realised_min": fg["specificity_realised"].min(),
        "fig6b_specificity_realised_max": fg["specificity_realised"].max(),
    })


def _f3(x) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x:.3f}"


def _pct(x) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{100 * x:.2f}%"


def _band(t: str) -> str:
    return {"below supporting": "below Supporting"}.get(t, t)


def render_page(cells: pd.DataFrame, genes: pd.DataFrame, s: dict) -> str:
    fin = cells[np.isfinite(cells["lr_scanned_median"])]
    ne = cells[~np.isfinite(cells["lr_scanned_median"])]
    up = fin[fin["lr_interpolated_median"] > fin["lr_scanned_median"] + 1e-12]
    down = fin[fin["lr_interpolated_median"] < fin["lr_scanned_median"] - 1e-12]
    changed = fin[fin["band_scanned"] != fin["band_interpolated"]]
    g = genes[np.isfinite(genes["lr_scanned"])]
    hi = g[g["specificity_realised"] > 0.955]
    strong_gene = genes[(genes["lr_scanned"] >= STRONG) | (genes["lr_interpolated"] >= STRONG)]
    top_s = fin.loc[fin["lr_scanned_median"].idxmax()]
    top_i = cells.loc[cells["lr_interpolated_median"].idxmax()]
    L: list[str] = []
    A = L.append
    A("# LR+ basis check: scanned threshold versus exactly 95% specificity")
    A("")
    A("Written by `python -m atlas.lr_basis_check`; the tables behind this page are "
      "`results/lr_basis_check_v1.tsv` (cells) and `results/lr_basis_check_genes_v1.tsv` (genes).")
    A("")
    A(f"**Conclusion.** The positive likelihood ratios in this study are read at the most sensitive observed "
      f"score threshold that reaches 95% specificity (`atlas.clinical_evidence.lr_at_specificity`), not "
      f"interpolated to exactly 95% specificity. On the interpolated basis no territory reaches the Strong band "
      f"either: the largest median is {_f3(top_i.lr_interpolated_median)} ({LABEL[top_i.model]}, "
      f"{STRAT[top_i.stratum]}), against {_f3(top_s.lr_scanned_median)} scanned.")
    A("")
    A(f"Below the Strong threshold the basis is not immaterial. Interpolation changes {len(changed)} band "
      f"assignment{'s' if len(changed) != 1 else ''} and raises {len(up)} cell{'s' if len(up) != 1 else ''}. "
      f"It also gives a value to the {len(ne)} cells that have none on the scanned basis.")
    A("")
    A("## What the atlas computes")
    A("")
    A("- `lr_at_specificity` scans the observed score values in ascending order and uses the first whose "
      "false-positive rate is at most 1 − spec; a false-positive rate of 0 is bounded at 1/(n_normal + 1). "
      "The manuscript's Methods describe the same scan.")
    A("- It is applied per gene (at least 10 abnormal and 10 normal variants). The reported "
      "`lr_at_spec95_median` in `results/clinical_evidence_v1.tsv` is the **median of the per-gene ratios**, "
      "over BARD1, PALB2 and RAD51C, the three genes with assay-derived functional calls. The evidence band "
      "is assigned to that median.")
    A("")
    A("## How this page is computed")
    A("")
    A("- This module repeats clinical_evidence's label binarisation, merge, territory assignment, gene gating "
      f"and median. Its scanned medians reproduce `results/clinical_evidence_v1.tsv` in all {len(cells)} cells "
      f"({len(fin)} with a value, {len(ne)} not evaluable), and it stops if they do not.")
    A("- The interpolated value is the companion benchmark's definition, implemented here as "
      "`lr_interp_at_specificity`. Sensitivity is read linearly on the empirical ROC between the two observed "
      "thresholds that bracket FPR = 0.05, and LR+ = sensitivity / 0.05. When the top score value is already a "
      "tie group holding more than 5% of the normal variants, the segment starts at the empty call set.")
    A("- Realised specificity is 1 − FPR at the scanned threshold, per gene.")
    A("")
    A("## The four ratios quoted in the text")
    A("")
    A("| predictor | territory | gene | n abnormal | n normal | scanned LR+ | realised specificity | interpolated LR+ |")
    A("|---|---|---|---|---|---|---|---|")
    for m, st in QUOTED:
        for r in genes[(genes["model"] == m) & (genes["stratum"] == st)].itertuples():
            A(f"| {LABEL[m]} | {STRAT[st]} | {r.gene} | {r.n_abnormal} | {r.n_normal} | {_f3(r.lr_scanned)} | "
              f"{_pct(r.specificity_realised)} | {_f3(r.lr_interpolated)} |")
        c = cells[(cells["model"] == m) & (cells["stratum"] == st)].iloc[0]
        A(f"| **{LABEL[m]}** | **{STRAT[st]}** | **median** | | | **{_f3(c.lr_scanned_median)}** | | "
          f"**{_f3(c.lr_interpolated_median)}** |")
    A("")
    A(f"Realised specificity for these per-gene values runs from {_pct(s['quoted_specificity_realised_min'])} to "
      f"{_pct(s['quoted_specificity_realised_max'])}, and interpolation moves a quoted median by at most "
      f"{_f3(s['quoted_max_abs_change'])}.")
    A("")
    A("## Fig. 6B as a whole")
    A("")
    m, st = s["fig6b_max_abs_change_cell"]
    A(f"Fig. 6B plots {s['fig6b_cells']} cells with a scanned value. Interpolation changes "
      f"{s['fig6b_band_changes']} of their evidence bands. The largest change is {_f3(s['fig6b_max_abs_change'])}, for "
      f"{LABEL[m]} in {STRAT[st]}, and realised specificity per gene runs from "
      f"{_pct(s['fig6b_specificity_realised_min'])} to {_pct(s['fig6b_specificity_realised_max'])}. The "
      "bounds quoted for the four ratios above do not extend to the whole panel.")
    A("")
    A("## Every cell")
    A("")
    A("Medians across genes. The realised specificity is the range across the genes with a scanned value.")
    A("")
    A("| predictor | territory | genes | scanned LR+ | realised specificity | interpolated LR+ | band, scanned → interpolated |")
    A("|---|---|---|---|---|---|---|")
    for c in cells.itertuples():
        band = (_band(c.band_scanned) if c.band_scanned == c.band_interpolated
                else f"**{_band(c.band_scanned)} → {_band(c.band_interpolated)}**")
        rng = ("—" if not math.isfinite(c.specificity_realised_min)
               else f"{_pct(c.specificity_realised_min)}–{_pct(c.specificity_realised_max)}")
        A(f"| {LABEL[c.model]} | {STRAT[c.stratum]} | {c.k_genes} | {_f3(c.lr_scanned_median)} | {rng} | "
          f"{_f3(c.lr_interpolated_median)} | {band} |")
    A("")
    A("## Findings")
    A("")
    A(f"1. **Strong band.** No median reaches {STRONG} on either basis. Per gene, "
      + "; ".join(f"{LABEL[r.model]} in {r.gene} ({STRAT[r.stratum]}) is {_f3(r.lr_scanned)} scanned and "
                  f"{_f3(r.lr_interpolated)} interpolated" for r in strong_gene.itertuples())
      + ". These single-gene values reach the threshold, but the evidence band is assigned to the median across "
        "genes.")
    A(f"2. **Realised specificity.** {len(hi)} of {len(g)} per-gene values sit above 95.5%, all in coarsely valued "
      "scores or small strata:")
    A("")
    A("   | predictor | territory | gene | scanned LR+ | realised specificity | interpolated LR+ |")
    A("   |---|---|---|---|---|---|")
    for r in hi.itertuples():
        A(f"   | {LABEL[r.model]} | {STRAT[r.stratum]} | {r.gene} | {_f3(r.lr_scanned)} | "
          f"{_pct(r.specificity_realised)} | {_f3(r.lr_interpolated)} |")
    A("")
    A(f"3. **Interpolation does not always lower the ratio.** It is lower in {len(down)} of {len(fin)} cells, "
      f"unchanged in {len(fin) - len(down) - len(up)}, and higher in "
      + (", ".join(f"{LABEL[r.model]} / {STRAT[r.stratum]} ({_f3(r.lr_scanned_median)} → "
                   f"{_f3(r.lr_interpolated_median)})" for r in up.itertuples()) or "none")
      + ". It is higher whenever the ROC just past the scanned point is steeper than the scanned ratio.")
    A(f"4. **Band changes.** " + ("; ".join(
        f"{LABEL[r.model]} in {STRAT[r.stratum]} goes from {_band(r.band_scanned)} ({_f3(r.lr_scanned_median)}) to "
        f"{_band(r.band_interpolated)} ({_f3(r.lr_interpolated_median)}), with realised specificity "
        f"{_pct(r.specificity_realised_min)}–{_pct(r.specificity_realised_max)}" for r in changed.itertuples()) or "None")
      + ".")
    A(f"5. **Cells without a scanned value.** The {len(ne)} cells where no observed threshold reaches 95% "
      "specificity all receive an interpolated value, read between calling no variant positive and calling the "
      "top tie group positive:")
    A("")
    A("   | predictor | territory | interpolated LR+ | band |")
    A("   |---|---|---|---|")
    for r in ne.itertuples():
        A(f"   | {LABEL[r.model]} | {STRAT[r.stratum]} | {_f3(r.lr_interpolated_median)} | "
          f"{_band(r.band_interpolated)} |")
    A("")
    A("   No single score threshold achieves those operating points, so the atlas keeps these cells as not "
      "evaluable (Note S15).")
    A("6. **Aggregation.** The atlas reports the median of per-gene ratios. The companion reports one ratio per "
      "object on the pooled set, with gene-clustered bootstrap intervals. Both use 95% specificity and the "
      "Tavtigian point system at a prior of 0.10; this comparison changes neither aggregation.")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=str(RESULTS / "score_matrix_atlas_v2.parquet"))
    ap.add_argument("--aux", default=str(RESULTS / "assay_aux_v1.parquet"))
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--page", default=str(PAGE))
    a = ap.parse_args(argv)
    labels = binarise(pd.read_parquet(a.aux))
    cells, genes = compare(pd.read_parquet(a.matrix), labels)
    out = Path(a.out)
    check_reproduces(cells, out / "clinical_evidence_v1.tsv")
    cells.to_csv(out / "lr_basis_check_v1.tsv", sep="\t", index=False)
    genes.to_csv(out / "lr_basis_check_genes_v1.tsv", sep="\t", index=False)
    s = summary(cells, genes)
    (out / "lr_basis_check_v1.json").write_text(json.dumps(s, indent=2) + "\n")
    Path(a.page).parent.mkdir(parents=True, exist_ok=True)
    Path(a.page).write_text(render_page(cells, genes, s))
    print(f"{len(cells)} cells reproduce clinical_evidence_v1.tsv; quoted ratios move by at most "
          f"{s['quoted_max_abs_change']:.3f} (realised specificity {100 * s['quoted_specificity_realised_min']:.2f}–"
          f"{100 * s['quoted_specificity_realised_max']:.2f}%); Fig. 6B band changes {s['fig6b_band_changes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
