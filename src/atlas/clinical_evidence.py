"""Territory-resolved classification performance and evidence strength.

Spearman ρ is the right metric for "does this predictor track function", but it
is not the quantity a diagnostic laboratory acts on. Three of the seven deposits
publish the assay authors' own functional calls — a hard four-class call for
RAD51C, a two-component posterior for BARD1 and PALB2 — which lets the same
territory grid be scored as a classification problem and, more usefully,
expressed as a likelihood ratio.

Metrics
-------
``AUROC``/``AUPRC``  per gene, pooled across genes on the logit scale with
                     DerSimonian–Laird and Hanley–McNeil variances.
``LR+``              positive likelihood ratio at a fixed high specificity
                     (90% and 95%), i.e. how much a high score from this
                     predictor, in this territory, should move a prior.

The LR+ figures are indicative, not a ClinGen-endorsed calibration: they are
computed against assay-derived labels on seven genes, with no prior-probability
model and no ClinGen-specified bootstrap procedure. They are reported so that
territory-resolved performance can be read on a scale clinicians use, and the
ACMG tier column should be read as "consistent with", not "certified as".

Usage (PYTHONPATH=src):  python -m atlas.clinical_evidence --out results/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from atlas.evaluate import classify_region, dl_pool

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

MODELS = [
    "alphagenome", "spliceai_ds", "pangolin_score", "cadd", "alphamissense",
    "evo2", "gpn_msa", "nucleotide_transformer", "phylop100way",
    "phastcons100way", "gnomad_af_global", "gnomad_af_popmax",
    # dbNSFP meta-predictors (missense-only; atlas.metapredictors)
    "revel", "bayesdel_addaf", "clinpred", "metarnn", "primateai", "vest4", "esm1b",
]

STRATA = ["all", "coding_or_utr", "splice_region",
          "splice_1_2", "splice_3_10", "splice_11_50", "splice_deep"]

# Tavtigian point system, prior 0.10: LR+ thresholds for pathogenic evidence tiers
ACMG_TIERS = [(350.0, "Very strong"), (18.7, "Strong"), (4.33, "Moderate"),
              (2.08, "Supporting")]

MIN_POS = 10
MIN_NEG = 10


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC via the Mann–Whitney U identity, with mid-ranks for ties.

    Implemented here rather than pulled from scikit-learn so that the repo's
    pinned environment stays minimal; ``tests/test_clinical_evidence.py`` checks
    it against a brute-force pair count.
    """
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = stats.rankdata(s, method="average")
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision: sum of precision at each distinct threshold, weighted
    by the recall it gains (the step-wise definition, ties grouped)."""
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    # last index of each run of equal scores
    cuts = np.flatnonzero(np.diff(ss)) if len(ss) > 1 else np.array([], dtype=int)
    cuts = np.append(cuts, len(ss) - 1)
    tp = np.cumsum(ys == 1)[cuts]
    fp = np.cumsum(ys == 0)[cuts]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def acmg_tier(lr: float) -> str:
    if lr is None or not np.isfinite(lr):
        # no threshold reaches the target specificity — the score is too coarsely
        # quantised to operate at that point, which is not the same as being weak
        return "not evaluable"
    for thresh, name in ACMG_TIERS:
        if lr >= thresh:
            return name
    return "below supporting"


def hanley_mcneil_var(auc: float, n_pos: int, n_neg: int) -> float:
    """Variance of an AUC estimate (Hanley & McNeil 1982, exponential approximation)."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    return (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
            + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def lr_at_specificity(y: np.ndarray, s: np.ndarray, spec: float) -> tuple[float, float]:
    """(LR+, threshold) at the most sensitive threshold meeting the target specificity.

    Thresholds are searched over the observed score values rather than taken as a
    quantile of the negatives. Several of these predictors emit heavily quantised
    scores, and a quantile threshold can land exactly on a large tie group: every
    negative in that group then satisfies ``score >= threshold``, the false-positive
    rate is driven to 1 and the likelihood ratio collapses to 1 even for a
    perfectly separating score. Scanning candidate thresholds avoids that.
    """
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    target_fpr = 1.0 - spec
    # ascending scan: the first threshold meeting the FPR budget is the most
    # sensitive one that does
    for thr in np.unique(s):
        fpr = float((neg >= thr).mean())
        if fpr <= target_fpr:
            tpr = float((pos >= thr).mean())
            # a zero observed FPR is bounded away from zero rather than reported
            # as an infinite likelihood ratio
            fpr = max(fpr, 1.0 / (len(neg) + 1))
            return tpr / fpr, float(thr)
    return float("nan"), float("nan")  # quantisation prevents reaching this specificity


def binarise(aux: pd.DataFrame, hi: float = 0.9, lo: float = 0.1) -> pd.DataFrame:
    """Hard labels: keep confident calls only, drop the ambiguous middle."""
    a = aux.dropna(subset=["label_abnormal"]).copy()
    hard = a["label_source"].str.startswith("class:")
    lab = pd.Series(np.nan, index=a.index, dtype=float)
    lab[hard] = a.loc[hard, "label_abnormal"]
    soft = ~hard
    lab[soft & (a["label_abnormal"] >= hi)] = 1.0
    lab[soft & (a["label_abnormal"] <= lo)] = 0.0
    a["y"] = lab
    return a.dropna(subset=["y"])[["variant_id", "gene", "y", "label_source"]]


def evaluate(mat: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = mat.merge(labels, on=["variant_id", "gene"], how="inner")
    df["region"] = df["hgvs_c"].map(classify_region)
    rows = []
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
            per_gene = []
            for gene, g in sub.groupby("gene"):
                y = g["y"].to_numpy()
                s = g[model].to_numpy()
                n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
                if n_pos < MIN_POS or n_neg < MIN_NEG:
                    continue
                auc = roc_auc(y, s)
                ap = average_precision(y, s)
                lr95, thr95 = lr_at_specificity(y, s, 0.95)
                lr90, _ = lr_at_specificity(y, s, 0.90)
                per_gene.append({"gene": gene, "n_pos": n_pos, "n_neg": n_neg,
                                 "auroc": auc, "auprc": ap, "lr95": lr95, "lr90": lr90,
                                 "var": hanley_mcneil_var(auc, n_pos, n_neg)})
            if len(per_gene) < 2:
                continue
            pg = pd.DataFrame(per_gene)
            # pool AUROC on the logit scale (delta-method variance)
            z = pg["auroc"].map(_logit)
            vz = pg["var"] / (pg["auroc"] * (1 - pg["auroc"])) ** 2
            p = dl_pool(z.tolist(), vz.tolist())
            inv = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
            rows.append({
                "model": model, "stratum": stratum, "k_genes": len(pg),
                "n_pos": int(pg["n_pos"].sum()), "n_neg": int(pg["n_neg"].sum()),
                "auroc": inv(p["pooled"]),
                "auroc_lo": inv(p["ci_lo"]), "auroc_hi": inv(p["ci_hi"]),
                "i2_pct": p["i2_pct"],
                "auprc_median": float(pg["auprc"].median()),
                "prevalence": float(pg["n_pos"].sum() / (pg["n_pos"].sum() + pg["n_neg"].sum())),
                "lr_at_spec95_median": float(pg["lr95"].median()),
                "lr_at_spec90_median": float(pg["lr90"].median()),
                "acmg_tier_spec95": acmg_tier(float(pg["lr95"].median())),
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # v2 carries the seven dbNSFP meta-predictors; v1 does not. The default was
    # v1, so a default run silently omitted those models from the delivered
    # table. v1 is kept only as the auditable pre-rescore matrix
    # (see atlas.matrix_v2).
    ap.add_argument("--matrix", default=str(RESULTS / "score_matrix_atlas_v2.parquet"))
    ap.add_argument("--aux", default=str(RESULTS / "assay_aux_v1.parquet"))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    mat = pd.read_parquet(args.matrix)
    # A declared model absent from the matrix used to be skipped in silence.
    absent = [m for m in MODELS if m not in mat.columns]
    if absent:
        print(f"[clinical_evidence] WARNING: {len(absent)} declared models are not "
              f"in this matrix and are omitted: {', '.join(absent)}", file=sys.stderr)
    aux = pd.read_parquet(args.aux)
    labels = binarise(aux)
    res = evaluate(mat, labels)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "clinical_evidence_v1.tsv", sep="\t", index=False)

    summary = {
        "n_labelled": int(len(labels)),
        "genes": sorted(labels["gene"].unique().tolist()),
        "by_gene": labels.groupby("gene")["y"].agg(["size", "sum"]).to_dict("index"),
        "label_sources": labels["label_source"].value_counts().to_dict(),
    }
    (out / "clinical_evidence_v1.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    print(f"labelled variants: {summary['n_labelled']:,} across "
          f"{len(summary['genes'])} genes ({', '.join(summary['genes'])})")
    for g, d in summary["by_gene"].items():
        print(f"  {g:<8} n={int(d['size']):>6,}  abnormal={int(d['sum']):>6,} "
              f"({d['sum']/d['size']:.1%})")

    for stratum in ["all", "coding_or_utr", "splice_region", "splice_3_10"]:
        s = res[res["stratum"] == stratum].sort_values("auroc", ascending=False)
        if s.empty:
            continue
        # label counts are per model: predictors differ in which variants they score
        print(f"\n=== {stratum} ===")
        print(s[["model", "k_genes", "n_pos", "n_neg", "auroc", "auroc_lo", "auroc_hi",
                 "auprc_median", "lr_at_spec95_median", "acmg_tier_spec95"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
