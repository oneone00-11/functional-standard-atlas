"""Estimator consistency: what changes when BRCA1 is scored by the SE estimator.

The primary ceiling prefers the replicate-based estimator wherever a gene
carries both (``atlas.reliability``): the two estimators are on different
scales, and the replicate chain is the one whose construction is verified
against the deposit. BRCA1 is the one gene carrying both, and its SE chain —
sd of the two fitness replicates over sqrt(2), derived in ``atlas.assay_aux``
— systematically reads *higher* reliability than the replicate chain. A reader
is entitled to ask how much of the paper's corrected coding-versus-splice
contrast depends on that choice.

This module recomputes the two manuscript quantities with BRCA1 switched to
its SE-derived ceilings, everything else untouched:

* the matched coding/UTR vs splice ±1-2 median realisation ratio
  (``atlas.matched_realisation``): 1.43 published -> 1.45 under the swap;
* the median ceiling per stratum: ±1-2 moves 0.772 -> 0.794.

It also tabulates the per-stratum cross-check itself — ceiling under each
estimator and their difference (0.049 to 0.062 across strata; the manuscript
quotes this range) — so the sensitivity and the disagreement it comes from sit
in one place.

Outputs (PYTHONPATH=src):  python -m atlas.estimator_consistency --out results/
    results/estimator_consistency_v1.tsv   per-stratum cross-check table
    results/estimator_consistency_v1.json  swap summary (ratios and medians)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from atlas.matched_realisation import matched_metrics, matched_wide

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

STRATA_ORDER = ["all", "coding_or_utr", "splice_region",
                "splice_1_2", "splice_3_10", "splice_11_50", "splice_deep"]


def cross_check(rel: pd.DataFrame, gene: str = "BRCA1") -> pd.DataFrame:
    """Ceiling under both estimators, per stratum, for the gene carrying both."""
    g = rel[rel["gene"] == gene]
    wide = g.pivot_table(index="stratum", columns="method",
                         values=["reliability", "ceiling", "n"], aggfunc="first")
    out = pd.DataFrame({
        "gene": gene,
        "stratum": wide.index,
        "n": wide[("n", "replicates")].astype("Int64").fillna(
            wide[("n", "se")].astype("Int64")),
        "reliability_replicates": wide[("reliability", "replicates")],
        "reliability_se": wide[("reliability", "se")],
        "ceiling_replicates": wide[("ceiling", "replicates")],
        "ceiling_se": wide[("ceiling", "se")],
    }).dropna(subset=["ceiling_replicates", "ceiling_se"])
    out["delta_se_minus_replicates"] = out["ceiling_se"] - out["ceiling_replicates"]
    out["stratum"] = pd.Categorical(out["stratum"], STRATA_ORDER, ordered=True)
    return (out.reset_index(drop=True).sort_values("stratum").reset_index(drop=True))


def swap_to_se(att: pd.DataFrame, ceilings_se: pd.Series, gene: str = "BRCA1") -> pd.DataFrame:
    """The attenuation table with one gene's ceilings replaced by its SE ones."""
    out = att.copy()
    m = out["gene"] == gene
    out.loc[m, "ceiling"] = out.loc[m, "stratum"].map(ceilings_se)
    out.loc[m, "realisation"] = out.loc[m, "rho"] / out.loc[m, "ceiling"]
    out.loc[m, "method"] = "se"
    return out


def ceiling_medians(att: pd.DataFrame) -> pd.Series:
    return (att[["gene", "stratum", "ceiling"]].drop_duplicates()
            .groupby("stratum")["ceiling"].median())


def run(results: Path) -> tuple[pd.DataFrame, dict]:
    rel = pd.read_csv(results / "reliability_v1.tsv", sep="\t")
    att = pd.read_csv(results / "attenuation_v1.tsv", sep="\t")

    cross = cross_check(rel)
    ceilings_se = cross.set_index("stratum")["ceiling_se"]
    att_se = swap_to_se(att, ceilings_se)

    ratios = {}
    for tag, frame in (("published_brca1_replicates", att),
                       ("brca1_swapped_to_se", att_se)):
        rec = next(r for r in matched_metrics(matched_wide(frame))
                   if r["metric"] == "realisation")
        ratios[tag] = rec

    med = pd.DataFrame({"published_brca1_replicates": ceiling_medians(att),
                        "brca1_swapped_to_se": ceiling_medians(att_se)})

    deltas = cross["delta_se_minus_replicates"]
    summary = {
        "cross_check": {
            "gene": "BRCA1",
            "delta_range": [float(deltas.min()), float(deltas.max())],
            "delta_by_stratum": {str(r["stratum"]): float(r["delta_se_minus_replicates"])
                                 for _, r in cross.iterrows()},
            "note": ("ceiling_se - ceiling_replicates; positive everywhere, so the "
                     "SE-derived ceiling overstates reproducibility relative to the "
                     "replicate chain (manuscript range 0.049-0.062 across the "
                     "strata quoted there)"),
        },
        "matched_realisation_ratio": {
            "published_brca1_replicates": ratios["published_brca1_replicates"]["median_ratio"],
            "brca1_swapped_to_se": ratios["brca1_swapped_to_se"]["median_ratio"],
            "n_matched_models": ratios["brca1_swapped_to_se"]["n_models"],
        },
        "ceiling_medians_by_stratum": {
            str(s): {"published_brca1_replicates": float(r["published_brca1_replicates"]),
                     "brca1_swapped_to_se": float(r["brca1_swapped_to_se"])}
            for s, r in med.iterrows()},
    }
    return cross, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out)

    cross, summary = run(out)
    cross.to_csv(out / "estimator_consistency_v1.tsv", sep="\t", index=False)
    (out / "estimator_consistency_v1.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    print("BRCA1 cross-check (ceiling, SE estimator minus replicate estimator):")
    print(cross[["stratum", "n", "ceiling_replicates", "ceiling_se",
                 "delta_se_minus_replicates"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    r = summary["matched_realisation_ratio"]
    print(f"\nmatched corrected median ratio: "
          f"{r['published_brca1_replicates']:.2f} -> {r['brca1_swapped_to_se']:.2f} "
          f"({r['n_matched_models']} matched predictors)")
    print("\nstratum median ceilings, published -> BRCA1=SE:")
    for s, d in summary["ceiling_medians_by_stratum"].items():
        print(f"  {s:<15s} {d['published_brca1_replicates']:.3f} -> "
              f"{d['brca1_swapped_to_se']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
