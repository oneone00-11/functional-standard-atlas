"""Matched-set comparison of coding/UTR against canonical splice sites (±1–2 bp).

Comparing the two territories predictor-by-predictor is only meaningful on the
subset of predictors that carry a realisation in *both*. Three columns present in
coding/UTR have none at ±1-2 -- AlphaMissense and the two gnomAD allele-frequency
columns -- and the missense-only meta-predictors cannot score a splice variant at
all. Taking each stratum's median over whatever it happens to cover therefore
compares different predictor sets, and in this atlas that unmatched comparison
flatters the ±1-2 result: the two allele-frequency columns drag the coding median
down, so the coding-vs-splice gap looks smaller than it is.

This module recomputes the comparison on the matched set only, reporting both the
raw pooled rho and the attenuation-corrected realisation, and the ratio between
the two territories under each. The manuscript quotes these numbers.

Usage (PYTHONPATH=src):  python -m atlas.matched_realisation --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CODING, SPLICE = "coding_or_utr", "splice_1_2"

# The seven dbNSFP meta-predictors are missense-only by construction. Where they
# carry a canonical-splice row at all it is an artefact of a handful of variants
# that dbNSFP files against an overlapping coding transcript, so they are excluded
# from a comparison whose whole subject is the splice territory.
MISSENSE_ONLY = ["revel", "bayesdel_addaf", "clinpred", "metarnn",
                 "primateai", "vest4", "esm1b"]


def pooled_table(results: Path) -> pd.DataFrame:
    """Pooled rho and realisation per model x stratum, pooled over genes."""
    # every gene in this table already carries exactly one ceiling (reliability
    # prefers the replicate estimate where a gene has both), so no method filter.
    att = pd.read_csv(results / "attenuation_v1.tsv", sep="\t")
    att = att[~att["model"].isin(MISSENSE_ONLY)]
    return (att.groupby(["stratum", "model"])[["rho", "realisation"]]
               .median().reset_index())


def compare(results: Path) -> tuple[pd.DataFrame, dict]:
    tab = pooled_table(results)
    wide = tab.pivot(index="model", columns="stratum", values=["rho", "realisation"])
    have = wide.dropna(subset=[("realisation", CODING), ("realisation", SPLICE)])
    matched = sorted(have.index)
    dropped = sorted(set(wide.dropna(subset=[("realisation", CODING)]).index) - set(matched))

    rows, summary = [], {"matched_models": matched, "n_matched": len(matched),
                         "dropped_from_coding": dropped}
    for metric in ("rho", "realisation"):
        c, s = have[(metric, CODING)], have[(metric, SPLICE)]
        rec = {
            "metric": metric,
            "n_models": len(have),
            "coding_median": float(c.median()),
            "splice_1_2_median": float(s.median()),
            "median_ratio": float(c.median() / s.median()),
            "coding_best": float(c.max()),
            "coding_best_model": str(c.idxmax()),
            "splice_1_2_best": float(s.max()),
            "splice_1_2_best_model": str(s.idxmax()),
            "best_ratio": float(c.max() / s.max()),
        }
        rows.append(rec)
        summary[metric] = rec

    per_model = have.copy()
    per_model.columns = [f"{a}_{b}" for a, b in per_model.columns]
    return pd.DataFrame(rows), {"summary": summary,
                                "per_model": per_model.reset_index().to_dict("records")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out)

    table, prov = compare(out)
    table.to_csv(out / "matched_realisation_v1.tsv", sep="\t", index=False)
    (out / "matched_realisation_v1.json").write_text(json.dumps(prov, indent=2) + "\n")

    s = prov["summary"]
    print(f"matched predictor set ({s['n_matched']}): {', '.join(s['matched_models'])}")
    print(f"dropped (no ±1-2 realisation): {', '.join(s['dropped_from_coding']) or 'none'}\n")
    for metric in ("rho", "realisation"):
        r = s[metric]
        print(f"{metric:<12} coding {r['coding_median']:.3f} vs ±1-2 {r['splice_1_2_median']:.3f} "
              f"(median ratio {r['median_ratio']:.2f}x) | "
              f"best {r['coding_best']:.3f} ({r['coding_best_model']}) vs "
              f"{r['splice_1_2_best']:.3f} ({r['splice_1_2_best_model']}) "
              f"(ratio {r['best_ratio']:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
