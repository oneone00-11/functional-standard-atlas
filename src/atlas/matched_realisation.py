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


def matched_wide(att: pd.DataFrame) -> pd.DataFrame:
    """The matched predictor x stratum block: predictors carrying a realisation
    in both coding/UTR and splice ±1-2, genes pooled by the median."""
    att = att[~att["model"].isin(MISSENSE_ONLY)]
    tab = (att.groupby(["stratum", "model"])[["rho", "realisation"]]
              .median().reset_index())
    wide = tab.pivot(index="model", columns="stratum", values=["rho", "realisation"])
    return wide.dropna(subset=[("realisation", CODING), ("realisation", SPLICE)])


def matched_metrics(have: pd.DataFrame) -> list[dict]:
    """Coding-vs-splice medians and their ratio, observed and corrected."""
    rows = []
    for metric in ("rho", "realisation"):
        c, s = have[(metric, CODING)], have[(metric, SPLICE)]
        rows.append({
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
        })
    return rows


def compare(results: Path) -> tuple[pd.DataFrame, dict]:
    att = pd.read_csv(results / "attenuation_v1.tsv", sep="\t")
    att_f = att[~att["model"].isin(MISSENSE_ONLY)]
    tab = (att_f.groupby(["stratum", "model"])[["rho", "realisation"]]
               .median().reset_index())
    wide = tab.pivot(index="model", columns="stratum", values=["rho", "realisation"])
    have = wide.dropna(subset=[("realisation", CODING), ("realisation", SPLICE)])
    matched = sorted(have.index)
    dropped = sorted(set(wide.dropna(subset=[("realisation", CODING)]).index) - set(matched))

    rows = matched_metrics(have)
    summary = {"matched_models": matched, "n_matched": len(matched),
               "dropped_from_coding": dropped}
    for rec in rows:
        summary[rec["metric"]] = rec

    per_model = have.copy()
    per_model.columns = [f"{a}_{b}" for a, b in per_model.columns]
    return pd.DataFrame(rows), {"summary": summary,
                                "per_model": per_model.reset_index().to_dict("records")}


LOO_GENES = ["BARD1", "BRCA1", "PALB2"]


def loo(results: Path) -> pd.DataFrame:
    """Leave-one-gene-out on the matched coding-vs-splice median ratio.

    The matched comparison rests on three genes' ceilings. With a ratio quoted
    to two figures, how far one gene can move it is a property a reviewer will
    ask after; this tabulates it: the full set, then each gene dropped in turn.
    Dropping BRCA1 reverses the corrected ratio below 1 -- the corrected
    coding/UTR advantage is not robust to which assay supplies the splice
    ceilings, and the table says so in numbers rather than prose.
    """
    att = pd.read_csv(results / "attenuation_v1.tsv", sep="\t")
    scopes = [("full (all 3 genes)", att)]
    scopes += [(f"drop {g}", att[att["gene"] != g]) for g in LOO_GENES]
    rows = []
    for tag, sub in scopes:
        have = matched_wide(sub)
        for rec in matched_metrics(have):
            rows.append({"scope": tag, **{k: v for k, v in rec.items()
                                          if "best" not in k}})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out)

    table, prov = compare(out)
    table.to_csv(out / "matched_realisation_v1.tsv", sep="\t", index=False)
    (out / "matched_realisation_v1.json").write_text(json.dumps(prov, indent=2) + "\n")

    loo_table = loo(out)
    loo_table.to_csv(out / "matched_realisation_loo_v1.tsv", sep="\t", index=False)

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

    print("\nLeave-one-gene-out (matched median ratio):")
    print(loo_table.pivot_table(index="scope", columns="metric", values="median_ratio")
              .to_string(float_format=lambda x: f"{x:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
