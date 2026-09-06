"""Do deposited error columns measure the same thing as replicate disagreement?

The attenuation ceiling is bounded by *reproducibility*: how far a repeat of the
experiment would move the score. A deposited error column need not measure that.
A curve-fitting confidence interval, for instance, measures how well a parameter
is determined given one melting curve, which is a different quantity and
generally a smaller one.

Where a deposit publishes both a replicate set that reconciles with its score and
a per-variant error column, both estimators can be run on the same variants and
compared:

    rel_replicate = spearman_brown(mean pairwise Spearman among the subset)
    rel_error     = 1 - mean(SE^2) / var(score)

A systematically positive ``delta = ceiling_error - ceiling_replicate`` means the
deposited error understates the error that matters, so taking it at face value
overstates the ceiling and understates how much correction is due.

BRCA1 is carried as a calibration point. It deposits no error column, so the
primary analysis derives one as sd(replicates)/sqrt(2); the manuscript reports
that this raises its ceiling by 0.049-0.062. Reproducing that here is a
regression check that this module and the main pipeline agree.

Usage (with PYTHONPATH=src):
    python -m atlas.mavedb_estimator_comparison --out results/mavedb_estimator_comparison_v1.tsv
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from atlas.mavedb_ceilings import MIN_N, SCORE, se_from
from atlas.mavedb_ceilings import reliability_from_replicates
from atlas.mavedb_survey import API, TIMEOUT

BRCA1 = "urn:mavedb:00000097-0-2"
BRCA1_REPS = ["score_rep1", "score_rep2"]


def fetch(urn: str) -> pd.DataFrame:
    r = requests.get(f"{API}/score-sets/{urn}/scores", timeout=TIMEOUT)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), low_memory=False)


def compare(urn: str, reps: list[str], derived_se: bool = False) -> dict:
    row = {"urn": urn, "n_variants": None, "rel_replicate": None, "rel_error": None,
           "ceiling_replicate": None, "ceiling_error": None, "delta": None,
           "error_column": "", "error_rule": "", "note": ""}
    df = fetch(urn)
    reps = [c for c in reps if c in df.columns]
    if len(reps) < 2 or SCORE not in df.columns:
        row["note"] = "missing columns"
        return row
    rel_r, n = reliability_from_replicates(df, reps)
    row["n_variants"], row["rel_replicate"] = n, rel_r

    if derived_se:
        # the primary analysis's construction for a deposit with no error column
        sub = df[reps].apply(pd.to_numeric, errors="coerce")
        se = sub.std(axis=1, ddof=1) / np.sqrt(len(reps))
        col, rule = "sd(replicates)/sqrt(k)", "derived_from_replicates"
    else:
        se, col, rule = se_from(df)
    row["error_column"], row["error_rule"] = col, rule
    if se is None:
        row["note"] = "no error column"
        return row

    score = pd.to_numeric(df[SCORE], errors="coerce")
    x = pd.concat([score.rename(SCORE), pd.Series(se, name="se")], axis=1).dropna()
    if len(x) >= MIN_N and x[SCORE].var() > 0:
        row["rel_error"] = 1 - float((x["se"] ** 2).mean()) / float(x[SCORE].var())
    for k, s in (("ceiling_replicate", "rel_replicate"), ("ceiling_error", "rel_error")):
        v = row[s]
        row[k] = float(np.sqrt(v)) if v is not None and v > 0 else None
    if row["ceiling_error"] is not None and row["ceiling_replicate"] is not None:
        row["delta"] = row["ceiling_error"] - row["ceiling_replicate"]
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", default="results/mavedb_survey_v1.tsv")
    ap.add_argument("--out", default="results/mavedb_estimator_comparison_v1.tsv")
    args = ap.parse_args(argv)

    s = pd.read_csv(args.survey, sep="\t").fillna("")
    both = s[(s.reconciling_replicates != "") & (s.error_columns != "")]
    print(f"deposits publishing both: {len(both)}")

    rows = []
    for _, r in both.iterrows():
        rec = compare(r["urn"], r["reconciling_replicates"].split(";"))
        rec["human"] = bool(r["human"])
        rec["deposited_error"] = True
        rows.append(rec)

    cal = compare(BRCA1, BRCA1_REPS, derived_se=True)
    cal["human"], cal["deposited_error"] = True, False
    cal["note"] = (cal["note"] + " BRCA1 calibration: SE derived from replicates, "
                   "as the primary analysis does").strip()
    rows.append(cal)

    d = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, sep="\t", index=False)
    print(f"\nwrote {out}")

    dd = d[d.delta.notna() & d.deposited_error]
    print(f"\ndeposits with both estimators computable: {len(dd)}")
    if len(dd):
        print(f"  delta (ceiling_error - ceiling_replicate):")
        print(f"    median {dd.delta.median():+.4f}   min {dd.delta.min():+.4f}   max {dd.delta.max():+.4f}")
        print(f"    positive in {int((dd.delta > 0).sum())} of {len(dd)}")
    b = d[d.urn == BRCA1]
    if len(b) and pd.notna(b.delta.iloc[0]):
        v = float(b.delta.iloc[0])
        ok = 0.049 <= v <= 0.062
        print(f"\n  BRCA1 calibration delta = {v:+.4f}  "
              f"({'within' if ok else 'OUTSIDE'} the manuscript's 0.049-0.062)")
    print()
    print(d[["urn", "n_variants", "ceiling_replicate", "ceiling_error", "delta",
             "error_rule"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
