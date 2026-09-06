"""MaveDB human-deposit accounting, k-sensitivity, and audit status.

Three reviewer-facing supplements to the whole-resource survey, all computed
from the delivered survey/ceiling/audit tables — no new API calls:

``accounting``   The deposit ledger under every counting definition the paper
                 and its critics use, side by side, because the headline
                 fraction changes with the denominator: all published score
                 sets (2,803), the human subset (1,202), human deposits whose
                 table permits a reliability estimate (991, 82.4% of human),
                 the paper's counting class — human, non-designed-stability
                 deposits entering the ceiling computation (728) — and the
                 ceilings actually computed (674, 92.6% of 728, 24.0% of every
                 published score set).
``k_sensitivity`` Ceiling CDFs when a deposited SD is read as the SE of a
                 k-replicate mean (SE = SD/sqrt(k)) for k = 1, 2, 3. Only the
                 606 ``sd_as_se`` rows move. The below-0.45 share of that
                 subset runs 3.9% -> 0.2% -> 0.5%; of the whole distribution
                 (designed-stability platform excluded, as in the paper)
                 3.3% -> 0.4% -> 0.7%. The 0.45 conclusion is therefore not
                 an artefact of the k = 1 reading, which is the reading that
                 stacks the deck against it.
``audit_status`` The two-direction audit's current state: the 131 sampled
                 deposits (60 classified none, 25 errors, 18 replicates, 28
                 replicates_unverified), each with its audit direction and an
                 adjudication field. No row has been manually adjudicated; the
                 25 errors rows are the false-positive direction and are
                 marked pending manual adjudication. This module deliberately
                 does not adjudicate them.

Usage (PYTHONPATH=src):  python -m atlas.mavedb_accounting --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

CEILING_BOUND = 0.45
K_GRID = (1, 2, 3)


def accounting(sv: pd.DataFrame, ce: pd.DataFrame) -> pd.DataFrame:
    """The deposit ledger under each counting definition."""
    n_all = len(sv)
    human = sv["human"].astype(bool)
    permit = sv["permits"].isin(["replicates", "errors"])
    n_human = int(human.sum())
    n_human_permit = int((permit & human).sum())

    # the paper's counting class: human deposits entering the ceiling
    # computation, i.e. the human stratum of the candidates table (the
    # designed-stability platform is its own stratum and is never pooled)
    paper = ce[ce["stratum"] == "human"]
    n_paper = len(paper)
    n_paper_ok = int((paper["status"] == "ok").sum())

    rows = [
        ("published score sets", n_all, None, None,
         "every published MaveDB score set enumerated through the API"),
        ("human score sets", n_human, n_all, n_human / n_all,
         "target gene recorded as Homo sapiens"),
        ("human permitting an estimate", n_human_permit, n_human,
         n_human_permit / n_human,
         "human score sets whose table carries reconcilable replicates or a "
         "per-variant error column (survey classification)"),
        ("human permitting / all published", n_human_permit, n_all,
         n_human_permit / n_all, "same numerator, all-published denominator"),
        ("paper counting class: human ceiling candidates", n_paper, n_human,
         n_paper / n_human,
         "human, non-designed-stability deposits entering the ceiling "
         "computation (stratum == 'human' in the ceilings table)"),
        ("ceilings computed (paper class)", n_paper_ok, n_paper, n_paper_ok / n_paper,
         "status ok among the paper counting class"),
        ("ceilings computed / all published", n_paper_ok, n_all, n_paper_ok / n_all,
         "the 24.0% figure: usable ceilings over every published score set"),
        ("ceiling candidates, all deposits", len(ce), None, None,
         "every surveyed deposit that permits an estimate, all organisms and "
         "the designed-stability platform included"),
        ("ceilings computed, all deposits", int((ce["status"] == "ok").sum()),
         len(ce), float((ce["status"] == "ok").mean()),
         "status ok among all candidates"),
    ]
    return pd.DataFrame(rows, columns=["metric", "count", "denominator",
                                       "fraction", "definition"])


def k_sensitivity(ce: pd.DataFrame) -> pd.DataFrame:
    """Ceiling CDF under SE = SD/sqrt(k), k = 1, 2, 3.

    Only rows whose error rule was sd_as_se change; every other rule already
    yields an SE-scale column. Two views: the sd_as_se subset alone, and the
    whole ceiling distribution with the designed-stability platform excluded
    (the paper's reporting convention).
    """
    sd = ce[ce["method"] == "sd_as_se"]
    rows = []
    for k in K_GRID:
        rel_k = 1 - sd["mean_se2"] / (k * sd["var_score"])
        usable = (rel_k > 0) & (rel_k < 0.99)
        ceil_k = np.sqrt(rel_k[usable].clip(lower=0))
        rows.append({
            "view": "sd_as_se subset", "k": k,
            "n_rows": len(sd), "n_usable": int(usable.sum()),
            "median_ceiling": float(ceil_k.median()),
            "below_0.30": float((ceil_k < 0.30).mean()),
            "below_0.45": float((ceil_k < CEILING_BOUND).mean()),
            "below_0.60": float((ceil_k < 0.60).mean()),
            "n_below_0.45": int((ceil_k < CEILING_BOUND).sum()),
        })
        if k == 1:
            adj = ce.copy()      # k = 1 is the published reading, unchanged
        else:
            adj = ce.copy()
            m = adj["method"] == "sd_as_se"
            rel_adj = 1 - adj.loc[m, "mean_se2"] / (k * adj.loc[m, "var_score"])
            adj.loc[m, "ceiling"] = np.sqrt(rel_adj.clip(lower=0))
            adj.loc[m, "status"] = np.where(
                (rel_adj > 0) & (rel_adj < 0.99), "ok",
                np.where(rel_adj <= 0, "degenerate", adj.loc[m, "status"]))
        ok = adj[(adj["status"] == "ok") & (adj["platform"] != "designed_stability")]
        rows.append({
            "view": "whole distribution (excl designed_stability)", "k": k,
            "n_rows": len(ok), "n_usable": len(ok),
            "median_ceiling": float(ok["ceiling"].median()),
            "below_0.30": float((ok["ceiling"] < 0.30).mean()),
            "below_0.45": float((ok["ceiling"] < CEILING_BOUND).mean()),
            "below_0.60": float((ok["ceiling"] < 0.60).mean()),
            "n_below_0.45": int((ok["ceiling"] < CEILING_BOUND).sum()),
        })
    return pd.DataFrame(rows)


AUDIT_DIRECTION = {
    "none": "false-negative check (survey found nothing; header inspected for a miss)",
    "errors": "false-positive check (survey accepted an error-column name)",
    "replicates": "enumerated in full (all reconciling-replicate deposits)",
    "replicates_unverified": "enumerated in full (replicate columns that do not reconcile)",
}


def audit_status(au: pd.DataFrame) -> pd.DataFrame:
    """The 131-row audit sample with direction and adjudication state."""
    out = au.copy()
    out["direction"] = out["permits"].map(AUDIT_DIRECTION)
    # No sampled row has been manually adjudicated. The false-positive
    # direction (an error-column name the survey accepted) is the one that
    # could inflate the headline figure, so those rows are flagged explicitly;
    # this module records the state, it does not adjudicate.
    out["adjudication"] = "pending manual adjudication"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out)

    sv = pd.read_csv(out / "mavedb_survey_v1.tsv", sep="\t")
    ce = pd.read_csv(out / "mavedb_ceilings_v1.tsv", sep="\t")
    au = pd.read_csv(out / "mavedb_survey_audit_v1.tsv", sep="\t")

    ledger = accounting(sv, ce)
    ledger.to_csv(out / "mavedb_accounting_v1.tsv", sep="\t", index=False)

    ks = k_sensitivity(ce)
    ks.to_csv(out / "mavedb_ceiling_k_sensitivity_v1.tsv", sep="\t", index=False)

    status = audit_status(au)
    status.to_csv(out / "mavedb_survey_audit_status_v1.tsv", sep="\t", index=False)

    counts = status["permits"].value_counts().to_dict()
    doc = {
        "audit_total": len(status),
        "by_permits": {k: int(v) for k, v in counts.items()},
        "adjudication": ("no sampled row has been manually adjudicated; the 25 "
                         "'errors' rows are the false-positive direction and are "
                         "marked pending manual adjudication"),
        "false_positive_rows": int((status["permits"] == "errors").sum()),
    }
    (out / "mavedb_survey_audit_status_v1.json").write_text(json.dumps(doc, indent=2) + "\n")

    print("MaveDB deposit ledger:")
    print(ledger.to_string(index=False,
                           float_format=lambda x: f"{x:.3f}" if x == x else ""))
    print("\nk-sensitivity (SE = SD/sqrt(k)):")
    print(ks.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\naudit status: {len(status)} rows, classes "
          f"{ {k: int(v) for k, v in sorted(counts.items())} }, "
          f"all pending manual adjudication")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
