"""Supplemental Table S12: the summary statistics the text quotes but nothing ships.

Seven groups of figures appear in the manuscript with no source inside the
submission package -- a reader has to download the Zenodo TSVs to check them.
That is a weak standard for supplementary material, and it is the same failure
mode as a number with no pipeline source: nothing is wrong, but nothing can be
checked either.

Each row names the value, what it is, and the results/ file it is computed from,
so the table is auditable without leaving the package.

Usage (PYTHONPATH=src):  python -m atlas.build_supplement_summary --write
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = RESULTS / "table_s12_quoted_summary_v1.tsv"


def rows() -> list[dict]:
    out: list[dict] = []

    def add(quantity, value, source, definition):
        out.append({"quantity": quantity, "value": value,
                    "source_file": source, "definition": definition})

    # --- MaveDB ceiling survey (Figure 3, Results, Discussion) --------------
    d = pd.read_csv(RESULTS / "mavedb_ceilings_v1.tsv", sep="\t")
    hd = d[(d.stratum == "human") & (d.status == "ok")].dropna(subset=["ceiling"])
    sub = hd[hd.method != "sd_as_se"]
    src = "mavedb_ceilings_v1.tsv"
    add("human deposits with a computable ceiling", f"{len(hd)}", src,
        "stratum == human, status == ok, ceiling present")
    add("median ceiling, human deposits", f"{hd.ceiling.median():.3f}", src, "median")
    add("human deposits below a ceiling of 0.90", f"{100 * (hd.ceiling < 0.90).mean():.1f}%",
        src, "share of the same set")
    add("deposits with an explicit SE, reconcilable replicates or a CI",
        f"{len(sub)}", src, "method != sd_as_se")
    add("that subset below a ceiling of 0.90", f"{100 * (sub.ceiling < 0.90).mean():.1f}%",
        src, "share of the subset")
    add("human deposits below the 0.45 validity bound",
        f"{100 * (hd.ceiling < 0.45).mean():.1f}%", src,
        "share below the bound the simulation sets for the correction")

    # --- what the survey found at all ---------------------------------------
    s = pd.read_csv(RESULTS / "mavedb_survey_v1.tsv", sep="\t")
    add("MaveDB score sets surveyed", f"{len(s)}", "mavedb_survey_v1.tsv", "all rows")
    # The manuscript's 2,452 counts deposits whose error model was verified.
    # permits != none also admits the 29 whose replicate columns reconcile with
    # no subset of themselves, which the text counts separately.
    add("score sets carrying a verified error model",
        f"{int(s.permits.isin(['errors', 'replicates']).sum())}",
        "mavedb_survey_v1.tsv", "permits in (errors, replicates)")
    add("score sets whose replicate columns do not reconcile",
        f"{int((s.permits == 'replicates_unverified').sum())}",
        "mavedb_survey_v1.tsv", "permits == replicates_unverified")

    # --- within-assay percentile of canonical splice variants ---------------
    p = pd.read_csv(RESULTS / "saturation_percentiles_v1.tsv", sep="\t")
    r = p[p.region == "splice_1_2"].iloc[0]
    add("median within-assay percentile, splice +-1-2", f"{r.median_percentile:.3f}",
        "saturation_percentiles_v1.tsv", "median over the stratum")
    add("splice +-1-2 variants above the 80th percentile",
        f"{100 * r.frac_above_p80:.1f}%", "saturation_percentiles_v1.tsv",
        "share of the stratum")

    # --- matched coding-versus-splice comparison ----------------------------
    m = pd.read_csv(RESULTS / "matched_realisation_v1.tsv", sep="\t").set_index("metric")
    for metric, label in (("rho", "observed"), ("realisation", "corrected")):
        q = m.loc[metric]
        add(f"matched set, best coding ({label})", f"{q.coding_best:.3f}",
            "matched_realisation_v1.tsv", f"best of {int(q.n_models)} models, coding/UTR")
        add(f"matched set, best splice +-1-2 ({label})", f"{q.splice_1_2_best:.3f}",
            "matched_realisation_v1.tsv", f"best of {int(q.n_models)} models, splice +-1-2")
        add(f"matched set, median ratio ({label})", f"{q.median_ratio:.3f}",
            "matched_realisation_v1.tsv", "coding median / splice median")
    return out


def build() -> Path:
    df = pd.DataFrame(rows())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t", index=False)
    return OUT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    df = pd.DataFrame(rows())
    print(df.to_string(index=False))
    if a.write:
        print(f"\n[s12] wrote {build()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
