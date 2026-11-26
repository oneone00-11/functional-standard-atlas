"""The error models of the rank-scale ceiling (A4b.2), for the nine genes with error
information, and the self-test's real cells under the neutral model.

Two models per gene, both reported:

``own``      the deposit's own error. The SE genes (BARD1, PALB2, BAP1 and the external
             five): Gaussian, scale the variant's SE. BRCA1, which has no SE of its own:
             the empirical distribution of half the difference of its two replicates,
             d = (rep1 - rep2) / 2, within the cell (gene x stratum x population; in the
             matched ceilings the predictor's own variants), with s = 1. The score is the
             replicates' mean (checked row by row), and the sum and the difference of two
             independent errors have the same distribution, so d is distributed as the
             mean's error. Within the cell rather than across the gene: d's spread
             differs between strata (+-1-2 bp against the rest).
``neutral``  the spread of the gene's neutral variants: SNVs that are synonymous and
             whose spliceai_ds is below 0.1, with error information; residuals
             z = (y - the batch's median) / s, s the SE (BRCA1 s = 1), their empirical
             distribution. It also takes in the noise an SE does not capture (variation
             the two replicates share, the tails); were there real effects among the
             synonymous variants, it would be conservative.

The neutral variants: the core four from the score matrix (``consequence`` and
``spliceai_ds``; their numbers must be the second batch of
results/reliability_neutral_check_v1.tsv: BRCA1 468, BARD1 1,231, PALB2 1,807, BAP1
1,322); the external five from results/external_coding_consequence_v1.parquet
(atlas.external_coding), by the same rule. y is functional_pathogenicity (-score for the
external deposits, as everywhere in the atlas): the residuals are on the scale the
ceilings use.

Outputs: results/error_models_v1.tsv, one row per gene x model (BRCA1's own per stratum
and population, and gene-wide): the residuals' number, centre (their median), sd, robust
sd (1.4826 MAD), kurtosis, the shares with |z| above 2 and 3, the bandwidth; a Gaussian
row gives only the number of variants with an SE and sd 1. results/error_models_v1.json:
the definitions and the checks. results/rank_ceiling_selftest_v2.tsv and .json: the
self-test of atlas.rank_ceiling with the neutral model's 18 real cells added (the first
version's rows are computed again and must be the same bits).

Usage (PYTHONPATH=src):  python -m atlas.error_models [--workers N]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import paths, population
from atlas import rank_ceiling as rc
from atlas.reliability import STRATA, _stratum_mask

CORE_SE = ("BARD1", "PALB2", "BAP1")
EXTERNAL = ("CTCF", "SFPQ", "RAD51D", "XRCC2", "SBDS")
GENES = ("BRCA1",) + CORE_SE + EXTERNAL
MODELS = ("own", "neutral")
SPLICEAI_NEUTRAL = 0.1
NEUTRAL_EXPECTED = {"BRCA1": 468, "BARD1": 1231, "PALB2": 1807, "BAP1": 1322}
MEAN_ATOL = 1e-12          # BRCA1's score against its replicates' mean
MIN_N = 30
SELFTEST_V1 = paths.RESULTS / "rank_ceiling_selftest_v1.json"     # atlas.rank_ceiling's

COLUMNS = ["gene", "model", "kind", "population", "stratum", "n", "centre", "sd",
           "robust_sd", "kurtosis", "share_abs_gt_2", "share_abs_gt_3", "bandwidth",
           "score_median", "score_sd", "n_synonymous_with_error"]


# ==========================================================================
# the variants
# ==========================================================================
@lru_cache(maxsize=1)
def core_frame() -> pd.DataFrame:
    """The core four genes' variants (both populations) with their error information."""
    m = population.load(None)
    m = m[m["gene"].isin(("BRCA1",) + CORE_SE)]
    aux = pd.read_parquet(paths.path("assay_aux"),
                          columns=["variant_id", "gene", "fit_rep1", "fit_rep2", "se",
                                   "se_status"])
    df = m.merge(aux, on=["variant_id", "gene"], how="left")
    rep = df["fit_rep1"].notna() & df["fit_rep2"].notna()
    se = df["se"].notna() & df["se_status"].eq("validated")
    df["has_error"] = np.where(df["gene"].eq("BRCA1"), rep, se)
    df["s"] = np.where(df["gene"].eq("BRCA1"), 1.0, df["se"])
    df["d"] = np.where(df["gene"].eq("BRCA1"),
                       ((-df["fit_rep1"]) - (-df["fit_rep2"])) / 2, np.nan)
    return df.reset_index(drop=True)


def half_difference(fit_rep1, fit_rep2) -> np.ndarray:
    """BRCA1's d on the functional_pathogenicity scale (-score)."""
    return ((-np.asarray(fit_rep1, float)) - (-np.asarray(fit_rep2, float))) / 2


def brca1_mean_check() -> dict:
    """The score is the mean of the two replicates, row by row."""
    d = core_frame()
    b = d[d["gene"].eq("BRCA1") & d["has_error"]]
    diff = np.abs(-(b["fit_rep1"] + b["fit_rep2"]) / 2 - b["functional_pathogenicity"])
    return {"rows": int(len(b)), "max_abs_diff": float(diff.max()),
            "holds": bool(diff.max() <= MEAN_ATOL)}


@lru_cache(maxsize=1)
def external_coding() -> pd.DataFrame:
    return pd.read_parquet(paths.path("external_coding_consequence"))


def neutral_variants(gene: str) -> pd.DataFrame:
    """The gene's neutral SNVs with error information: variant identifier, y, s, z."""
    if gene in EXTERNAL:
        x = external_coding()
        g = x[x["gene"].eq(gene) & x["consequence"].eq("synonymous")
              & x["standard_error"].notna() & x["functional_pathogenicity"].notna()]
        syn = len(g)
        g = g[g["spliceai_ds"] < SPLICEAI_NEUTRAL]
        out = pd.DataFrame({"id": g["variant_urn"], "y": g["functional_pathogenicity"],
                            "s": g["standard_error"]})
    else:
        d = core_frame()
        g = d[d["gene"].eq(gene) & d["population"].eq(population.SNV)
              & d["consequence"].eq("synonymous") & d["has_error"]
              & d["functional_pathogenicity"].notna()]
        syn = len(g)
        g = g[g["spliceai_ds"] < SPLICEAI_NEUTRAL]
        out = pd.DataFrame({"id": g["variant_id"], "y": g["functional_pathogenicity"],
                            "s": g["s"]})
    out = out.sort_values("id", kind="stable").reset_index(drop=True)
    out["z"] = (out["y"] - out["y"].median()) / out["s"]
    out.attrs["n_synonymous_with_error"] = syn
    return out


@lru_cache(maxsize=None)
def neutral_model(gene: str) -> rc.ErrorModel:
    return rc.ErrorModel(neutral_variants(gene)["z"].to_numpy(float), name="neutral")


def neutral_check() -> dict:
    got = {g: int(len(neutral_variants(g))) for g in NEUTRAL_EXPECTED}
    return {"counts": got, "expected": NEUTRAL_EXPECTED, "holds": got == NEUTRAL_EXPECTED}


# ==========================================================================
# the summary table
# ==========================================================================
def _row(gene, model, kind, pop, stratum, stats, **extra) -> dict:
    return {"gene": gene, "model": model, "kind": kind, "population": pop, "stratum": stratum,
            **stats, **extra}


def summary_table() -> pd.DataFrame:
    rows = []
    core = core_frame()
    for gene in GENES:
        if gene == "BRCA1":
            b = core[core["gene"].eq("BRCA1") & core["has_error"]]
            whole = rc.ErrorModel(b["d"].to_numpy(float)).describe()
            rows.append(_row(gene, "own", "empirical: replicate half-differences", "all",
                             "gene", whole))
            for pop in population.POPULATIONS:
                bp = b[b["population"].eq(pop)]
                for stratum in STRATA:
                    c = bp[_stratum_mask(bp["region"], stratum)]
                    if len(c) < MIN_N:
                        continue
                    rows.append(_row(gene, "own", "empirical: replicate half-differences "
                                     "within the cell", pop, stratum,
                                     rc.ErrorModel(c["d"].to_numpy(float)).describe()))
        else:
            if gene in EXTERNAL:
                n_se = None            # the deposit's rows with an SE (deposit_rows_with_se)
            else:
                n_se = int(core[core["gene"].eq(gene) & core["has_error"]].shape[0])
            rows.append(_row(gene, "own", "gaussian", "all", "gene",
                             rc.ErrorModel().describe() | {"n": n_se}))
        nv = neutral_variants(gene)
        rows.append(_row(gene, "neutral", "empirical: neutral SNVs", population.SNV, "gene",
                         neutral_model(gene).describe(),
                         score_median=float(nv["y"].median()), score_sd=float(nv["y"].std(ddof=0)),
                         n_synonymous_with_error=int(nv.attrs["n_synonymous_with_error"])))
    return pd.DataFrame(rows).reindex(columns=COLUMNS)


def deposit_rows_with_se() -> dict[str, int]:
    """The external deposits' rows with a score and an SE (the own model's variants)."""
    from atlas.external_reliability import EXTERNAL as SPECS, load_deposit
    out = {}
    for urn, spec in SPECS.items():
        if spec["gene"] in EXTERNAL:
            d = load_deposit(urn)
            ok = (pd.to_numeric(d["score"], errors="coerce").notna()
                  & pd.to_numeric(d[spec["se"]], errors="coerce").notna())
            out[spec["gene"]] = int(ok.sum())
    return out


# ==========================================================================
# the self-test's real cells under the neutral model
# ==========================================================================
def neutral_cells(strata=rc.REAL_STRATA) -> list[tuple]:
    """The same 18 cells as atlas.rank_ceiling.own_cells, each with its gene's neutral
    residuals (a fixed model)."""
    tasks = []
    for gene, stratum, _, ids, y, s, _res in rc.own_cells(strata):
        tasks.append((gene, stratum, "neutral", ids, y, s, None,
                      neutral_variants(gene)["z"].to_numpy(float)))
    return tasks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(paths.RESULTS))
    args = ap.parse_args(argv)
    out = Path(args.out)
    checks = {"neutral_counts": neutral_check(), "brca1_score_is_the_mean": brca1_mean_check()}
    if not checks["neutral_counts"]["holds"] or not checks["brca1_score_is_the_mean"]["holds"]:
        print(json.dumps(rc.jsonable(checks), indent=1))
        return 1
    table = summary_table()
    n_se = deposit_rows_with_se()
    own_ext = table["gene"].isin(EXTERNAL) & table["model"].eq("own")
    table.loc[own_ext, "n"] = table.loc[own_ext, "gene"].map(n_se)
    table.to_csv(out / paths.name("error_models"), sep="\t", index=False)
    rec = {"definition": {
               "own": {"se_genes": "Gaussian, scale the variant's SE",
                       "BRCA1": "the empirical distribution of d = (rep1 - rep2) / 2 within the "
                                "cell, s = 1 (on the -score scale)"},
               "neutral": f"synonymous SNVs with spliceai_ds below {SPLICEAI_NEUTRAL} and "
                          "error information; z = (y - median) / s (BRCA1 s = 1)",
               "kde": rc.definition()["kde"],
               "gaussian_rows": "n = the gene's variants with an SE (external: the deposit's "
                                "rows with a score and an SE); sd 1"},
           "checks": checks,
           "neutral_variants": {g: int(len(neutral_variants(g))) for g in GENES},
           "synonymous_with_error": {g: int(neutral_variants(g).attrs["n_synonymous_with_error"])
                                     for g in GENES},
           "inputs": [paths.name("score_matrix"), paths.name("assay_aux"),
                      paths.name("external_coding_consequence"), paths.name("external_scores")]}
    (out / paths.name("error_models_summary")).write_text(
        json.dumps(rc.jsonable(rec), indent=1) + "\n")
    print(table.to_string(float_format=lambda v: f"{v:.4g}"))

    # the self-test, second version: the first version's rows again, and the neutral model
    t0 = time.time()
    known = rc.known_truth(args.workers)
    own = rc.real_cells(rc.own_cells(), args.workers)
    t1 = time.time()
    neu = rc.real_cells(neutral_cells(), args.workers)
    t2 = time.time()
    rows = pd.concat([known, own, neu], ignore_index=True)
    v1 = json.loads(SELFTEST_V1.read_text())["data_sets"]
    v1 = pd.DataFrame(v1)
    again = rows.iloc[:len(v1)].drop(columns=["seconds"]).reset_index(drop=True)
    same = (len(v1) == len(again)
            and all((again[c].astype(str) == v1[c].astype(str)).all() for c in ("part", "error",
                                                                                "case", "rep"))
            and all((again[c].to_numpy(float).view(np.uint64)
                     == v1[c].to_numpy(float).view(np.uint64)).all()
                    for c in ("truth", "estimate", "difference")))
    if not same:
        print("the self-test's first version is not reproduced")
        return 1
    acc = rc.write_selftest(rows, out, "rank_ceiling_selftest", {
        "this_version": "known truths, and the real cells with both error models (the "
                        "neutral model's from atlas.error_models); the first version's rows "
                        "computed again, the same bits",
        "first_version_reproduced": bool(same)})
    print(json.dumps(rc.jsonable(acc), indent=1))
    print(json.dumps({"workers": args.workers,
                      "seconds_wall": {"first_version_again": round(t1 - t0, 1),
                                       "neutral_cells": round(t2 - t1, 1)},
                      "seconds_per_estimate_neutral": neu.groupby("case")["seconds"].mean()
                      .round(3).to_dict()}))
    return 0 if acc["known_truth_holds"] and acc["real_cells_hold"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
