"""The error models of the rank-scale ceiling (A4b), for the nine genes with error
information, their error scales, and the self-test's real cells.

Neither model takes a deposited SE variant by variant: on the neutral variants an SE
overstates how much the precision differs between variants (results/error_scale_v1.tsv),
and a per-variant SE taken as known overstates the agreement of a remeasurement (the
validation, atlas.rank_ceiling_validation). Two models per gene, both reported, the
neutral model first:

``neutral``  the spread of the gene's neutral variants: SNVs that are synonymous and
             whose spliceai_ds is below 0.1, with error information; residuals
             r = y - the batch's median. The SE genes: the error scale sigma(SE),
             calibrated on these residuals and their SEs (atlas.rank_ceiling.ErrorScale:
             five equal-count bins by SE, 1.4826 MAD per bin, non-decreasing, linear in
             log SE between the bins' median log SEs, constant beyond); the shape the
             empirical distribution of z = r / sigma(SE); in any cell s_i = sigma(SE_i),
             non-SNVs included. sigma and z are the gene's, not estimated again in a
             resample. BRCA1: z = r, s = 1. The model takes in the noise an SE does not
             capture (variation the two replicates share, the tails); were there real
             effects among the synonymous variants, it would be conservative.
``own``      the deposit's own error. The SE genes (BARD1, PALB2, BAP1 and the external
             five): Gaussian, the SE pooled within the cell, one value sqrt(mean(SE^2))
             of the cell's variants (in the matched ceilings the predictor's own
             variants), computed again from the variants each resample draws; the floor
             uses the cell's value. BRCA1, which has no SE of its own: the empirical
             distribution of half the difference of its two replicates,
             d = (rep1 - rep2) / 2, within the cell (gene x stratum x population), with
             s = 1. The score is the replicates' mean (checked row by row), and the sum
             and the difference of two independent errors have the same distribution, so
             d is distributed as the mean's error. Within the cell rather than across the
             gene: d's spread differs between strata (+-1-2 bp against the rest).

BRCA1's two models are those of the first generation (the same bits).

The neutral variants: the core four from the score matrix (``consequence`` and
``spliceai_ds``; their numbers must be the second batch of
results/reliability_neutral_check_v1.tsv: BRCA1 468, BARD1 1,231, PALB2 1,807, BAP1
1,322); the external five from results/external_coding_consequence_v1.parquet
(atlas.external_coding), by the same rule. y is functional_pathogenicity (-score for the
external deposits, as everywhere in the atlas): the residuals are on the scale the
ceilings use.

Outputs:
  results/error_scale_v1.tsv     one row per SE gene x bin: the neutral variants in the
                                 bin, their SE's minimum, median and maximum, x_b,
                                 sigma_b before and after the monotone fit, the block the
                                 fit put the bin in, the bin's sqrt(mean SE^2) and
                                 sigma_b's ratio to it;
  results/error_models_v2.tsv    one row per gene x model (BRCA1's own per stratum and
                                 population, and gene-wide), the columns of v1: the
                                 residuals' number, centre (their median), sd, robust sd
                                 (1.4826 MAD), kurtosis, the shares with |z| above 2 and
                                 3, the bandwidth; a Gaussian row gives the number of
                                 variants with an SE and sd 1;
  results/error_models_v2.json   the definitions and the checks;
  results/rank_ceiling_selftest_v3.tsv and .json  the self-test of atlas.rank_ceiling:
                                 the known truths of the second version (computed again,
                                 the same bits) and the 36 real cells under these models
                                 (BRCA1's the same bits as the second version's).

Usage (PYTHONPATH=src):  python -m atlas.error_models [--workers N]
                         (``selftest`` alone: python -m atlas.rank_ceiling selftest)
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
SE_GENES = CORE_SE + EXTERNAL
GENES = ("BRCA1",) + SE_GENES
MODELS = ("neutral", "own")
SPLICEAI_NEUTRAL = 0.1
NEUTRAL_EXPECTED = {"BRCA1": 468, "BARD1": 1231, "PALB2": 1807, "BAP1": 1322}
MEAN_ATOL = 1e-12          # BRCA1's score against its replicates' mean
MIN_N = 30
SELFTEST_PREVIOUS = paths.RESULTS / "rank_ceiling_selftest_v2.json"   # the second version

COLUMNS = ["gene", "model", "kind", "population", "stratum", "n", "centre", "sd",
           "robust_sd", "kurtosis", "share_abs_gt_2", "share_abs_gt_3", "bandwidth",
           "score_median", "score_sd", "n_synonymous_with_error"]
KIND_OWN_SE = "gaussian: the SE pooled within the cell, sqrt(mean SE^2)"
KIND_NEUTRAL_SE = "empirical: neutral SNVs, z = r / sigma(SE)"


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


@lru_cache(maxsize=None)
def _neutral_frame(gene: str) -> pd.DataFrame:
    """The gene's neutral SNVs with error information, sorted by identifier: id, y, se
    (BRCA1: none), r = y - the batch's median."""
    if gene in EXTERNAL:
        x = external_coding()
        g = x[x["gene"].eq(gene) & x["consequence"].eq("synonymous")
              & x["standard_error"].notna() & x["functional_pathogenicity"].notna()]
        syn = len(g)
        g = g[g["spliceai_ds"] < SPLICEAI_NEUTRAL]
        out = pd.DataFrame({"id": g["variant_urn"], "y": g["functional_pathogenicity"],
                            "se": g["standard_error"]})
    else:
        d = core_frame()
        g = d[d["gene"].eq(gene) & d["population"].eq(population.SNV)
              & d["consequence"].eq("synonymous") & d["has_error"]
              & d["functional_pathogenicity"].notna()]
        syn = len(g)
        g = g[g["spliceai_ds"] < SPLICEAI_NEUTRAL]
        out = pd.DataFrame({"id": g["variant_id"], "y": g["functional_pathogenicity"],
                            "se": g["se"] if gene != "BRCA1" else np.nan})
    out = out.sort_values("id", kind="stable").reset_index(drop=True)
    out["r"] = out["y"] - out["y"].median()
    out.attrs["n_synonymous_with_error"] = syn
    return out


@lru_cache(maxsize=None)
def error_scale(gene: str) -> rc.ErrorScale:
    """sigma(SE) of an SE gene, calibrated on its neutral variants."""
    if gene not in SE_GENES:
        raise ValueError(f"{gene} has no SE")
    nv = _neutral_frame(gene)
    return rc.ErrorScale(nv["se"].to_numpy(float), nv["r"].to_numpy(float))


def neutral_scale(gene: str, se) -> np.ndarray:
    """The neutral model's s for variants of the gene with these SEs (BRCA1: 1)."""
    if gene == "BRCA1":
        return np.ones(len(np.atleast_1d(se)))
    return error_scale(gene)(se)


def neutral_variants(gene: str) -> pd.DataFrame:
    """The gene's neutral SNVs with error information: identifier, y, se, r, s, z, with
    s = sigma(SE) (BRCA1: 1) and z = r / s."""
    out = _neutral_frame(gene).copy()
    out.attrs = dict(_neutral_frame(gene).attrs)
    out["s"] = (1.0 if gene == "BRCA1" else error_scale(gene)(out["se"].to_numpy(float)))
    out["z"] = out["r"] / out["s"]
    return out


@lru_cache(maxsize=None)
def neutral_model(gene: str) -> rc.ErrorModel:
    return rc.ErrorModel(neutral_variants(gene)["z"].to_numpy(float), name="neutral")


def neutral_check() -> dict:
    got = {g: int(len(_neutral_frame(g))) for g in NEUTRAL_EXPECTED}
    return {"counts": got, "expected": NEUTRAL_EXPECTED, "holds": got == NEUTRAL_EXPECTED}


def scale_record() -> dict:
    """Per SE gene, the most bins the monotone fit pooled into one. A record, not a check:
    where the SE says little about the spread, the fit makes sigma flat (RAD51D pools
    bins 2-4; XRCC2 and CTCF are as flat by slope)."""
    return {"largest_block": {g: error_scale(g).largest_block() for g in SE_GENES}}


# ==========================================================================
# the tables
# ==========================================================================
def scale_table() -> pd.DataFrame:
    rows = []
    for gene in SE_GENES:
        t = error_scale(gene).table()
        t.insert(0, "gene", gene)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


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
            rows.append(_row(gene, "own", KIND_OWN_SE, "all", "gene",
                             rc.ErrorModel().describe() | {"n": n_se}))
        nv = neutral_variants(gene)
        rows.append(_row(gene, "neutral", "empirical: neutral SNVs" if gene == "BRCA1"
                         else KIND_NEUTRAL_SE, population.SNV, "gene",
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
# the self-test's real cells
# ==========================================================================
def neutral_cells(strata=rc.REAL_STRATA) -> list[tuple]:
    """The same 18 cells as atlas.rank_ceiling.own_cells, each with its gene's neutral
    residuals (a fixed model) and s = sigma(SE) (BRCA1: 1)."""
    tasks = []
    for c in rc.real_cell_inputs(strata):
        gene = c["gene"]
        s = np.ones(len(c["y"])) if gene == "BRCA1" else error_scale(gene)(c["se"])
        tasks.append((gene, c["stratum"], "neutral", c["ids"], c["y"], s, None,
                      neutral_variants(gene)["z"].to_numpy(float)))
    return tasks


def _same_bits(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    return (len(a) == len(b)
            and all((a[c].astype(str) == b[c].astype(str)).all()
                    for c in ("part", "error", "case", "rep"))
            and all((a[c].to_numpy(float).view(np.uint64)
                     == b[c].to_numpy(float).view(np.uint64)).all()
                    for c in ("truth", "estimate", "difference")))


def selftest(out: Path, workers: int) -> int:
    """The self-test, third version: the second version's known truths computed again
    (the same bits), and the real cells under both models (BRCA1's the same bits)."""
    t0 = time.time()
    known = rc.known_truth(workers)
    t1 = time.time()
    own = rc.real_cells(rc.own_cells(), workers)
    t2 = time.time()
    neu = rc.real_cells(neutral_cells(), workers)
    t3 = time.time()
    prev = pd.DataFrame(json.loads(SELFTEST_PREVIOUS.read_text())["data_sets"])
    same_known = _same_bits(known, prev[prev["part"].eq("known_truth")])
    real = pd.concat([own, neu], ignore_index=True)
    brca1_new = real[real["case"].str.startswith("BRCA1 ")]
    brca1_old = prev[prev["part"].eq("real_cells") & prev["case"].str.startswith("BRCA1 ")]
    same_brca1 = _same_bits(brca1_new, brca1_old)
    if not (same_known and same_brca1):
        print(f"the second version is not reproduced: known truths {same_known}, "
              f"BRCA1 {same_brca1}")
        return 1
    rows = pd.concat([known, neu, own], ignore_index=True)
    acc = rc.write_selftest(rows, out, "rank_ceiling_selftest", {
        "this_version": "the known truths of the second version, computed again (the same "
                        "bits); the real cells under the third generation's error models: "
                        "neutral, the gene's residuals over the calibrated scale sigma(SE); "
                        "own, the SE pooled within the cell; BRCA1's cells unchanged (the "
                        "same bits as the second version's)",
        "previous_version_reproduced": {"known_truth": bool(same_known),
                                        "brca1_real_cells": bool(same_brca1)}})
    print(json.dumps(rc.jsonable(acc), indent=1))
    print(json.dumps({"workers": workers,
                      "seconds_wall": {"known_truth": round(t1 - t0, 1),
                                       "own_cells": round(t2 - t1, 1),
                                       "neutral_cells": round(t3 - t2, 1)},
                      "seconds_per_estimate": {
                          f"{e} {c}": v for (e, c), v in
                          real.groupby(["error", "case"])["seconds"].mean().round(3).items()}},
                     default=str))
    return 0 if acc["known_truth_holds"] and acc["real_cells_hold"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(paths.RESULTS))
    args = ap.parse_args(argv)
    out = Path(args.out)
    checks = {"neutral_counts": neutral_check(), "brca1_score_is_the_mean": brca1_mean_check()}
    record = {"monotone_fit": scale_record()}
    print(json.dumps(rc.jsonable(checks | record), indent=1))
    if not checks["neutral_counts"]["holds"] or not checks["brca1_score_is_the_mean"]["holds"]:
        return 1
    scales = scale_table()
    scales.to_csv(out / paths.name("error_scale"), sep="\t", index=False)
    table = summary_table()
    n_se = deposit_rows_with_se()
    own_ext = table["gene"].isin(EXTERNAL) & table["model"].eq("own")
    table.loc[own_ext, "n"] = table.loc[own_ext, "gene"].map(n_se)
    table.to_csv(out / paths.name("error_models"), sep="\t", index=False)
    rec = {"definition": {
               "neutral": {"variants": f"synonymous SNVs with spliceai_ds below "
                                       f"{SPLICEAI_NEUTRAL} and error information; "
                                       "r = y - their median",
                           "se_genes": "z = r / sigma(SE), s = sigma(SE) in every cell; sigma "
                                       "and z the gene's, fixed in the resamples",
                           "BRCA1": "z = r, s = 1",
                           "sigma": {"bins": rc.SCALE_BINS,
                                     "cut": "the neutral variants sorted by SE (stable, ties "
                                            "in identifier order), numpy.array_split",
                                     "x_b": "the bin's median log SE",
                                     "sigma_b": f"{rc.MAD_FACTOR} x the median absolute "
                                                "deviation of the bin's residuals from their "
                                                "median",
                                     "monotone": "pool adjacent violators, equal weights, "
                                                 "non-decreasing in bin order",
                                     "between": "linear in log SE between (x_b, sigma_b); the "
                                                "first bin's value below x_1, the last's "
                                                "above x_5"}},
               "own": {"se_genes": "Gaussian, the SE pooled within the cell: sqrt(mean SE^2) "
                                   "of its variants, of the resampled variants in a "
                                   "resample, the cell's value for the floor",
                       "BRCA1": "the empirical distribution of d = (rep1 - rep2) / 2 within the "
                                "cell, s = 1 (on the -score scale)"},
               "per_variant": "the SE taken variant by variant: the validation's diagnostic "
                              "only, in no ceiling table",
               "order": "neutral first, own beside it",
               "kde": rc.definition()["kde"],
               "gaussian_rows": "n = the gene's variants with an SE (external: the deposit's "
                                "rows with a score and an SE); sd 1"},
           "checks": checks,
           "records": record,
           "neutral_variants": {g: int(len(_neutral_frame(g))) for g in GENES},
           "synonymous_with_error": {g: int(_neutral_frame(g).attrs["n_synonymous_with_error"])
                                     for g in GENES},
           "inputs": [paths.name("score_matrix"), paths.name("assay_aux"),
                      paths.name("external_coding_consequence"), paths.name("external_scores")]}
    (out / paths.name("error_models_summary")).write_text(
        json.dumps(rc.jsonable(rec), indent=1) + "\n")
    print(scales.to_string(float_format=lambda v: f"{v:.4g}"))
    print(table.to_string(float_format=lambda v: f"{v:.4g}"))
    return selftest(out, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
