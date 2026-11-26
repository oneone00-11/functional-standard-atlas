"""Attenuation ceilings matched to the variants each correlation uses, with bootstrap
intervals, and the realisation: core and external genes, per population.

The current generation (results/matched_ceilings_v3.tsv and .json; A4b.4): the ceiling on
the rank scale. The cells, their variants, n, rho and rho's interval are v2's (the second
generation is computed again and checked against its file, every column bit for bit);
the external genes' ``all`` rows are no longer written (G18: their predictors score
intronic SNVs only, so ``all`` repeats ``splice_region`` and shares its name with the core
genes' ``all``, coding included). v2's reliability, ceiling, realisation and estimable
columns keep their values under the names with the suffix _v2 (with seed_v2 and
bootstrap_B_v2, which also gave rho's interval); BRCA1 also carries ceiling_replicate_k1;
ceiling_inflated and breakeven_inflation are gone; set_sha is the variant set's sha256
(first 16 hex digits). The nine genes with an error model carry, under both models of
atlas.error_models (neutral first, then own; atlas.rank_cells), the rank-scale ceiling of
the cell's set with its interval, floor and estimable, the realisation rho / ceiling with
its interval on the same 200 resamples, the share of resamples above 1 and the number
where it is undefined, and lof_share and neutral_in_range. RAD51C, BRCA2 and VHL have rho
alone. One set gets the same ceiling in every cell and in the layer tables
(results/reliability_v3.tsv, results/external_reliability_v3.tsv), checked here.

Its summary, per model (snv; a cell with selection_overlap enters no quantity across
genes; a realisation enters only where the cell's ceiling is estimable and, for the
neutral model, in range): (1) per stratum and group (core, external, all), the genes
whose layer ceiling enters, its median and range, and per predictor rho and the
realisation across genes; (2) per gene x stratum, n, both layer ceilings, the predictor
with the highest rho and the splice family's, among the predictors that scored at least
90% of the cell's variants (G17), with rho and both realisations; (3) at 3-10 and 11-50
bp, that predictor's realisation and the cells whose interval's upper bound is below 1
and below 0.5; (4) beyond 50 bp, the layer ceilings; (5) the cells whose realisation
exceeds 1 or whose interval crosses 1; (6) v2 against v3, with the trial of the first
supplement. Per stratum, the cells in and out of the neutral model's range.

The second generation (results/matched_ceilings_v2.tsv and .json):

Cells: group (core, external) x gene x stratum (atlas.reliability.STRATA, the seven) x
predictor x population (atlas.population), with at least MIN_N (30) variants. A cell's
variants are those the predictor scored that carry a functional score; for a gene with an
error model, also its error information (A2's rule). Core: the seven genes of
results/score_matrix_atlas_v3.parquet with results/assay_aux_v2.parquet, BRCA1 by
replicates, BARD1, PALB2 and BAP1 by validated SE, and RAD51C, BRCA2 and VHL, which have no
usable error model, with rho and its interval only (method ``none``); the 23 predictors
of atlas.predictors. External: CTCF, SFPQ, RAD51D, XRCC2 and SBDS by SE, their 13 columns
(results/external_scores_v2.parquet; all SNVs). A2's two scopes become one: the
population. rho, the reliability and ceiling on the cell's own variants and the
realisation come from one set of variants, each with its bootstrap interval (B = 2,000,
the seed from SEED and the cell's key: group, gene, stratum, predictor, population).
Columns added: ``estimable`` (the ceiling's lower bound above 0), ``selection_overlap``
(atlas.predictors.SELECTION_OVERLAP), ``ceiling_inflated`` (BARD1, PALB2, BAP1: SE^2
times the gene's inflation factor from atlas.reliability's neutral check) and
``breakeven_inflation`` = (1 - rho^2) / (1 - the reliability before clipping), the factor
by which the error variance would have to grow for the ceiling to fall to the
predictor's rho (empty where that reliability is not below 1). A realisation whose cell
is not estimable is kept but enters no summary of realisations. The columns comparing
with v5's stratum ceiling are gone.

The summary (snv population; a flagged cell enters no quantity across genes): (1) per
stratum and group (core, external, all), the genes whose layer ceiling
(results/reliability_v2.tsv, results/external_reliability_v2.tsv) is estimable, its
median and range, and per predictor the median, range and number of genes of rho and of
the realisation; (2) per gene x stratum, n, the layer ceiling and its interval, the
predictor with the highest rho and the splice family's, each with rho, realisation and
their intervals, and the cell's breakeven factor; (3) at 3-10 and 11-50 bp, that best
predictor's realisation, and the cells whose realisation interval's upper bound is below
1, and below 0.5; (4) beyond 50 bp, the layer ceiling per gene; (5) the cells whose
realisation exceeds 1 or whose interval crosses 1. With the regression against
results/matched_ceilings_v1.tsv.

A2's first generation (results/matched_ceilings_v1.*) follows; its functions are kept.

Attenuation ceilings matched to the variants each correlation uses, with bootstrap
intervals; and the realisation of the external genes.

v5 estimates one ceiling per gene x stratum from every variant with error information
(results/reliability_v1.tsv), and each predictor's rho from the variants that predictor
scored (results/attenuation_v1.tsv). Where the two sets differ, realisation's numerator
and denominator come from different variants: BARD1's +-1-2 bp stratum holds 147 variants,
28 of them 3 bp deletions across the splice site, while SpliceAI, Pangolin and CADD score
only the 119 SNVs. Here the ceiling is estimated again on exactly the variants that enter
each rho. Nothing v5 wrote is rewritten; its functions are imported, not changed.

Genes and estimators (v5's):
  core      BRCA1 by replicates (Spearman between the two replicate fitness scores, raised
            to their mean by Spearman-Brown, atlas.reliability.spearman_brown), BARD1 and
            PALB2 by validated SE (1 - mean(SE^2) / var(functional_pathogenicity), clipped to
            [0, 1]); results/score_matrix_atlas_v2.parquet with results/assay_aux_v1.parquet;
            the nineteen predictors of atlas.reliability.MODELS; the seven strata of
            atlas.reliability.STRATA
  external  CTCF, SFPQ, RAD51D, XRCC2 and SBDS by SE (checked against each deposit's 95%
            interval in atlas.external_reliability); results/external_scores_v1.parquet;
            its nine score columns; the four intronic strata and splice_region
Scopes:
  scored    the variants the predictor scored that carry a functional score and the error
            information: v5's rho set, less any variant without error information
  snv       those of them whose ref and alt are single bases; the external genes hold SNVs
            only, so for them this is the one scope
Cells: gene x stratum x predictor x scope with at least MIN_N (30) variants. Each gives n,
rho (Spearman of the predictor against functional_pathogenicity, atlas.evaluate's
direction), the reliability and ceiling on the cell's own variants (for SE also the
reliability before clipping), realisation = rho / ceiling (empty where the ceiling is 0),
and for comparison v5's stratum ceiling (core: reliability_v1.tsv's primary route;
external: external_reliability_v1.tsv's main rows) with the realisation it gives.

Bootstrap: B = 2,000 resamples of the cell's variants with replacement (BRCA1's replicate
pairs are resampled as variants), from a seed fixed per cell (from SEED and the cell's key;
written out). Each resample recomputes rho, reliability, ceiling and realisation; their
2.5% and 97.5% quantiles are reported, with the share of resamples whose realisation
exceeds 1 among those where it is defined.

Outputs: results/matched_ceilings_v1.tsv, one row per cell, and
results/matched_ceilings_v1.json: (1) by stratum, for the core three, the external five and
all eight, the matched ceiling's median and range and each predictor's cross-gene median
rho and realisation (snv scope); (2) v5's coding-against-+-1-2 comparison redone with the
matched ceilings, both scopes, with each core gene left out in turn; (3) the cells whose
matched ceiling differs from the stratum ceiling by more than 0.05, and those whose
realisation exceeds 1 or whose interval crosses 1; with the checkpoints and the functional
score's distribution per gene x stratum.

Usage (PYTHONPATH=src):  python -m atlas.matched_ceilings [--workers N] [--layer-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from atlas import paths, population, predictors
from atlas.evaluate import classify_region
from atlas.matched_realisation import CODING, MISSENSE_ONLY, SPLICE
from atlas.reliability import MIN_N, MODELS, STRATA, _stratum_mask, spearman_brown

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
RAW = REPO / "data" / "raw"

SEED = 20261122
B = 2000
# the core genes' error models (atlas.reliability.CORE_METHOD), and the genes without one
CORE = {"BRCA1": "replicates", "BARD1": "se", "PALB2": "se", "BAP1": "se",
        "RAD51C": "none", "BRCA2": "none", "VHL": "none"}
EXTERNAL_GENES = ("CTCF", "SFPQ", "RAD51D", "XRCC2", "SBDS")
EXTERNAL_MODELS = predictors.EXTERNAL
POPULATIONS = population.POPULATIONS        # the cells' scope in this generation
INTRONIC = ["splice_1_2", "splice_3_10", "splice_11_50", "splice_deep"]
EXTERNAL_STRATA = ["splice_region"] + INTRONIC
SCOPES = ("scored", "snv")
DIFF_LIST = 0.05
# the instruction's checkpoints (a trial outside this repository, not a result of it):
# (all variants with error information, the SNVs SpliceAI, Pangolin and CADD scored)
CHECKPOINTS = {("BARD1", "splice_1_2"): (0.586, 0.454), ("PALB2", "splice_1_2"): (0.794, 0.752)}
CHECK_TOL = 0.005
CHECK_MODELS = ["spliceai_ds", "pangolin_score", "cadd"]


# ==========================================================================
# the variants
# ==========================================================================
def load_core(matrix: Path | None = None, aux: Path | None = None) -> pd.DataFrame:
    m = population.load(None, matrix)
    m = m[m["gene"].isin(CORE)]
    a = pd.read_parquet(aux or paths.path("assay_aux"),
                        columns=["variant_id", "gene", "fit_rep1", "fit_rep2", "se", "se_status"])
    df = m.merge(a, on=["variant_id", "gene"], how="left")
    df["method"] = df["gene"].map(CORE)
    rep = df["fit_rep1"].notna() & df["fit_rep2"].notna()
    se = df["se"].notna() & (df["se_status"] == "validated")
    # "has_error": the variant carries what its gene's estimator needs (nothing for none)
    df["has_error"] = np.select([df["method"] == "replicates", df["method"] == "se"],
                                [rep, se], default=True)
    df["group"] = "core"
    return _common(df)


def load_external(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path or paths.path("external_scores"))
    df = df.rename(columns={"standard_error": "se"})
    df["region"] = df["hgvs_c"].map(classify_region)
    if not (df["region"] == df["stratum"]).all():
        raise SystemExit("the external table's stratum differs from classify_region")
    df["fit_rep1"] = df["fit_rep2"] = np.nan
    df["method"] = "se"
    df["has_error"] = df["se"].notna()
    df["group"] = "external"
    return _common(df)


def _common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["snv"] = (df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)
    return df.reset_index(drop=True)


# ==========================================================================
# the estimators, for one set of variants (point) and for many resamples (rows)
# ==========================================================================
def point(x, y, method, se=None, r1=None, r2=None) -> dict:
    """rho, reliability and ceiling as v5 computes them (the same expressions, so the
    same floats where the variants are the same)."""
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    if method == "none":                       # no error model: rho alone
        return {"rho": rho, "rep_rho": np.nan, "reliability": np.nan,
                "reliability_unclipped": np.nan, "ceiling": np.nan, "realisation": np.nan}
    if method == "replicates":
        rep = float(pd.Series(r1).corr(pd.Series(r2), method="spearman"))
        rel = spearman_brown(rep)
        raw = np.nan
    else:
        var = float(pd.Series(y).var())
        err = float((pd.Series(se) ** 2).mean())
        raw = 1 - err / var if var > 0 else np.nan
        rel = float(np.clip(raw, 0.0, 1.0)) if var > 0 else np.nan
        rep = np.nan
    ceiling = float(np.sqrt(max(rel, 0.0))) if np.isfinite(rel) else np.nan
    real = rho / ceiling if np.isfinite(ceiling) and ceiling > 0 else np.nan
    return {"rho": rho, "rep_rho": rep, "reliability": rel, "reliability_unclipped": raw,
            "ceiling": ceiling, "realisation": real}


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman rho of each row of a with the same row of b (average ranks for ties)."""
    ra = rankdata(a, axis=1)
    rb = rankdata(b, axis=1)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ra * rb).sum(axis=1) / den, np.nan)


def resample_stats(idx: np.ndarray, x, y, method, se=None, r1=None, r2=None) -> dict:
    """The estimators on each resample, one per row of idx."""
    rho = spearman_rows(x[idx], y[idx])
    if method == "none":
        nan = np.full(len(rho), np.nan)
        return {"rho": rho, "reliability": nan, "ceiling": nan, "realisation": nan}
    if method == "replicates":
        r = spearman_rows(r1[idx], r2[idx])
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(1 + r != 0, 2 * r / (1 + r), np.nan)
    else:
        yy = y[idx]
        var = yy.var(axis=1, ddof=1)
        err = (se[idx] ** 2).mean(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = np.where(var > 0, 1 - err / var, np.nan)
        rel = np.clip(raw, 0.0, 1.0)
    ceiling = np.sqrt(np.clip(rel, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        real = np.where(ceiling > 0, rho / ceiling, np.nan)
    return {"rho": rho, "reliability": rel, "ceiling": ceiling, "realisation": real}


def cell_seed(key: tuple) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}|{'|'.join(map(str, key))}".encode())
                          .digest()[:8], "big")


def bootstrap(arrays: dict, method: str, seed: int, b: int = B, budget: int = 4_000_000) -> dict:
    """Quantiles of the estimators over b resamples, drawn in chunks of rows."""
    n = len(arrays["x"])
    rng = np.random.default_rng(seed)
    chunk = max(1, min(b, budget // max(n, 1)))
    out = {k: [] for k in ("rho", "reliability", "ceiling", "realisation")}
    done = 0
    while done < b:
        k = min(chunk, b - done)
        idx = rng.integers(0, n, size=(k, n))
        res = resample_stats(idx, arrays["x"], arrays["y"], method, arrays.get("se"),
                             arrays.get("r1"), arrays.get("r2"))
        for name in out:
            out[name].append(res[name])
        done += k
    stats = {}
    for name, parts in out.items():
        v = np.concatenate(parts)
        ok = v[np.isfinite(v)]
        stats[f"{name}_lo"] = float(np.quantile(ok, 0.025)) if len(ok) else np.nan
        stats[f"{name}_hi"] = float(np.quantile(ok, 0.975)) if len(ok) else np.nan
        if name == "realisation":
            stats["share_realisation_gt_1"] = float((ok > 1).mean()) if len(ok) else np.nan
            stats["bootstrap_realisation_undefined"] = int(len(v) - len(ok))
    return stats


# ==========================================================================
# the cells
# ==========================================================================
def cell_tasks(df: pd.DataFrame, models: list[str], strata: list[str],
               scopes=SCOPES) -> list[tuple]:
    tasks = []
    for gene, g in df.groupby("gene"):
        method = g["method"].iloc[0]
        for stratum in strata:
            s = g[_stratum_mask(g["region"], stratum) & g["has_error"]
                  & g["functional_pathogenicity"].notna()]
            for model in models:
                if model not in s.columns:
                    continue
                base = s[s[model].notna()]
                for scope in scopes:
                    c = (base[base["snv"]] if scope == "snv" else
                         base[~base["snv"]] if scope == "non_snv" else base)
                    if len(c) < MIN_N:
                        continue
                    arrays = {"x": c[model].to_numpy(float),
                              "y": c["functional_pathogenicity"].to_numpy(float)}
                    idcol = "variant_urn" if g["group"].iloc[0] == "external" else "variant_id"
                    if idcol in c.columns:        # the variant set, for the rank-scale ceiling
                        arrays["ids"] = c[idcol].astype(str).tolist()
                    if method == "replicates":
                        arrays |= {"r1": c["fit_rep1"].to_numpy(float),
                                   "r2": c["fit_rep2"].to_numpy(float)}
                    elif method == "se":
                        arrays["se"] = c["se"].to_numpy(float)
                    key = (g["group"].iloc[0], gene, stratum, model, scope)
                    tasks.append((key, method, arrays))
    return tasks


def run_cell(task: tuple) -> dict:
    key, method, arrays = task
    group, gene, stratum, model, scope = key
    p = point(arrays["x"], arrays["y"], method, arrays.get("se"), arrays.get("r1"),
              arrays.get("r2"))
    seed = cell_seed(key)
    return {"group": group, "gene": gene, "stratum": stratum, "model": model, "scope": scope,
            "n": len(arrays["x"]), "method": method, **p, "seed": seed, "bootstrap_B": B,
            **bootstrap(arrays, method, seed)}


def compute(tasks: list[tuple], workers: int = 1) -> pd.DataFrame:
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(run_cell, tasks, chunksize=4))
    else:
        rows = [run_cell(t) for t in tasks]
    return pd.DataFrame(rows)


def stratum_ceilings() -> pd.DataFrame:
    """v5's stratum ceiling per gene: the core genes' primary route, the external genes'
    main rows."""
    from atlas.external_reliability import v5_primary
    core = v5_primary(pd.read_csv(RESULTS / "reliability_v1.tsv", sep="\t"))
    ext = pd.read_csv(RESULTS / "external_reliability_v1.tsv", sep="\t")
    ext = ext[ext["role"] == "main"]
    both = pd.concat([core[["gene", "stratum", "method", "ceiling"]],
                      ext[["gene", "stratum", "method", "ceiling"]]], ignore_index=True)
    return both.rename(columns={"method": "v5_method", "ceiling": "v5_ceiling"})


def add_v5(table: pd.DataFrame, ceil: pd.DataFrame) -> pd.DataFrame:
    t = table.merge(ceil, on=["gene", "stratum"], how="left")
    with np.errstate(invalid="ignore", divide="ignore"):
        t["realisation_v5_ceiling"] = np.where(t["v5_ceiling"] > 0,
                                               t["rho"] / t["v5_ceiling"], np.nan)
    t["ceiling_minus_v5"] = t["ceiling"] - t["v5_ceiling"]
    return t


# ==========================================================================
# checkpoints
# ==========================================================================
def check_attenuation(table: pd.DataFrame, df: pd.DataFrame,
                      att: pd.DataFrame | None = None) -> dict:
    """The scored scope's n and rho against attenuation_v1.tsv, cell by cell. A cell
    with fewer variants only because some lack error information is listed and kept;
    any other difference is a failure."""
    att = att if att is not None else pd.read_csv(RESULTS / "attenuation_v1.tsv", sep="\t",
                                                  float_precision="round_trip")
    ours = table[(table["group"] == "core") & (table["scope"] == "scored")] \
        .set_index(["gene", "stratum", "model"])
    same, fewer, other = 0, [], []
    for r in att.itertuples(index=False):
        key = (r.gene, r.stratum, r.model)
        if key not in ours.index:
            other.append({"cell": key, "why": "no matched cell"})
            continue
        o = ours.loc[key]
        if int(o["n"]) == int(r.n) and o["rho"] == r.rho:
            same += 1
            continue
        g = df[(df["gene"] == r.gene) & _stratum_mask(df["region"], r.stratum)
               & df[r.model].notna() & df["functional_pathogenicity"].notna()]
        lacking = int((~g["has_error"]).sum())
        if int(o["n"]) == int(r.n) - lacking and lacking > 0:
            fewer.append({"cell": key, "v5_n": int(r.n), "n": int(o["n"]),
                          "without_error_information": lacking})
        else:
            other.append({"cell": key, "v5_n": int(r.n), "n": int(o["n"]),
                          "v5_rho": float(r.rho), "rho": float(o["rho"])})
    return {"cells_in_attenuation": int(len(att)), "identical": same,
            "fewer_only_for_missing_error_information": fewer, "other_differences": other}


def check_trial(table: pd.DataFrame, df: pd.DataFrame) -> dict:
    """The instruction's checkpoint: BARD1 and PALB2 at +-1-2 bp, the ceiling on every
    variant with error information, and on the SNVs SpliceAI, Pangolin and CADD scored."""
    out = {}
    for (gene, stratum), (want_all, want_snv) in CHECKPOINTS.items():
        g = df[(df["gene"] == gene) & (df["region"] == stratum) & df["has_error"]]
        p_all = point(g["functional_pathogenicity"], g["functional_pathogenicity"], "se",
                      g["se"])
        snv = g[g["snv"] & g[CHECK_MODELS].notna().all(axis=1)]
        p_snv = point(snv["functional_pathogenicity"], snv["functional_pathogenicity"], "se",
                      snv["se"])
        cells = table[(table["gene"] == gene) & (table["stratum"] == stratum)
                      & (table["scope"] == "snv") & table["model"].isin(CHECK_MODELS)]
        pg = cells[cells["model"] == "pangolin_score"].iloc[0]
        out[f"{gene} {stratum}"] = {
            "n_all": int(len(g)), "ceiling_all": p_all["ceiling"], "instruction_all": want_all,
            "n_snv": int(len(snv)), "ceiling_snv": p_snv["ceiling"], "instruction_snv": want_snv,
            "snv_cells_ceiling": cells.set_index("model")["ceiling"].to_dict(),
            "pangolin_rho_snv": float(pg["rho"]), "pangolin_realisation_snv": float(pg["realisation"]),
            "holds": bool(abs(p_all["ceiling"] - want_all) <= CHECK_TOL
                          and abs(p_snv["ceiling"] - want_snv) <= CHECK_TOL
                          and all(abs(c - want_snv) <= CHECK_TOL
                                  for c in cells["ceiling"]))}
    return out


# ==========================================================================
# the summary
# ==========================================================================
def _range(s: pd.Series) -> dict:
    s = s.dropna()
    return ({"n_genes": int(len(s)), "median": float(s.median()), "min": float(s.min()),
             "max": float(s.max())} if len(s) else {"n_genes": 0})


def snv_ceilings(df: pd.DataFrame, strata: list[str]) -> pd.DataFrame:
    """Per gene x stratum, the ceiling on every SNV with a functional score and error
    information (no predictor's coverage applied)."""
    rows = []
    for gene, g in df.groupby("gene"):
        method = g["method"].iloc[0]
        for stratum in strata:
            s = g[_stratum_mask(g["region"], stratum) & g["has_error"] & g["snv"]
                  & g["functional_pathogenicity"].notna()]
            if len(s) < MIN_N:
                continue
            p = point(s["functional_pathogenicity"], s["functional_pathogenicity"], method,
                      s["se"], s["fit_rep1"], s["fit_rep2"])
            rows.append({"group": g["group"].iloc[0], "gene": gene, "stratum": stratum,
                         "n": int(len(s)), "ceiling": p["ceiling"]})
    return pd.DataFrame(rows)


GROUPS = {"core_3": ["core"], "external_5": ["external"], "all_8": ["core", "external"]}


def by_stratum(table: pd.DataFrame, snvc: pd.DataFrame) -> dict:
    """Part 1, on the snv scope: per stratum and group, the SNV ceiling's median and
    range across genes, and per predictor the cross-gene medians."""
    out = {}
    snv = table[table["scope"] == "snv"]
    for stratum in STRATA:
        rec = {}
        for name, groups in GROUPS.items():
            c = snvc[(snvc["stratum"] == stratum) & snvc["group"].isin(groups)]
            t = snv[(snv["stratum"] == stratum) & snv["group"].isin(groups)]
            if not len(c) and not len(t):
                continue
            per_model = {}
            for model, d in t.groupby("model"):
                per_model[model] = {"n_genes": int(d["gene"].nunique()),
                                    "matched_ceiling": _range(d["ceiling"]),
                                    "rho_median": float(d["rho"].median()),
                                    "realisation_median": float(d["realisation"].median())
                                    if d["realisation"].notna().any() else None}
            rec[name] = {"snv_ceiling": _range(c["ceiling"]),
                         "snv_ceiling_by_gene": dict(zip(c["gene"], c["ceiling"])),
                         "per_predictor": per_model}
        if rec:
            out[stratum] = rec
    return out


def coding_vs_splice(table: pd.DataFrame) -> dict:
    """Part 2: v5's matched coding-against-+-1-2 comparison (atlas.matched_realisation),
    with the matched ceilings, for each scope; the full set and each core gene left out."""
    from atlas.matched_realisation import matched_metrics
    core = table[(table["group"] == "core") & ~table["model"].isin(MISSENSE_ONLY)]
    v5 = pd.read_csv(RESULTS / "matched_realisation_loo_v1.tsv", sep="\t")
    out = {"v5": v5.to_dict("records")}
    for scope in SCOPES:
        s = core[core["scope"] == scope]
        rows = []
        for tag, sub in [("full (all 3 genes)", s)] + [(f"drop {g}", s[s["gene"] != g])
                                                       for g in sorted(CORE)]:
            tab = (sub.groupby(["stratum", "model"])[["rho", "realisation"]].median()
                   .reset_index())
            wide = tab.pivot(index="model", columns="stratum", values=["rho", "realisation"])
            if ("realisation", CODING) not in wide.columns or \
                    ("realisation", SPLICE) not in wide.columns:
                continue
            have = wide.dropna(subset=[("realisation", CODING), ("realisation", SPLICE)])
            for rec in matched_metrics(have):
                rows.append({"scope": tag, **rec, "models": sorted(have.index)})
        out[scope] = rows
    return out


def lists(table: pd.DataFrame) -> dict:
    """Part 3: the cells whose matched ceiling differs from the stratum ceiling by more
    than DIFF_LIST, and those whose realisation exceeds 1 or whose interval crosses 1."""
    keep = ["group", "gene", "stratum", "model", "scope", "n"]
    d = table[table["ceiling_minus_v5"].abs() > DIFF_LIST]
    over = table[(table["realisation"] > 1) | (table["realisation_hi"] > 1)]
    return {
        "ceiling_differs_from_stratum_by_more_than_0.05": d[keep + [
            "ceiling", "v5_ceiling", "ceiling_minus_v5"]].to_dict("records"),
        "realisation_above_1_or_interval_crossing_1": over[keep + [
            "rho", "ceiling", "realisation", "realisation_lo", "realisation_hi",
            "share_realisation_gt_1"]].to_dict("records")}


def deposit_class_map() -> dict[str, str]:
    """accession -> the deposit's own class (abnormal / normal / ...), for the eight genes."""
    from atlas.external_reliability import EXTERNAL
    from atlas.external_scores import deposit_classes
    out = {}
    paths = {"urn:mavedb:00000097-0-2": RAW / "mavedb" / "urn_mavedb_00000097-0-2",
             "urn:mavedb:00001250-a-2": RAW / "mavedb" / "urn_mavedb_00001250-a-2",
             "urn:mavedb:00001259-a-2": RAW / "mavedb" / "urn_mavedb_00001259-a-2"}
    for urn, spec in EXTERNAL.items():
        if spec["gene"] in EXTERNAL_GENES:
            paths[urn] = RAW / "mavedb_external" / ("urn_mavedb_" + urn.split(":")[-1])
    for urn, d in paths.items():
        dep = pd.read_csv(d / "scores.csv", low_memory=False)
        meta = json.loads((d / "metadata.json").read_text())
        _, cls = deposit_classes(meta, dep)
        out.update(dict(zip(dep["accession"], cls)))
    return out


def distributions(df: pd.DataFrame, classes: dict[str, str]) -> list[dict]:
    """Per gene x stratum, functional_pathogenicity's 5/25/50/75/95th percentiles and the
    deposit's class counts, on every variant with error information and on the SNVs among
    them (no predictor's coverage applied)."""
    rows = []
    key = np.where(df["group"] == "core", df["variant_id"], df.get("variant_urn"))
    df = df.assign(cls=pd.Series(key).map(classes).fillna("unclassified").to_numpy())
    for gene, g in df.groupby("gene"):
        strata = STRATA if g["group"].iloc[0] == "core" else EXTERNAL_STRATA
        for stratum in strata:
            base = g[_stratum_mask(g["region"], stratum) & g["has_error"]
                     & g["functional_pathogenicity"].notna()]
            for scope, s in (("with_error", base), ("snv_with_error", base[base["snv"]])):
                if not len(s):
                    continue
                q = s["functional_pathogenicity"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                rows.append({"gene": gene, "stratum": stratum, "scope": scope, "n": int(len(s)),
                             "functional_pathogenicity_quantiles":
                                 dict(zip(["q05", "q25", "q50", "q75", "q95"], map(float, q))),
                             "deposit_classes": {k: int(v) for k, v in
                                                 s["cls"].value_counts().items()}})
    return rows


def jsonable(obj):
    """NaN as null, numpy scalars as Python ones, recursively."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ==========================================================================
# command line
# ==========================================================================
COLUMNS = ["group", "gene", "stratum", "model", "scope", "n", "method", "rho", "rep_rho",
           "reliability", "reliability_unclipped", "ceiling", "realisation", "v5_method",
           "v5_ceiling", "ceiling_minus_v5", "realisation_v5_ceiling", "seed", "bootstrap_B",
           "rho_lo", "rho_hi", "reliability_lo", "reliability_hi", "ceiling_lo", "ceiling_hi",
           "realisation_lo", "realisation_hi", "share_realisation_gt_1",
           "bootstrap_realisation_undefined"]     # A2's table (matched_ceilings_v1)


# ==========================================================================
# the current generation
# ==========================================================================
COLUMNS_V2 = ["group", "gene", "stratum", "model", "population", "n", "method", "rho",
              "rep_rho", "reliability", "reliability_unclipped", "ceiling", "realisation",
              "estimable", "selection_overlap", "ceiling_inflated", "breakeven_inflation",
              "seed", "bootstrap_B", "rho_lo", "rho_hi", "reliability_lo", "reliability_hi",
              "ceiling_lo", "ceiling_hi", "realisation_lo", "realisation_hi",
              "share_realisation_gt_1", "bootstrap_realisation_undefined"]
GROUPS_V2 = {"core": ["core"], "external": ["external"], "all": ["core", "external"]}
BEST_STRATA = ("splice_3_10", "splice_11_50")
POINT = ["n", "rho", "reliability", "ceiling", "realisation"]
INTERVAL = ["rho_lo", "rho_hi", "reliability_lo", "reliability_hi", "ceiling_lo", "ceiling_hi",
            "realisation_lo", "realisation_hi", "share_realisation_gt_1"]
REGRESSION_TOL = 1e-12


def finish(raw: pd.DataFrame, inflation: dict[str, float]) -> pd.DataFrame:
    """The cells as computed, with the population, estimable, the flag, the inflated
    ceiling and the breakeven factor."""
    t = raw.rename(columns={"scope": "population"}).copy()
    t["estimable"] = t["ceiling_lo"] > 0
    t["selection_overlap"] = [predictors.is_overlap(m, g) for m, g in zip(t["model"], t["gene"])]
    f = t["gene"].map(inflation)
    core_se = (t["group"] == "core") & (t["method"] == "se")
    with np.errstate(invalid="ignore"):
        rel = np.clip(1 - f * (1 - t["reliability_unclipped"]), 0.0, 1.0)
    t["ceiling_inflated"] = np.where(core_se & f.notna(), np.sqrt(rel), np.nan)
    se = t["method"] == "se"
    room = 1 - t["reliability_unclipped"]
    with np.errstate(invalid="ignore", divide="ignore"):
        t["breakeven_inflation"] = np.where(se & (room > 0), (1 - t["rho"] ** 2) / room, np.nan)
    return t[COLUMNS_V2]


def layer_ceilings() -> pd.DataFrame:
    """The layer ceilings of the main rows (results/reliability_v2.tsv and
    results/external_reliability_v2.tsv: the second generation's), by group."""
    core = pd.read_csv(RESULTS / "reliability_v2.tsv", sep="\t").assign(group="core")
    ext = pd.read_csv(RESULTS / "external_reliability_v2.tsv", sep="\t").assign(group="external")
    t = pd.concat([core, ext], ignore_index=True)
    t = t[t["role"] == "main"]
    return t[["group", "gene", "population", "stratum", "method", "n", "ceiling", "ceiling_lo",
              "ceiling_hi", "estimable"]].reset_index(drop=True)


def variant_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per gene x stratum x population, the variants with a functional score and (for a
    gene with an error model) its error information."""
    rows = []
    for (gene, pop), g in df.assign(pop=np.where(df["snv"], population.SNV,
                                                 population.NON_SNV)).groupby(["gene", "pop"]):
        for stratum in STRATA:
            s = g[_stratum_mask(g["region"], stratum) & g["has_error"]
                  & g["functional_pathogenicity"].notna()]
            if len(s):
                rows.append({"group": g["group"].iloc[0], "gene": gene, "population": pop,
                             "stratum": stratum, "n_variants": int(len(s))})
    return pd.DataFrame(rows)


def _cell_record(r: pd.Series) -> dict:
    return {k: r[k] for k in ("model", "n", "rho", "rho_lo", "rho_hi", "ceiling", "ceiling_lo",
                              "ceiling_hi", "estimable", "realisation", "realisation_lo",
                              "realisation_hi", "breakeven_inflation")}


def by_stratum_v2(table: pd.DataFrame, layer: pd.DataFrame) -> dict:
    """1: per stratum and group, the estimable layer ceilings and the cross-gene medians."""
    t = table[(table["population"] == population.SNV) & ~table["selection_overlap"]]
    lay = layer[layer["population"] == population.SNV]
    out = {}
    for stratum in STRATA:
        rec = {}
        for name, groups in GROUPS_V2.items():
            L = lay[(lay["stratum"] == stratum) & lay["group"].isin(groups)]
            E = L[L["estimable"]]
            d = t[(t["stratum"] == stratum) & t["group"].isin(groups)]
            if not len(L) and not len(d):
                continue
            per = {}
            for model, m in d.groupby("model", sort=False):
                per[model] = {"rho": _range(m["rho"]),
                              "realisation": _range(m.loc[m["estimable"], "realisation"])}
            rec[name] = {"genes_with_layer_ceiling": sorted(L["gene"]),
                         "genes_estimable": sorted(E["gene"]),
                         "n_genes_estimable": int(len(E)), "ceiling": _range(E["ceiling"]),
                         "per_predictor": per}
        if rec:
            out[stratum] = rec
    return out


def by_gene_stratum(table: pd.DataFrame, layer: pd.DataFrame, counts: pd.DataFrame) -> list[dict]:
    """2: per gene x stratum (snv), n, the layer ceiling, the best predictor by rho and
    the splice family's best (a flagged cell is not a candidate)."""
    t = table[table["population"] == population.SNV]
    lay = layer[layer["population"] == population.SNV].set_index(["gene", "stratum"])
    cnt = counts[counts["population"] == population.SNV].set_index(["gene", "stratum"])
    out = []
    for (group, gene, stratum), d in t.groupby(["group", "gene", "stratum"], sort=False):
        elig = d[~d["selection_overlap"] & d["rho"].notna()]
        fam = elig[elig["model"].isin(predictors.SPLICE_FAMILY)]
        L = lay.loc[(gene, stratum)] if (gene, stratum) in lay.index else None
        out.append({
            "group": group, "gene": gene, "stratum": stratum,
            "n_variants": int(cnt.loc[(gene, stratum), "n_variants"])
            if (gene, stratum) in cnt.index else None,
            "layer_ceiling": None if L is None else {
                "n": int(L["n"]), "method": L["method"], "ceiling": L["ceiling"],
                "ceiling_lo": L["ceiling_lo"], "ceiling_hi": L["ceiling_hi"],
                "estimable": bool(L["estimable"])},
            "best": _cell_record(elig.loc[elig["rho"].idxmax()]) if len(elig) else None,
            "best_splice_family": _cell_record(fam.loc[fam["rho"].idxmax()]) if len(fam) else None})
    return out


def best_realisation(per_gene: list[dict]) -> dict:
    """3: at 3-10 and 11-50 bp, the best predictor's realisation per gene (estimable cells),
    and the genes where its interval's upper bound is below 1, and below 0.5."""
    out = {}
    for stratum in BEST_STRATA:
        rows = [r for r in per_gene if r["stratum"] == stratum and r["best"]
                and r["best"]["estimable"]]
        out[stratum] = {
            "cells": [{"gene": r["gene"], **r["best"]} for r in rows],
            "upper_bound_below_1": [f"{r['gene']} {r['best']['model']}" for r in rows
                                    if r["best"]["realisation_hi"] < 1],
            "upper_bound_below_0.5": [f"{r['gene']} {r['best']['model']}" for r in rows
                                      if r["best"]["realisation_hi"] < 0.5],
            "not_estimable": [r["gene"] for r in per_gene if r["stratum"] == stratum
                              and r["best"] and not r["best"]["estimable"]]}
    return out


def deep(layer: pd.DataFrame) -> list[dict]:
    """4: beyond 50 bp, the layer ceiling per gene (snv)."""
    d = layer[(layer["population"] == population.SNV) & (layer["stratum"] == "splice_deep")]
    return d[["group", "gene", "method", "n", "ceiling", "ceiling_lo", "ceiling_hi",
              "estimable"]].to_dict("records")


def above_one(table: pd.DataFrame) -> dict:
    """5: the estimable snv cells whose realisation exceeds 1 or whose interval crosses 1."""
    t = table[table["population"] == population.SNV]
    hit = (t["realisation"] > 1) | (t["realisation_hi"] > 1)
    keep = ["group", "gene", "stratum", "model", "n", "rho", "ceiling", "ceiling_lo",
            "realisation", "realisation_lo", "realisation_hi", "share_realisation_gt_1",
            "selection_overlap"]
    return {"cells": t[hit & t["estimable"]][keep].to_dict("records"),
            "not_estimable_cells_left_out": int((hit & ~t["estimable"]).sum())}


def regression_v1(table: pd.DataFrame, v1_path: Path | None = None) -> dict:
    """A2's matched_ceilings_v1.tsv (scope snv) against this table (population snv): the
    core three genes and the eighteen predictors whose definition did not change, the
    external five genes and their nine columns (alphagenome_official as alphagenome)."""
    v1 = pd.read_csv(v1_path or RESULTS / "matched_ceilings_v1.tsv", sep="\t",
                     float_precision="round_trip")
    v1 = v1[v1["scope"] == "snv"].copy()
    v1["model"] = v1["model"].replace({"alphagenome_official": "alphagenome"})
    v1 = v1[~((v1["group"] == "core") & (v1["model"] == "alphagenome"))]
    v2 = table[table["population"] == population.SNV].set_index(["group", "gene", "stratum",
                                                                  "model"])
    rec = {"rows_compared": 0, "missing_here": [], "point_identical": 0, "point_differ": [],
           "point_max_abs_diff": 0.0, "interval_identical": 0, "interval_differ": []}
    for r in v1.itertuples(index=False):
        key = (r.group, r.gene, r.stratum, r.model)
        if key not in v2.index:
            rec["missing_here"].append(list(key))
            continue
        rec["rows_compared"] += 1
        o = v2.loc[key]
        diffs = []
        for c in POINT:
            a, b = float(getattr(r, c)), float(o[c])
            if not ((np.isnan(a) and np.isnan(b)) or
                    np.float64(a).view(np.uint64) == np.float64(b).view(np.uint64)):
                diffs.append(c)
                rec["point_max_abs_diff"] = max(rec["point_max_abs_diff"], abs(a - b))
        if diffs:
            rec["point_differ"].append(list(key) + [diffs])
        else:
            rec["point_identical"] += 1
        idiff = [c for c in INTERVAL if not (
            (np.isnan(float(getattr(r, c))) and np.isnan(float(o[c]))) or
            np.float64(getattr(r, c)).view(np.uint64) == np.float64(o[c]).view(np.uint64))]
        if idiff:
            rec["interval_differ"].append({"cell": list(key), "columns": idiff,
                                           "seed_v1": int(r.seed), "seed_here": int(o["seed"])})
        else:
            rec["interval_identical"] += 1
    rec["interval_differ_with_same_seed"] = [d for d in rec["interval_differ"]
                                             if d["seed_v1"] == d["seed_here"]]
    rec["holds"] = bool(not rec["missing_here"] and rec["point_max_abs_diff"] <= REGRESSION_TOL
                        and not rec["interval_differ_with_same_seed"])
    rec["point_bitwise"] = not rec["point_differ"]
    return rec


# ==========================================================================
# the third generation: the ceiling on the rank scale (A4b.4)
# ==========================================================================
V2_RENAMED = ["reliability", "reliability_unclipped", "ceiling", "realisation", "estimable",
              "reliability_lo", "reliability_hi", "ceiling_lo", "ceiling_hi", "realisation_lo",
              "realisation_hi", "share_realisation_gt_1", "bootstrap_realisation_undefined",
              "seed", "bootstrap_B"]
V2_DROPPED = ["ceiling_inflated", "breakeven_inflation"]
COVERAGE_MIN = 0.9          # G17: "the highest rho" among predictors scoring 90% of the cell
TABLE_V2 = RESULTS / "matched_ceilings_v2.tsv"


def columns_v3() -> list[str]:
    from atlas import rank_cells
    return (["group", "gene", "stratum", "model", "population", "n", "method", "rho",
             "rho_lo", "rho_hi", "rep_rho", "selection_overlap", "set_sha"]
            + [f"{c}_v2" for c in V2_RENAMED] + ["ceiling_replicate_k1", "lof_share",
                                                  "neutral_in_range"]
            + [c for m in rank_cells.MODELS
               for c in rank_cells.model_columns(m) + rank_cells.realisation_columns(m)]
            + ["pooled_scale_own"])


def g18(tasks: list[tuple]) -> tuple[list[tuple], int]:
    """The cells less the external genes' all rows."""
    keep = [t for t in tasks if not (t[0][0] == "external" and t[0][2] == "all")]
    return keep, len(tasks) - len(keep)


def check_v2(t2: pd.DataFrame, path: Path = TABLE_V2) -> dict:
    """The second generation computed again against its file (less the external all rows):
    every column, bit for bit (regressions 1 and 2)."""
    from atlas.reliability import _same
    f = pd.read_csv(path, sep="\t", float_precision="round_trip")
    dropped = (f["group"] == "external") & (f["stratum"] == "all")
    f = f[~dropped]
    keys = ["group", "gene", "stratum", "model", "population"]
    a = t2.sort_values(keys, kind="stable").reset_index(drop=True)
    b = f.sort_values(keys, kind="stable").reset_index(drop=True)
    differ = [c for c in COLUMNS_V2 if not _same(a[c], b[c])]
    return {"file": path.name, "rows": int(len(b)), "rows_here": int(len(a)),
            "external_all_rows_dropped": int(dropped.sum()), "columns_differing": differ,
            "holds": bool(len(a) == len(b) and not differ)}


def rank_tasks(tasks: list[tuple]) -> tuple[list[tuple], list[tuple]]:
    """One task per distinct variant set of the genes with an error model, carrying the
    predictors of every cell on that set; and per task its set's key."""
    from atlas import rank_cells
    sets: dict[tuple, dict] = {}
    for key, method, arrays in tasks:
        group, gene, stratum, model, pop = key
        if gene not in rank_cells.GENES:
            continue
        sha = hashlib.sha256("\n".join(sorted(arrays["ids"])).encode()).hexdigest()
        k = (group, gene, stratum, pop, sha)
        rec = sets.setdefault(k, {"arrays": arrays, "method": method, "predictors": {}})
        rec["predictors"][model] = arrays["x"]
    out, keys = [], []
    for k, rec in sets.items():
        group, gene, stratum, pop, sha = k
        a = rec["arrays"]
        d = (((-a["r1"]) - (-a["r2"])) / 2) if rec["method"] == "replicates" else None
        out.append(rank_cells.task(group, gene, stratum, pop, a["ids"], a["y"],
                                   se=a.get("se"), d=d, predictors=rec["predictors"]))
        keys.append(k)
    return out, keys


def table_v3(t2: pd.DataFrame, tasks: list[tuple], workers: int = 1) -> pd.DataFrame:
    """v2's cells renamed, with the set's sha and both models' rank-scale columns."""
    from atlas import rank_cells
    t = t2.rename(columns={c: f"{c}_v2" for c in V2_RENAMED}).drop(columns=V2_DROPPED)
    t["ceiling_replicate_k1"] = np.where(t["method"] == "replicates",
                                         np.sqrt(np.clip(t["rep_rho"], 0.0, None)), np.nan)
    sha = {key: hashlib.sha256("\n".join(sorted(a["ids"])).encode()).hexdigest()
           for key, _, a in tasks}
    keys = ["group", "gene", "stratum", "model", "population"]
    t["set_sha"] = [sha[tuple(r)][:16] for r in t[keys].itertuples(index=False)]
    rt, rkeys = rank_tasks(tasks)
    res = dict(zip(rkeys, rank_cells.compute(rt, workers)))
    extra = []
    for r in t.itertuples(index=False):
        k = (r.group, r.gene, r.stratum, r.population, sha[(r.group, r.gene, r.stratum, r.model,
                                                           r.population)])
        if k in res:
            extra.append(rank_cells.columns(res[k]) | rank_cells.realisation(res[k], r.model,
                                                                             r.rho))
        else:
            extra.append({})
    e = pd.DataFrame(extra, index=t.index).drop(columns=["set_sha"], errors="ignore")
    t = t.join(e)
    return t.reindex(columns=columns_v3())


def layer_ceilings_v3(layer_dir: Path = RESULTS) -> pd.DataFrame:
    """The layer tables' main rows with a rank-scale ceiling (results/reliability_v3.tsv,
    results/external_reliability_v3.tsv), by group."""
    core = pd.read_csv(Path(layer_dir) / paths.name("reliability"), sep="\t",
                       float_precision="round_trip").assign(group="core")
    ext = pd.read_csv(Path(layer_dir) / paths.name("external_reliability"), sep="\t",
                      float_precision="round_trip").assign(group="external")
    t = pd.concat([core, ext], ignore_index=True)
    return t[(t["role"] == "main") & t["ceiling_neutral"].notna()].reset_index(drop=True)


def check_layer(table: pd.DataFrame, layer: pd.DataFrame) -> dict:
    """Regression 3: one set and one model, the same ceiling, interval and floor in the
    layer and the matched tables, bit for bit."""
    from atlas import rank_cells
    from atlas.reliability import _same
    keys = ["group", "gene", "stratum", "population", "set_sha"]
    m = table.dropna(subset=["ceiling_neutral"]).merge(layer, on=keys, suffixes=("", "_layer"))
    cols = [c for mm in rank_cells.MODELS for c in rank_cells.model_columns(mm)]
    differ = [c for c in cols if not _same(m[c], m[f"{c}_layer"])]
    return {"matched_cells_on_a_layer_set": int(len(m)),
            "layer_sets_matched": int(m[keys].drop_duplicates().shape[0]),
            "columns": cols, "columns_differing": differ, "holds": not differ}


def _usable(t: pd.DataFrame, model: str) -> pd.Series:
    from atlas import rank_cells
    return rank_cells.usable(t, model)


def _cell_v3(r: pd.Series, model: str) -> dict:
    from atlas import rank_cells
    out = {k: r[k] for k in ("model", "n", "rho", "rho_lo", "rho_hi", f"ceiling_{model}",
                             f"ceiling_{model}_lo", f"ceiling_{model}_hi", f"floor_{model}",
                             f"estimable_{model}", "lof_share", "neutral_in_range",
                             "realisation_v2", "estimable_v2")}
    for m in rank_cells.MODELS:
        out |= {c: r[c] for c in rank_cells.realisation_columns(m)}
    return out


def by_stratum_v3(table: pd.DataFrame, layer: pd.DataFrame, model: str) -> dict:
    """1: per stratum and group, the layer ceilings that enter and the cross-gene medians."""
    t = table[(table["population"] == population.SNV) & ~table["selection_overlap"]]
    t = t[t["gene"].isin(set(layer["gene"]))]
    lay = layer[layer["population"] == population.SNV]
    out = {}
    for stratum in STRATA:
        rec = {}
        for name, groups in GROUPS_V2.items():
            L = lay[(lay["stratum"] == stratum) & lay["group"].isin(groups)]
            E = L[_usable(L, model)]
            d = t[(t["stratum"] == stratum) & t["group"].isin(groups)]
            if not len(L) and not len(d):
                continue
            per = {}
            for pred, m in d.groupby("model", sort=False):
                per[pred] = {"rho": _range(m["rho"]),
                             "realisation": _range(m.loc[_usable(m, model), f"realisation_{model}"])}
            rec[name] = {"genes_with_layer_ceiling": sorted(L["gene"]),
                         "genes_entering": sorted(E["gene"]), "n_genes_entering": int(len(E)),
                         "genes_not_estimable": sorted(L.loc[~L[f"estimable_{model}"].astype(bool), "gene"]),
                         "genes_out_of_neutral_range": sorted(L.loc[~L["neutral_in_range"].astype(bool), "gene"]),
                         "ceiling": _range(E[f"ceiling_{model}"]), "per_predictor": per}
        if rec:
            out[stratum] = rec
    return out


def by_gene_stratum_v3(table: pd.DataFrame, layer: pd.DataFrame, counts: pd.DataFrame,
                       model: str) -> list[dict]:
    """2: per gene x stratum (snv): n, the layer ceilings, the best predictor by rho and
    the splice family's best, among those scoring 90% of the cell's variants (G17)."""
    from atlas import rank_cells
    t = table[(table["population"] == population.SNV) & table["gene"].isin(set(layer["gene"]))]
    lay = layer[layer["population"] == population.SNV].set_index(["gene", "stratum"])
    cnt = counts[counts["population"] == population.SNV].set_index(["gene", "stratum"])
    out = []
    for (group, gene, stratum), d in t.groupby(["group", "gene", "stratum"], sort=False):
        nv = int(cnt.loc[(gene, stratum), "n_variants"]) if (gene, stratum) in cnt.index else None
        cover = d["n"] / nv if nv else pd.Series(np.nan, index=d.index)
        elig = d[~d["selection_overlap"] & d["rho"].notna() & (cover >= COVERAGE_MIN)]
        fam = elig[elig["model"].isin(predictors.SPLICE_FAMILY)]
        L = lay.loc[(gene, stratum)] if (gene, stratum) in lay.index else None
        layer_rec = None
        if L is not None:
            layer_rec = {"n": int(L["n"]), "ceiling_v2": L["ceiling_v2"],
                         "lof_share": L["lof_share"], "neutral_in_range": bool(L["neutral_in_range"])}
            for m in rank_cells.MODELS:
                layer_rec |= {f"ceiling_{m}": L[f"ceiling_{m}"], f"ceiling_{m}_lo": L[f"ceiling_{m}_lo"],
                              f"ceiling_{m}_hi": L[f"ceiling_{m}_hi"], f"floor_{m}": L[f"floor_{m}"],
                              f"estimable_{m}": bool(L[f"estimable_{m}"])}
        out.append({"group": group, "gene": gene, "stratum": stratum, "n_variants": nv,
                    "predictors_scoring_90pct": sorted(elig["model"]),
                    "layer_ceiling": layer_rec,
                    "best": _cell_v3(elig.loc[elig["rho"].idxmax()], model) if len(elig) else None,
                    "best_splice_family": (_cell_v3(fam.loc[fam["rho"].idxmax()], model)
                                           if len(fam) else None)})
    return out


def best_realisation_v3(per_gene: list[dict], model: str) -> dict:
    """3: at 3-10 and 11-50 bp, the best predictor's realisation per gene, where its cell's
    ceiling enters; the genes where the interval's upper bound is below 1, and below 0.5."""
    def enters(b):
        return bool(b[f"estimable_{model}"]) and (model != "neutral" or bool(b["neutral_in_range"]))
    out = {}
    for stratum in BEST_STRATA:
        rows = [r for r in per_gene if r["stratum"] == stratum and r["best"]]
        ok = [r for r in rows if enters(r["best"])]
        hi = f"realisation_{model}_hi"
        out[stratum] = {
            "cells": [{"gene": r["gene"], **r["best"]} for r in ok],
            "upper_bound_below_1": [f"{r['gene']} {r['best']['model']}" for r in ok
                                    if r["best"][hi] < 1],
            "upper_bound_below_0.5": [f"{r['gene']} {r['best']['model']}" for r in ok
                                      if r["best"][hi] < 0.5],
            "not_entering": [r["gene"] for r in rows if not enters(r["best"])]}
    return out


def deep_v3(layer: pd.DataFrame, model: str) -> list[dict]:
    """4: beyond 50 bp, the layer ceilings per gene (snv)."""
    d = layer[(layer["population"] == population.SNV) & (layer["stratum"] == "splice_deep")]
    cols = ["group", "gene", "n", "ceiling_v2", f"ceiling_{model}", f"ceiling_{model}_lo",
            f"ceiling_{model}_hi", f"floor_{model}", f"estimable_{model}", "lof_share",
            "neutral_in_range"]
    return d[cols].to_dict("records")


def above_one_v3(table: pd.DataFrame, model: str) -> dict:
    """5: the snv cells whose ceiling enters and whose realisation exceeds 1 or whose
    interval crosses 1."""
    t = table[(table["population"] == population.SNV) & table[f"ceiling_{model}"].notna()]
    hit = (t[f"realisation_{model}"] > 1) | (t[f"realisation_{model}_hi"] > 1)
    ok = _usable(t, model)
    keep = ["group", "gene", "stratum", "model", "n", "rho", f"ceiling_{model}",
            f"ceiling_{model}_lo", f"floor_{model}", f"realisation_{model}",
            f"realisation_{model}_lo", f"realisation_{model}_hi",
            f"share_realisation_gt_1_{model}", "selection_overlap"]
    return {"cells": t[hit & ok][keep].to_dict("records"),
            "cells_left_out_not_entering": int((hit & ~ok).sum())}


def v2_to_v3(per_gene: list[dict], model: str) -> list[dict]:
    """6: per gene x stratum (snv), ceiling_v2 beside both rank-scale ceilings; the splice
    family's best predictor's three realisations; the first supplement's trial."""
    from atlas import rank_cells
    out = []
    for r in per_gene:
        L, f = r["layer_ceiling"], r["best_splice_family"]
        if L is None:
            continue
        rec = {"group": r["group"], "gene": r["gene"], "stratum": r["stratum"], "n": L["n"],
               "ceiling_v2": L["ceiling_v2"], "ceiling_neutral": L["ceiling_neutral"],
               "ceiling_own": L["ceiling_own"]}
        trial = rank_cells.TRIAL.get((r["gene"], r["stratum"]))
        if trial is not None:
            want = trial[1] if model == "neutral" else trial[0]
            near = (r["gene"], r["stratum"], model) in rank_cells.TRIAL_NEAR_FLOOR
            rec |= {"trial": want, "difference": L[f"ceiling_{model}"] - want,
                    "trial_near_the_floor": near,
                    "within_tolerance": None if near else
                    bool(abs(L[f"ceiling_{model}"] - want) <= rank_cells.TRIAL_TOL)}
        if f is not None:
            rec |= {"splice_family_best": f["model"], "realisation_v2": f["realisation_v2"],
                    "realisation_neutral": f["realisation_neutral"],
                    "realisation_own": f["realisation_own"]}
        out.append(rec)
    return out


def range_by_stratum(table: pd.DataFrame, layer: pd.DataFrame) -> dict:
    """Per population and stratum, the cells in and out of the neutral model's range."""
    from atlas import rank_cells
    return {pop: {"layer": rank_cells.range_counts(layer[layer["population"] == pop], STRATA),
                  "matched": rank_cells.range_counts(table[table["population"] == pop], STRATA)}
            for pop in POPULATIONS}


def summary_v3(table: pd.DataFrame, layer: pd.DataFrame, counts: pd.DataFrame) -> dict:
    from atlas import rank_cells
    out = {}
    for m in rank_cells.MODELS:
        per_gene = by_gene_stratum_v3(table, layer, counts, m)
        out[m] = {"1_by_stratum": by_stratum_v3(table, layer, m),
                  "2_by_gene_stratum": per_gene,
                  "3_best_realisation_3_10_and_11_50": best_realisation_v3(per_gene, m),
                  "4_beyond_50bp": deep_v3(layer, m),
                  "5_realisation_above_1_or_interval_crossing_1": above_one_v3(table, m),
                  "6_v2_to_v3": v2_to_v3(per_gene, m)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--layer-dir", default=str(RESULTS),
                    help="where the layer tables of this generation are read from")
    args = ap.parse_args(argv)
    import time
    from atlas import rank_cells
    from atlas.reliability import inflation_factors, neutral_check
    t0 = time.time()
    core, ext = load_core(), load_external()
    inflation = inflation_factors(neutral_check(population.load(None),
                                                pd.read_parquet(paths.path("assay_aux"))))
    tasks = (cell_tasks(core, MODELS, STRATA, scopes=POPULATIONS)
             + cell_tasks(ext, EXTERNAL_MODELS, STRATA, scopes=POPULATIONS))
    tasks, n_g18 = g18(tasks)
    print(f"[matched] {len(tasks)} cells ({n_g18} external all rows dropped), {B} resamples "
          f"each, {args.workers} workers", flush=True)
    t2 = finish(compute(tasks, args.workers), inflation)
    t1 = time.time()
    reg_v2 = check_v2(t2)
    reg_v1 = regression_v1(t2)
    out = Path(args.out)
    if not reg_v2["holds"]:
        print("the second generation is not computed again:",
              json.dumps(jsonable(reg_v2), indent=1), file=sys.stderr)
        return 1
    print(f"[matched] the second generation computed again in {t1 - t0:.0f} s: holds", flush=True)
    table = table_v3(t2, tasks, args.workers)
    t3 = time.time()
    layer = layer_ceilings_v3(Path(args.layer_dir))
    reg_layer = check_layer(table, layer)
    counts = variant_counts(pd.concat([core, ext], ignore_index=True))
    summary = {"definition": {"seed": SEED, "bootstrap_B_v2": B, "min_n": MIN_N,
                              "populations": list(POPULATIONS), "core": CORE,
                              "external_genes": list(EXTERNAL_GENES),
                              "predictors_core": MODELS, "predictors_external": EXTERNAL_MODELS,
                              "cross_gene_population": population.SNV,
                              "selection_overlap": predictors.SELECTION_OVERLAP,
                              "g18": "the external genes' all rows are not written",
                              "g17": f"the highest rho among the predictors scoring at least "
                                     f"{COVERAGE_MIN:.0%} of the cell's variants (gene x stratum "
                                     "x population with a functional score and the error "
                                     "information)",
                              "v2_columns": {f"{c}_v2": c for c in V2_RENAMED},
                              "dropped": V2_DROPPED,
                              "models": list(rank_cells.MODELS),
                              "enters_a_summary": "a cell's ceiling or realisation under a model "
                                                  "enters when the ceiling is estimable and, "
                                                  "for neutral, in range",
                              "neutral_range": f"lof_share <= {rank_cells.LOF_SHARE_MAX}",
                              "layer_ceilings": [paths.name("reliability"),
                                                 paths.name("external_reliability")]},
               "checks": {"v2_computed_again": reg_v2,
                          "same_set_same_ceiling_in_the_layer_tables": reg_layer,
                          "regression_matched_ceilings_v1": reg_v1},
               "neutral_range_by_stratum": range_by_stratum(table, layer),
               "by_model": summary_v3(table, layer, counts),
               "seconds": {"v2_computed_again": round(t1 - t0, 1),
                           "rank_scale": round(t3 - t1, 1), "workers": args.workers}}
    if not reg_layer["holds"]:
        print("a set's ceiling differs between the tables:",
              json.dumps(jsonable(reg_layer), indent=1), file=sys.stderr)
        (out / "_scratch").mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "_scratch" / "matched_ceilings_v3_failed.tsv", sep="\t", index=False)
        return 1
    table.to_csv(out / paths.name("matched_ceilings"), sep="\t", index=False)
    (out / paths.name("matched_ceilings_summary")).write_text(
        json.dumps(jsonable(summary), indent=1) + "\n")
    print(f"[matched] wrote {len(table)} cells; v2 computed again holds, the layer sets agree "
          f"({reg_layer['matched_cells_on_a_layer_set']} cells); rank scale {t3 - t1:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
