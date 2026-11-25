"""Measurement reliability of the functional standard, and the attenuation
ceiling it imposes on any predictor: the layer ceilings.

A Spearman rho between a predictor and a noisy measurement is bounded above by the
square root of that measurement's reliability. This module estimates the bound per
gene x stratum x population (``atlas.population``) from the assay's own error model,
on every variant of the stratum that carries the error information. The ceiling
matched to each predictor's own variants, and the realisation, are
``atlas.matched_ceilings``'s.

Reliability estimators (per gene, per stratum, per population)
--------------------------------------------------------------
``replicates``  BRCA1: two independent replicate scores whose mean *is* the deposited
                score. Spearman rho between replicates gives the reliability of a
                single replicate; Spearman-Brown steps it up to the 2-replicate mean:
                ``rel = 2r / (1 + r)``.
``se``          BARD1, PALB2 and BAP1: per-variant standard errors checked against
                another quantity the deposit publishes (atlas.assay_aux: the 95% CI for
                BARD1 and PALB2, the test statistic for BAP1). Classical decomposition:
                ``rel = 1 - mean(SE^2) / var(functional_pathogenicity)``, clipped to
                [0, 1]; the value before clipping is kept.

BRCA1 also carries the SE of its replicate mean; that row is kept as a cross-check
(role ``cross_check``), not as the gene's ceiling. RAD51C (no error on the score's
scale), BRCA2 and VHL (no error information) are not estimated.

Intervals: B = 2,000 resamples of the variants (BRCA1's replicate pairs as variants),
the seed from atlas.matched_ceilings.SEED and the cell's key (group, gene, stratum,
method, population); ``estimable`` when the ceiling's lower bound is above 0.

The SE's adequacy (``neutral_check``): per gene, the synonymous SNVs, and those of them
whose SpliceAI score is below 0.1; the variance of the functional score against the
mean SE^2. A ratio near 1 means the SE accounts for the whole spread of the neutral
variants; above 1, there is noise the SE does not capture (or real effects among
them). ``inflation_factor`` = max(1, the second batch's ratio); for the genes estimated
by SE the ceiling is computed again with SE^2 times that factor (``ceiling_inflated``).

Outputs: results/reliability_v2.tsv (one row per gene x stratum x population x
estimator), results/reliability_v2.json (the summary), results/reliability_corroboration_v2.tsv
(VHL's second condition, BAP1's nested time points: bounds, as in v5) and
results/reliability_neutral_check_v1.tsv.

Usage (PYTHONPATH=src):  python -m atlas.reliability [--workers N]
"""

from __future__ import annotations

import argparse
import sys
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import paths, population, predictors
from atlas.evaluate import classify_region

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

STRATA = ["all", "coding_or_utr", "splice_region",
          "splice_1_2", "splice_3_10", "splice_11_50", "splice_deep"]

MODELS = predictors.PREDICTORS

MIN_N = 30

# Where one gene x stratum has both estimates validated, the replicate estimate is the
# primary one. An explicit order: sorting the method names happens to put "se" first.
METHOD_PREFERENCE = {"replicates": 0, "se": 1}


def spearman_brown(r: float, k: int = 2) -> float:
    """Reliability of a k-fold mean given the reliability of one measurement."""
    if not np.isfinite(r) or (1 + (k - 1) * r) == 0:
        return float("nan")
    return float(k * r / (1 + (k - 1) * r))


def _stratum_mask(region: pd.Series, stratum: str) -> pd.Series:
    if stratum == "all":
        return pd.Series(True, index=region.index)
    if stratum == "splice_region":
        return region.str.startswith("splice_")
    return region == stratum


# the core genes' error models (atlas.assay_aux) and the genes not estimated
CORE_METHOD = {"BRCA1": "replicates", "BARD1": "se", "PALB2": "se", "BAP1": "se"}
SE_GENES = [g for g, m in CORE_METHOD.items() if m == "se"]
NOT_ESTIMATED = {
    "RAD51C": ("the deposit carries no error on the score's scale: its score is "
               "processed_adj_lfc_D4_D14 / SE_bind_D4_D14, a z-score (atlas.assay_aux, "
               "scale_mismatch)"),
    "BRCA2": "the deposit carries the score only",
    "VHL": ("no per-variant error and no replicates; its second selection condition "
            "bounds the reliability (reliability_corroboration)"),
}
USABLE_SE = ("validated", "derived_from_replicates")
B = 2000
TABLE_COLUMNS = ["gene", "urn", "population", "stratum", "method", "k", "n", "rep_rho",
                 "score_var", "mean_se2", "reliability_unclipped", "reliability", "ceiling",
                 "status", "role", "seed", "bootstrap_B", "reliability_lo", "reliability_hi",
                 "ceiling_lo", "ceiling_hi", "estimable", "inflation_factor",
                 "ceiling_inflated"]


def layer_seed(group: str, gene: str, stratum: str, method: str, pop: str) -> int:
    """The seed of a layer-ceiling cell: atlas.matched_ceilings.SEED and the cell's key,
    as A2 composes it."""
    from atlas.matched_ceilings import cell_seed
    return cell_seed((group, gene, stratum, method, pop))


def bootstrap_layer(method: str, seed: int, y=None, se=None, reps=None, k: int = 2,
                    b: int = B, budget: int = 4_000_000) -> dict:
    """The 2.5% and 97.5% quantiles of the reliability and the ceiling over b resamples of
    the variants (with replacement). replicates: the mean pairwise Spearman r of the
    replicate arrays, raised to a k-fold mean by Spearman-Brown; se: 1 - mean(SE^2) /
    var(y), clipped to [0, 1]."""
    from atlas.matched_ceilings import spearman_rows
    n = len(y) if y is not None else len(reps[0])
    rng = np.random.default_rng(seed)
    chunk = max(1, min(b, budget // max(n, 1)))
    parts, done = [], 0
    while done < b:
        kk = min(chunk, b - done)
        idx = rng.integers(0, n, size=(kk, n))
        if method == "replicates":
            r = np.mean([spearman_rows(reps[i][idx], reps[j][idx])
                         for i, j in combinations(range(len(reps)), 2)], axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.where(1 + (k - 1) * r != 0, k * r / (1 + (k - 1) * r), np.nan)
        else:
            yy = y[idx]
            var = yy.var(axis=1, ddof=1)
            err = (se[idx] ** 2).mean(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                raw = np.where(var > 0, 1 - err / var, np.nan)
            rel = np.clip(raw, 0.0, 1.0)
        parts.append(rel)
        done += kk
    rel = np.concatenate(parts)
    ceiling = np.sqrt(np.clip(rel, 0.0, None))
    out = {}
    for name, v in (("reliability", rel), ("ceiling", ceiling)):
        ok = v[np.isfinite(v)]
        out[f"{name}_lo"] = float(np.quantile(ok, 0.025)) if len(ok) else np.nan
        out[f"{name}_hi"] = float(np.quantile(ok, 0.975)) if len(ok) else np.nan
    return out


def inflated_ceiling(err: float, var: float, factor: float | None) -> float:
    """The ceiling with SE^2 multiplied by factor (clipped as the estimate is)."""
    if factor is None or not np.isfinite(factor) or not var > 0:
        return np.nan
    return float(np.sqrt(np.clip(1 - factor * err / var, 0.0, 1.0)))


def reliability_table(mat: pd.DataFrame, aux: pd.DataFrame,
                      inflation: dict[str, float] | None = None, b: int = B) -> pd.DataFrame:
    """Per gene x stratum x population: the replicate and the SE estimates, on every
    variant of the stratum that carries the error information, with intervals."""
    inflation = inflation or {}
    keep = ["variant_id", "gene", "fit_rep1", "fit_rep2", "se", "se_status"]
    df = population.select(mat, None).merge(aux[keep], on=["variant_id", "gene"], how="left")
    rows = []
    for pop in population.POPULATIONS:
        for gene, g in df[df["population"] == pop].groupby("gene"):
            urn = g["urn"].iloc[0]
            for stratum in STRATA:
                s = g[_stratum_mask(g["region"], stratum)]
                if len(s) < MIN_N:
                    continue

                rep = s.dropna(subset=["fit_rep1", "fit_rep2"])
                if len(rep) >= MIN_N:
                    r = float(rep["fit_rep1"].corr(rep["fit_rep2"], method="spearman"))
                    rel = spearman_brown(r)
                    seed = layer_seed("core", gene, stratum, "replicates", pop)
                    rows.append({
                        "gene": gene, "urn": urn, "population": pop, "stratum": stratum,
                        "method": "replicates", "k": 2, "n": len(rep), "rep_rho": r,
                        "score_var": np.nan, "mean_se2": np.nan,
                        "reliability_unclipped": np.nan, "reliability": rel,
                        "ceiling": np.sqrt(max(rel, 0.0)), "status": "validated",
                        "role": "main", "seed": seed, "bootstrap_B": b,
                        **bootstrap_layer("replicates", seed,
                                          reps=[rep["fit_rep1"].to_numpy(float),
                                                rep["fit_rep2"].to_numpy(float)], b=b),
                        "inflation_factor": inflation.get(gene, np.nan),
                        "ceiling_inflated": np.nan})

                se = s.dropna(subset=["se", "functional_pathogenicity"])
                if len(se) >= MIN_N:
                    status = se["se_status"].iloc[0]
                    if status not in USABLE_SE:
                        continue                          # not estimated (NOT_ESTIMATED)
                    var = float(se["functional_pathogenicity"].var())
                    err = float((se["se"] ** 2).mean())
                    raw = 1 - err / var if var > 0 else np.nan
                    rel = float(np.clip(raw, 0.0, 1.0)) if var > 0 else np.nan
                    seed = layer_seed("core", gene, stratum, "se", pop)
                    factor = inflation.get(gene) if CORE_METHOD.get(gene) == "se" else None
                    rows.append({
                        "gene": gene, "urn": urn, "population": pop, "stratum": stratum,
                        "method": "se", "k": np.nan, "n": len(se), "rep_rho": np.nan,
                        "score_var": var, "mean_se2": err, "reliability_unclipped": raw,
                        "reliability": rel, "ceiling": np.sqrt(max(rel, 0.0)),
                        "status": status,
                        "role": "main" if status == "validated" else "cross_check",
                        "seed": seed, "bootstrap_B": b,
                        **bootstrap_layer("se", seed,
                                          y=se["functional_pathogenicity"].to_numpy(float),
                                          se=se["se"].to_numpy(float), b=b),
                        "inflation_factor": inflation.get(gene, np.nan),
                        "ceiling_inflated": inflated_ceiling(err, var, factor)})
    t = pd.DataFrame(rows)
    t["estimable"] = t["ceiling_lo"] > 0
    return t[TABLE_COLUMNS]


# the SE's adequacy, on the neutral variants of the genes with an error model
NEUTRAL_GENES = ("BRCA1", "BARD1", "PALB2", "BAP1")
SPLICEAI_NEUTRAL = 0.1
BATCHES = ("synonymous", "synonymous_spliceai_below_0.1")
SE_SOURCE = {"BRCA1": "sd of the two replicate fitness scores / sqrt(2) (atlas.assay_aux)"}


def neutral_check(mat: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:
    """Per gene, two batches of SNVs: every synonymous one, and those whose spliceai_ds is
    below 0.1. n, the functional score's variance, the mean SE^2 and their ratio."""
    keep = ["variant_id", "gene", "se", "se_status", "se_validated_by"]
    df = population.select(mat, population.SNV).merge(aux[keep], on=["variant_id", "gene"],
                                                      how="left")
    rows = []
    for gene in NEUTRAL_GENES:
        g = df[(df["gene"] == gene) & df["consequence"].eq("synonymous") & df["se"].notna()
               & df["se_status"].isin(USABLE_SE) & df["functional_pathogenicity"].notna()]
        for batch, d in ((BATCHES[0], g), (BATCHES[1], g[g["spliceai_ds"] < SPLICEAI_NEUTRAL])):
            var = float(d["functional_pathogenicity"].var())
            err = float((d["se"] ** 2).mean())
            by = d["se_validated_by"].dropna().unique()
            rows.append({"gene": gene, "batch": batch, "n": int(len(d)), "score_var": var,
                         "mean_se2": err, "ratio": var / err if err > 0 else np.nan,
                         "method": CORE_METHOD[gene],
                         "se_source": SE_SOURCE.get(gene, "the deposit's SE, validated by "
                                                    + ";".join(map(str, by)))})
    return pd.DataFrame(rows)


def inflation_factors(neutral: pd.DataFrame) -> dict[str, float]:
    """max(1, the second batch's ratio), per gene."""
    b2 = neutral[neutral["batch"] == BATCHES[1]].set_index("gene")["ratio"]
    return {g: float(max(1.0, r)) for g, r in b2.items() if np.isfinite(r)}


# Two further deposits carry repeated measurements that are *not* clean
# replicates. They cannot enter the primary ceiling, but they bound it in known
# directions and are reported as corroboration.
CORROBORATION = {
    # VHL: a second selection condition, not a replicate. Between-condition
    # agreement is attenuated by both conditions' noise *and* by any real
    # difference between the conditions, so it is a lower bound on reliability.
    "VHL": {"urn": "urn_mavedb_00000675-a-1", "a": "score", "b": "score_no_DAB",
            "kind": "cross-condition", "direction": "lower bound"},
    # BAP1: cumulative time points sharing a day-4 baseline. The shared baseline
    # inflates their agreement relative to independent replicates, so this is an
    # upper bound on reliability.
    "BAP1": {"urn": "urn_mavedb_00000662-0-1", "a": "processed_LFC_D4_D7",
             "b": "processed_LFC_D4_D21", "kind": "nested time points",
             "direction": "upper bound"},
}


def corroboration_table(frozen: pd.DataFrame) -> pd.DataFrame:
    """Bounding reproducibility estimates for the genes without a clean error model,
    per population (``frozen`` carries variant_id, hgvs_c, ref and alt).

    These do not give a usable ceiling, but they test whether the pattern seen in
    the genes with a validated error model — reproducibility falling steeply with
    intronic offset — holds more widely, and in particular whether the near-zero
    deep-intronic ceiling from BARD1 is echoed in the only other gene with
    deep-intronic coverage.
    """
    ids = population.select(frozen, None)[["variant_id", "population", "region"]]
    rows = []
    for gene, spec in CORROBORATION.items():
        path = REPO / "data" / "raw" / "mavedb" / spec["urn"] / "scores.csv"
        raw = pd.read_csv(path, low_memory=False).set_index("accession")
        j = raw.join(ids.set_index("variant_id"), how="inner")
        for pop in population.POPULATIONS:
            jp = j[j["population"] == pop]
            for stratum in STRATA:
                s = jp[_stratum_mask(jp["region"], stratum)].dropna(subset=[spec["a"], spec["b"]])
                if len(s) < MIN_N:
                    continue
                r = float(s[spec["a"]].corr(s[spec["b"]], method="spearman"))
                rows.append({"gene": gene, "population": pop, "stratum": stratum, "n": len(s),
                             "comparison": f"{spec['a']} vs {spec['b']}",
                             "kind": spec["kind"], "direction": spec["direction"],
                             "agreement_rho": r,
                             "implied_ceiling": np.sqrt(max(spearman_brown(r), 0.0))})
    return pd.DataFrame(rows)


def per_gene_rho(mat: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per gene × stratum Spearman ρ for one model (same convention as atlas.evaluate)."""
    df = mat.copy()
    df["region"] = df["hgvs_c"].map(classify_region)
    rows = []
    for gene, g in df.groupby("gene"):
        for stratum in STRATA:
            s = g[_stratum_mask(g["region"], stratum)].dropna(
                subset=[model, "functional_pathogenicity"])
            if len(s) < MIN_N:
                continue
            rows.append({
                "gene": gene, "stratum": stratum, "model": model, "n": len(s),
                "rho": float(s[model].corr(s["functional_pathogenicity"], method="spearman")),
            })
    return pd.DataFrame(rows)


def realisation_table(mat: pd.DataFrame, rel: pd.DataFrame) -> pd.DataFrame:
    """Observed ρ as a fraction of the attenuation ceiling, per gene × stratum × model."""
    primary = rel[rel["status"] == "validated"].copy()
    # one ceiling per gene × stratum: the replicate estimate where both exist
    primary["_order"] = primary["method"].map(METHOD_PREFERENCE).fillna(len(METHOD_PREFERENCE))
    primary = (primary.sort_values(["gene", "stratum", "_order"], kind="stable")
                      .drop_duplicates(["gene", "stratum"], keep="first"))
    ceil = primary.set_index(["gene", "stratum"])[["ceiling", "reliability", "method"]]

    out = []
    # A model declared in MODELS but absent from the matrix used to be skipped in
    # silence. Running with the default matrix therefore produced a table missing
    # all seven dbNSFP meta-predictors -- 48 of 257 rows -- and the only sign was
    # that the delivered Supplemental Table S3 no longer regenerated.
    absent = [m for m in MODELS if m not in mat.columns]
    if absent:
        print(f"[reliability] WARNING: {len(absent)} declared models are not in this "
              f"matrix and are omitted: {', '.join(absent)}", file=sys.stderr)
    for model in MODELS:
        if model in absent:
            continue
        pg = per_gene_rho(mat, model)
        pg = pg.join(ceil, on=["gene", "stratum"], how="inner")
        pg["realisation"] = pg["rho"] / pg["ceiling"]
        out.append(pg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# a trial outside this repository, for comparison only (not an acceptance value):
# the second batch's ratio per gene, and BAP1's SNV layer ceilings with their n
INSTRUCTION_TRIAL = {
    "neutral_ratio_batch2": {"BAP1": (3.4, 1322), "BARD1": (1.0, None), "PALB2": (0.8, None),
                             "BRCA1": (2.1, None)},
    "bap1_snv_ceiling": {"coding_or_utr": (0.992, 6981), "splice_1_2": (0.992, 192),
                         "splice_3_10": (0.992, 768), "splice_11_50": (0.969, 1500)},
}
TRIAL_TOL = 0.005


def stratum_summary(rel: pd.DataFrame) -> list[dict]:
    """Per stratum, on the main rows: the genes, those estimable, the ceiling's median
    and range."""
    out = []
    for stratum in STRATA:
        s = rel[(rel["stratum"] == stratum) & (rel["role"] == "main")]
        if s.empty:
            continue
        e = s[s["estimable"]]
        out.append({"stratum": stratum, "n_genes": int(s["gene"].nunique()),
                    "genes": sorted(s["gene"]), "n_genes_estimable": int(e["gene"].nunique()),
                    "genes_not_estimable": sorted(set(s["gene"]) - set(e["gene"])),
                    "ceiling_median": float(s["ceiling"].median()),
                    "ceiling_min": float(s["ceiling"].min()),
                    "ceiling_max": float(s["ceiling"].max())})
    return out


def trial_comparison(rel: pd.DataFrame, neutral: pd.DataFrame) -> dict:
    snv = rel[(rel["population"] == population.SNV) & (rel["role"] == "main")]
    out = {"note": ("a trial outside this repository, for comparison only; differences "
                    f"above {TRIAL_TOL} are listed"), "neutral_ratio_batch2": {},
           "bap1_snv_ceiling": {}, "differ_by_more_than_tolerance": []}
    b2 = neutral[neutral["batch"] == BATCHES[1]].set_index("gene")
    for g, (want, n) in INSTRUCTION_TRIAL["neutral_ratio_batch2"].items():
        got = float(b2.loc[g, "ratio"])
        out["neutral_ratio_batch2"][g] = {"trial": want, "trial_n": n, "here": got,
                                          "n": int(b2.loc[g, "n"])}
    bap1 = snv[snv["gene"] == "BAP1"].set_index("stratum")
    for st, (want, n) in INSTRUCTION_TRIAL["bap1_snv_ceiling"].items():
        got = float(bap1.loc[st, "ceiling"])
        rec = {"trial": want, "trial_n": n, "here": got, "n": int(bap1.loc[st, "n"]),
               "difference": got - want}
        out["bap1_snv_ceiling"][st] = rec
        if abs(got - want) > TRIAL_TOL or int(bap1.loc[st, "n"]) != n:
            out["differ_by_more_than_tolerance"].append(st)
    return out


def jsonable(obj):
    """NaN as null, numpy scalars as Python ones, recursively."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def summary(rel: pd.DataFrame, neutral: pd.DataFrame, inflation: dict) -> dict:
    cross = []
    b = rel[rel["gene"] == "BRCA1"]
    for (pop, st), d in b.groupby(["population", "stratum"]):
        m = d.set_index("method")
        if {"replicates", "se"} <= set(m.index):
            cross.append({"population": pop, "stratum": st,
                          "replicates": float(m.loc["replicates", "ceiling"]),
                          "se_derived_from_replicates": float(m.loc["se", "ceiling"]),
                          "se_minus_replicates": float(m.loc["se", "ceiling"]
                                                       - m.loc["replicates", "ceiling"])})
    return jsonable({
        "definition": {"estimators": CORE_METHOD, "not_estimated": NOT_ESTIMATED,
                       "min_n": MIN_N, "bootstrap_B": B,
                       "seed": "atlas.matched_ceilings.cell_seed((group, gene, stratum, "
                               "method, population)), SEED 20261122 as in A2",
                       "estimable": "the ceiling's 2.5% bound is above 0",
                       "clipping": "reliability clipped to [0, 1]; reliability_unclipped "
                                   "keeps the SE estimate before clipping",
                       "inflation": "inflation_factor = max(1, the ratio of the second "
                                    "neutral batch); ceiling_inflated for the genes "
                                    "estimated by SE"},
        "ceiling_by_stratum": {pop: stratum_summary(rel[rel["population"] == pop])
                               for pop in population.POPULATIONS},
        "cross_check_replicates_vs_se": cross,
        "neutral_check": neutral.to_dict("records"),
        "inflation_factor": inflation,
        "instruction_trial": trial_comparison(rel, neutral),
    })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default=str(paths.path("score_matrix")))
    ap.add_argument("--aux", default=str(paths.path("assay_aux")))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    mat = population.load(None, Path(args.matrix))
    aux = pd.read_parquet(args.aux)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    neutral = neutral_check(mat, aux)
    inflation = inflation_factors(neutral)
    rel = reliability_table(mat, aux, inflation)
    corr = corroboration_table(mat[["variant_id", "hgvs_c", "ref", "alt"]])
    rel.to_csv(out / paths.name("reliability"), sep="\t", index=False)
    corr.to_csv(out / paths.name("reliability_corroboration"), sep="\t", index=False)
    neutral.to_csv(out / paths.name("reliability_neutral_check"), sep="\t", index=False)
    summ = summary(rel, neutral, inflation)
    (out / paths.name("reliability_summary")).write_text(json.dumps(summ, indent=1) + "\n")

    print("Layer ceilings, snv, main rows:")
    snv = rel[(rel["population"] == population.SNV) & (rel["role"] == "main")]
    print(snv.pivot(index="gene", columns="stratum", values="ceiling")
             .reindex(columns=[s for s in STRATA if s in set(snv["stratum"])])
             .to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nNeutral check:")
    print(neutral[["gene", "batch", "n", "score_var", "mean_se2", "ratio"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\ninflation factors:", {k: round(v, 3) for k, v in inflation.items()})
    t = summ["instruction_trial"]
    print("trial comparison, differing by more than the tolerance:",
          t["differ_by_more_than_tolerance"] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
