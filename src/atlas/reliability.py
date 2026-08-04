"""Measurement reliability of the functional standard, and the attenuation
ceiling it imposes on any predictor.

A Spearman ρ between a predictor and a noisy measurement is bounded above by the
square root of that measurement's reliability. Without this bound, a territory
map conflates two very different situations: a stratum where predictors are bad,
and a stratum where the assay cannot measure anything. This module estimates the
bound per gene × stratum from the assay's own error model, and expresses each
predictor's observed ρ as a fraction of it ("realisation").

Reliability estimators (per gene, per stratum)
----------------------------------------------
``replicates``  Two independent replicate scores whose mean *is* the deposited
                score (verified for BRCA1). Spearman ρ between replicates gives
                the reliability of a single replicate; Spearman–Brown steps it
                up to the 2-replicate mean: ``rel = 2r / (1 + r)``.
``se``          Per-variant standard errors validated against the deposit's own
                published 95% CI. Classical decomposition of observed variance
                into true + error variance: ``rel = 1 - mean(SE^2) / var(score)``.

The two estimators are on different scales (rank vs variance). BRCA1 carries
both and is used as the cross-check; the agreement is recorded in the output.
Assays whose SE column is unvalidated or demonstrably on a different scale
(BAP1, RAD51C — see ``atlas.assay_aux``) are reported separately and excluded
from the primary estimate.

Usage (PYTHONPATH=src):  python -m atlas.reliability --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.evaluate import classify_region

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

STRATA = ["all", "coding_or_utr", "splice_region",
          "splice_1_2", "splice_3_10", "splice_11_50", "splice_deep"]

MODELS = [
    "alphagenome", "spliceai_ds", "pangolin_score", "cadd", "alphamissense",
    "evo2", "gpn_msa", "nucleotide_transformer", "phylop100way",
    "phastcons100way", "gnomad_af_global", "gnomad_af_popmax",
    # dbNSFP meta-predictors (missense-only; atlas.metapredictors)
    "revel", "bayesdel_addaf", "clinpred", "metarnn", "primateai", "vest4", "esm1b",
]

MIN_N = 30


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


def reliability_table(mat: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:
    """Per gene × stratum reliability from replicates and/or validated SEs."""
    df = mat.merge(aux, on=["variant_id", "gene"], how="left")
    df["region"] = df["hgvs_c"].map(classify_region)
    rows = []
    for gene, g in df.groupby("gene"):
        for stratum in STRATA:
            s = g[_stratum_mask(g["region"], stratum)]
            if len(s) < MIN_N:
                continue

            rep = s.dropna(subset=["fit_rep1", "fit_rep2"])
            if len(rep) >= MIN_N:
                r = float(rep["fit_rep1"].corr(rep["fit_rep2"], method="spearman"))
                rel = spearman_brown(r)
                rows.append({
                    "gene": gene, "stratum": stratum, "method": "replicates",
                    "n": len(rep), "rep_rho": r,
                    "reliability": rel, "ceiling": np.sqrt(max(rel, 0.0)),
                    "status": "validated",
                })

            se = s.dropna(subset=["se", "functional_pathogenicity"])
            if len(se) >= MIN_N:
                status = se["se_status"].iloc[0]
                var = float(se["functional_pathogenicity"].var())
                err = float((se["se"] ** 2).mean())
                rel = float(np.clip(1 - err / var, 0.0, 1.0)) if var > 0 else np.nan
                rows.append({
                    "gene": gene, "stratum": stratum, "method": "se",
                    "n": len(se), "rep_rho": np.nan,
                    "reliability": rel, "ceiling": np.sqrt(max(rel, 0.0)),
                    "status": status,
                })
    return pd.DataFrame(rows)


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
    """Bounding reproducibility estimates for the genes without a clean error model.

    These do not give a usable ceiling, but they test whether the pattern seen in
    the three validated genes — reproducibility falling steeply with intronic
    offset — holds more widely, and in particular whether the near-zero
    deep-intronic ceiling from BARD1 is echoed in the only other gene with
    deep-intronic coverage.
    """
    rows = []
    for gene, spec in CORROBORATION.items():
        path = REPO / "data" / "raw" / "mavedb" / spec["urn"] / "scores.csv"
        raw = pd.read_csv(path, low_memory=False).set_index("accession")
        j = raw.join(frozen.set_index("variant_id")[["hgvs_c"]], how="inner")
        j["region"] = j["hgvs_c"].map(classify_region)
        for stratum in STRATA:
            s = j[_stratum_mask(j["region"], stratum)].dropna(subset=[spec["a"], spec["b"]])
            if len(s) < MIN_N:
                continue
            r = float(s[spec["a"]].corr(s[spec["b"]], method="spearman"))
            rows.append({"gene": gene, "stratum": stratum, "n": len(s),
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
    primary = rel[rel["status"] == "validated"]
    # one ceiling per gene × stratum: prefer the replicate estimate where both exist
    primary = (primary.sort_values("method", ascending=False)  # 'se' < 'replicates'
                      .drop_duplicates(["gene", "stratum"], keep="first"))
    ceil = primary.set_index(["gene", "stratum"])[["ceiling", "reliability", "method"]]

    out = []
    for model in MODELS:
        if model not in mat.columns:
            continue
        pg = per_gene_rho(mat, model)
        pg = pg.join(ceil, on=["gene", "stratum"], how="inner")
        pg["realisation"] = pg["rho"] / pg["ceiling"]
        out.append(pg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=str(RESULTS / "score_matrix_atlas_v1.parquet"))
    ap.add_argument("--aux", default=str(RESULTS / "assay_aux_v1.parquet"))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    mat = pd.read_parquet(args.matrix)
    aux = pd.read_parquet(args.aux)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rel = reliability_table(mat, aux)
    rel.to_csv(out / "reliability_v1.tsv", sep="\t", index=False)

    corr = corroboration_table(mat[["variant_id", "hgvs_c"]])
    corr.to_csv(out / "reliability_corroboration_v1.tsv", sep="\t", index=False)

    real = realisation_table(mat, rel)
    real.to_csv(out / "attenuation_v1.tsv", sep="\t", index=False)

    # cross-check the two estimators where one gene carries both
    both = rel[rel["status"].isin(["validated", "derived_from_replicates"])].pivot_table(
        index=["gene", "stratum"], columns="method", values="ceiling")
    cross = both.dropna() if {"replicates", "se"} <= set(both.columns) else pd.DataFrame()
    if len(cross):
        cross = cross.assign(se_minus_replicates=cross["se"] - cross["replicates"])

    # summary: median ceiling per stratum, and median realisation per stratum
    ceil_sum = (rel[rel["status"] == "validated"]
                .groupby("stratum")
                .agg(genes=("gene", "nunique"),
                     ceiling_median=("ceiling", "median"),
                     ceiling_min=("ceiling", "min"),
                     ceiling_max=("ceiling", "max"))
                .reindex([s for s in STRATA if s in set(rel["stratum"])]))

    summary = {
        "n_genes_with_reliability": int(rel[rel["status"] == "validated"]["gene"].nunique()),
        "excluded": rel[rel["status"] != "validated"][["gene", "status"]]
                      .drop_duplicates().to_dict("records"),
        "cross_check_replicates_vs_se": (
            cross.reset_index().to_dict("records") if len(cross) else None),
        "ceiling_by_stratum": ceil_sum.reset_index().to_dict("records"),
    }
    (out / "reliability_v1.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    print("Attenuation ceiling (sqrt of reliability), validated estimators only")
    print(ceil_sum.to_string(float_format=lambda x: f"{x:.3f}"))
    if len(cross):
        print("\nCross-check on genes carrying both estimators (ceiling):")
        print(cross.to_string(float_format=lambda x: f"{x:.3f}"))
    if len(corr):
        print("\nCorroboration from genes without a clean error model "
              "(bounds, not usable ceilings):")
        print(corr.pivot_table(index=["gene", "direction"], columns="stratum",
                               values="agreement_rho")
                  .reindex(columns=[s for s in STRATA if s in set(corr["stratum"])])
                  .to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nExcluded SE columns:",
          rel[rel["status"] != "validated"][["gene", "status"]].drop_duplicates().to_dict("records"))

    if len(real):
        top = (real.groupby(["stratum", "model"])
                   .agg(genes=("gene", "nunique"), rho=("rho", "median"),
                        ceiling=("ceiling", "median"), realisation=("realisation", "median"))
                   .reset_index())
        best = top.sort_values("realisation", ascending=False).groupby("stratum").head(1)
        print("\nBest-realising model per stratum (median over genes):")
        print(best.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
