"""Robustness checks for the territory map.

Four checks, each answering an objection the pooled ρ table invites:

``ties``       Several predictors emit heavily quantised scores (the SpliceAI and
               Pangolin CLIs round delta scores to 2 dp). Where most variants in
               a stratum share one value, Spearman ρ is bounded well below 1 for
               arithmetic reasons alone, so a low ρ cannot be read as "the model
               is wrong". Reports the tie structure and the exact maximum ρ that
               the observed score vector permits.
``power``      Minimum ρ detectable at 80% power, per stratum, given the per-gene
               n actually used. A null result in a small stratum is only evidence
               of absence if the stratum could have detected a relevant effect.
``head2head``  The manuscript makes ordering claims ("AlphaGenome leads at
               3–10 bp"). Two correlations measured on the same variants are
               dependent, so overlapping marginal CIs are the wrong test. Uses
               Steiger's test for dependent correlations per gene, DL-pooled,
               plus a gene-cluster bootstrap of Δρ.
``logo``       Leave-one-gene-out pooled ρ: with I² up to 99%, a headline number
               that moves when one gene is dropped is a property of that gene.

Usage (PYTHONPATH=src):  python -m atlas.robustness --out results/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from atlas.evaluate import classify_region, dl_pool, fisher_z

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

# ordering claims made in the manuscript, as (model_a, model_b, stratum)
HEAD_TO_HEAD = [
    ("alphagenome", "spliceai_ds", "splice_3_10"),
    ("alphagenome", "pangolin_score", "splice_3_10"),
    ("pangolin_score", "spliceai_ds", "splice_3_10"),
    ("pangolin_score", "spliceai_ds", "splice_11_50"),
    ("pangolin_score", "alphagenome", "splice_region"),
    ("evo2", "cadd", "all"),
    ("evo2", "gpn_msa", "all"),
    ("cadd", "gpn_msa", "coding_or_utr"),
    ("evo2", "nucleotide_transformer", "all"),
    ("cadd", "phylop100way", "all"),
    # missense: AlphaMissense against the dbNSFP meta-predictors clinicians use
    ("alphamissense", "metarnn", "missense"),
    ("alphamissense", "clinpred", "missense"),
    ("alphamissense", "revel", "missense"),
    ("alphamissense", "bayesdel_addaf", "missense"),
    ("alphamissense", "vest4", "missense"),
    ("alphamissense", "esm1b", "missense"),
    ("alphamissense", "primateai", "missense"),
    ("alphamissense", "cadd", "missense"),
]

MIN_N = 30


CONSEQUENCE_STRATA = ("missense", "synonymous", "nonsense")


def _mask(region: pd.Series, stratum: str,
          consequence: pd.Series | None = None) -> pd.Series:
    if stratum == "all":
        return pd.Series(True, index=region.index)
    if stratum == "splice_region":
        return region.str.startswith("splice_")
    if stratum in CONSEQUENCE_STRATA:
        if consequence is None:
            raise ValueError(
                f"stratum {stratum!r} needs a 'consequence' column; none was supplied. "
                "Pass a matrix carrying it, or --consequence pointing at "
                "consequence_v1.parquet.")
        return consequence == stratum
    return region == stratum


# --------------------------------------------------------------------------
# 1. tie structure and the ρ ceiling it implies
# --------------------------------------------------------------------------
def max_spearman_given_ties(x: np.ndarray) -> float:
    """Largest Spearman ρ any target can achieve against this score vector.

    Ties in ``x`` receive the average rank; the best possible target is one that
    is strictly increasing within every tie group. Correlating the tied ranks
    against that ideal target gives the arithmetic ceiling.
    """
    n = len(x)
    if n < 3:
        return float("nan")
    rx = stats.rankdata(x, method="average")
    if rx.std() == 0:
        return 0.0
    order = np.argsort(x, kind="stable")
    ideal = np.empty(n, dtype=float)
    ideal[order] = np.arange(1, n + 1, dtype=float)
    return float(np.corrcoef(rx, ideal)[0, 1])


def tie_audit(mat: pd.DataFrame) -> pd.DataFrame:
    df = mat.copy()
    df["region"] = df["hgvs_c"].map(classify_region)
    rows = []
    for model in MODELS:
        if model not in df.columns:
            continue
        for stratum in STRATA:
            s = df[_mask(df["region"], stratum)].dropna(
                subset=[model, "functional_pathogenicity"])
            if len(s) < MIN_N:
                continue
            v = s[model].to_numpy()
            vals, counts = np.unique(v, return_counts=True)
            rows.append({
                "model": model, "stratum": stratum, "n": len(v),
                "n_distinct": int(len(vals)),
                "modal_value": float(vals[counts.argmax()]),
                "modal_share": float(counts.max() / len(v)),
                "rho_ceiling_ties": max_spearman_given_ties(v),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. minimum detectable ρ
# --------------------------------------------------------------------------
def min_detectable_rho(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest |ρ| detectable at the given power, via the Fisher-z normal approximation."""
    if n <= 3:
        return float("nan")
    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    return float(math.tanh((za + zb) / math.sqrt(n - 3)))


def power_table(mat: pd.DataFrame) -> pd.DataFrame:
    df = mat.copy()
    df["region"] = df["hgvs_c"].map(classify_region)
    rows = []
    for model in MODELS:
        if model not in df.columns:
            continue
        for stratum in STRATA:
            s = df[_mask(df["region"], stratum)].dropna(
                subset=[model, "functional_pathogenicity"])
            per_gene = s.groupby("gene").size()
            per_gene = per_gene[per_gene >= MIN_N]
            if per_gene.empty:
                continue
            # DL pooling of k independent Fisher-z estimates: under homogeneity the
            # pooled variance is 1/sum(n_i - 3), so the pooled study behaves like a
            # single sample of size sum(n_i - 3) + 3.
            n_eff = int((per_gene - 3).sum() + 3)
            rows.append({
                "model": model, "stratum": stratum,
                "k_genes": int(len(per_gene)),
                "n_total": int(per_gene.sum()),
                "n_min_gene": int(per_gene.min()),
                "mdr_per_gene_min": min_detectable_rho(int(per_gene.min())),
                "mdr_pooled_homogeneous": min_detectable_rho(n_eff),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. head-to-head: Steiger's dependent-correlation test, DL-pooled
# --------------------------------------------------------------------------
def steiger_z_diff(r_ay: float, r_by: float, r_ab: float, n: int) -> tuple[float, float]:
    """(z_a - z_b, variance) for two correlations sharing the variable y."""
    za, _ = fisher_z(r_ay, n)
    zb, _ = fisher_z(r_by, n)
    r2 = (r_ay ** 2 + r_by ** 2) / 2.0
    f = (1 - r_ab) / (2 * (1 - r2)) if r2 < 1 else 1.0
    f = min(f, 1.0)
    h = (1 - f * r2) / (1 - r2) if r2 < 1 else 1.0
    var = 2 * (1 - r_ab) * h / (n - 3)
    return za - zb, max(var, 1e-12)


def head_to_head(mat: pd.DataFrame, n_boot: int = 4000, seed: int = 20260803) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = mat.copy()
    df["region"] = df["hgvs_c"].map(classify_region)
    rows = []
    for a, b, stratum in HEAD_TO_HEAD:
        if a not in df.columns or b not in df.columns:
            continue
        s = df[_mask(df["region"], stratum, df.get("consequence"))].dropna(
            subset=[a, b, "functional_pathogenicity"])
        per_gene = []
        for gene, g in s.groupby("gene"):
            n = len(g)
            if n < MIN_N:
                continue
            y = g["functional_pathogenicity"]
            r_ay = float(g[a].corr(y, method="spearman"))
            r_by = float(g[b].corr(y, method="spearman"))
            r_ab = float(g[a].corr(g[b], method="spearman"))
            dz, var = steiger_z_diff(r_ay, r_by, r_ab, n)
            za, va = fisher_z(r_ay, n)
            zb, vb = fisher_z(r_by, n)
            per_gene.append({"gene": gene, "n": n, "r_a": r_ay, "r_b": r_by,
                             "r_ab": r_ab, "dz": dz, "var": var,
                             "za": za, "va": va, "zb": zb, "vb": vb})
        if len(per_gene) < 2:
            continue
        pg = pd.DataFrame(per_gene)
        pooled = dl_pool(pg["dz"].tolist(), pg["var"].tolist())
        z_stat = pooled["pooled"] / pooled["se"]
        p = float(2 * stats.norm.sf(abs(z_stat)))

        # gene-cluster bootstrap of Δρ on the ρ scale
        idx = np.arange(len(pg))
        deltas = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.choice(idx, size=len(idx), replace=True)
            sub = pg.iloc[pick]
            pa = dl_pool(sub["za"].tolist(), sub["va"].tolist())
            pb = dl_pool(sub["zb"].tolist(), sub["vb"].tolist())
            deltas[i] = math.tanh(pa["pooled"]) - math.tanh(pb["pooled"])
        full_a = dl_pool(pg["za"].tolist(), pg["va"].tolist())
        full_b = dl_pool(pg["zb"].tolist(), pg["vb"].tolist())
        rows.append({
            "model_a": a, "model_b": b, "stratum": stratum,
            "k_genes": len(pg), "n_paired": int(pg["n"].sum()),
            "rho_a": math.tanh(full_a["pooled"]), "rho_b": math.tanh(full_b["pooled"]),
            "delta_rho": math.tanh(full_a["pooled"]) - math.tanh(full_b["pooled"]),
            "boot_ci_lo": float(np.percentile(deltas, 2.5)),
            "boot_ci_hi": float(np.percentile(deltas, 97.5)),
            "median_r_ab": float(pg["r_ab"].median()),
            "steiger_pooled_dz": pooled["pooled"],
            "steiger_p": p,
            "i2_pct": pooled["i2_pct"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. leave-one-gene-out
# --------------------------------------------------------------------------
def logo(mat: pd.DataFrame) -> pd.DataFrame:
    df = mat.copy()
    df["region"] = df["hgvs_c"].map(classify_region)
    rows = []
    for model in MODELS:
        if model not in df.columns:
            continue
        for stratum in STRATA:
            s = df[_mask(df["region"], stratum)].dropna(
                subset=[model, "functional_pathogenicity"])
            per_gene = []
            for gene, g in s.groupby("gene"):
                if len(g) < MIN_N:
                    continue
                r = float(g[model].corr(g["functional_pathogenicity"], method="spearman"))
                z, v = fisher_z(r, len(g))
                per_gene.append({"gene": gene, "z": z, "var": v})
            if len(per_gene) < 3:
                continue
            pg = pd.DataFrame(per_gene)
            full = math.tanh(dl_pool(pg["z"].tolist(), pg["var"].tolist())["pooled"])
            drops = {}
            for gene in pg["gene"]:
                sub = pg[pg["gene"] != gene]
                drops[gene] = math.tanh(dl_pool(sub["z"].tolist(), sub["var"].tolist())["pooled"])
            lo, hi = min(drops.values()), max(drops.values())
            worst = max(drops, key=lambda gname: abs(drops[gname] - full))
            rows.append({
                "model": model, "stratum": stratum, "k_genes": len(pg),
                "rho_full": full, "logo_min": lo, "logo_max": hi,
                "logo_range": hi - lo,
                "most_influential_gene": worst,
                "delta_if_dropped": drops[worst] - full,
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=str(RESULTS / "score_matrix_atlas_v1.parquet"))
    ap.add_argument("--consequence", default=str(RESULTS / "consequence_v1.parquet"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args(argv)

    mat = pd.read_parquet(args.matrix)
    cons_path = Path(args.consequence)
    if "consequence" not in mat.columns and cons_path.exists():
        mat = mat.merge(pd.read_parquet(cons_path)[["variant_id", "consequence"]],
                        on="variant_id", how="left")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ties = tie_audit(mat)
    ties.to_csv(out / "tie_audit_v1.tsv", sep="\t", index=False)
    pw = power_table(mat)
    pw.to_csv(out / "power_v1.tsv", sep="\t", index=False)
    h2h = head_to_head(mat, n_boot=args.n_boot)
    h2h.to_csv(out / "head_to_head_v1.tsv", sep="\t", index=False)
    lo = logo(mat)
    lo.to_csv(out / "logo_v1.tsv", sep="\t", index=False)

    worst_ties = ties.sort_values("rho_ceiling_ties").head(12)
    print("Most tie-limited model × stratum cells (arithmetic ρ ceiling from ties alone):")
    print(worst_ties[["model", "stratum", "n", "n_distinct", "modal_value",
                      "modal_share", "rho_ceiling_ties"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nMinimum detectable ρ (80% power, α=0.05) by stratum:")
    print(pw.groupby("stratum")
            .agg(k=("k_genes", "max"), n_min_gene=("n_min_gene", "min"),
                 mdr_gene=("mdr_per_gene_min", "max"),
                 mdr_pooled=("mdr_pooled_homogeneous", "max"))
            .reindex([s for s in STRATA if s in set(pw["stratum"])])
            .to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nHead-to-head (paired, Steiger + gene-cluster bootstrap):")
    print(h2h[["model_a", "model_b", "stratum", "k_genes", "n_paired", "rho_a", "rho_b",
               "delta_rho", "boot_ci_lo", "boot_ci_hi", "steiger_p"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nLargest leave-one-gene-out swings:")
    print(lo.sort_values("logo_range", ascending=False).head(10)
            [["model", "stratum", "rho_full", "logo_min", "logo_max",
              "most_influential_gene", "delta_if_dropped"]]
            .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    (out / "robustness_v1.json").write_text(json.dumps({
        "n_boot": args.n_boot,
        "tie_limited_cells": int((ties["rho_ceiling_ties"] < 0.95).sum()),
        "head_to_head_significant": int((h2h["steiger_p"] < 0.05).sum()),
        "head_to_head_tested": int(len(h2h)),
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
