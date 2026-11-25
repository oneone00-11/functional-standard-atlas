"""The pooled territory grid, per population, with the per-gene values it is pooled from.

Per gene, Spearman rho of each predictor against ``functional_pathogenicity``; per
stratum, the genes with at least MIN_N (30) variants, pooled with DerSimonian-Laird on
Fisher-z when there are at least two. Everything is imported from ``atlas.evaluate``, so
the numbers are directly comparable with v5's.

Populations (``atlas.population``):

``snv``      the main analysis; sixteen strata:
             the seven territories (all, coding_or_utr, splice_region, splice_1_2,
             splice_3_10, splice_11_50, splice_deep);
             five consequence classes (missense, synonymous, nonsense, utr, and
             coding_snv_nontruncating = missense or synonymous), from the matrix's
             ``consequence``;
             ClinVar's three arms (clinvar_classified, clinvar_recorded_unclassified,
             clinvar_unrecorded) from ``clinvar_arm`` (A3.3), and the sub-arm
             clinvar_no_assertion (``clinvar_arm_strict`` == recorded_no_assertion).
             v5's recorded/absent split and its list of recorded ids are not used.
``non_snv``  the territories with enough variants for a pooled value (at least two
             genes with MIN_N scored variants for some predictor), and the variant
             classes deletion and complex; only for the predictors that score a non-SNV
             (``alphagenome``, ``phylop100way``, ``phastcons100way`` score every one; the
             two gnomAD columns 128).

The selection overlap (``atlas.predictors.SELECTION_OVERLAP``): avi's BRCA1 and RAD51C
cells are computed and listed in the per-gene table with ``selection_overlap`` set, and
are left out of avi's pooled values; the grid's ``excluded_genes`` names them.

Outputs: results/eval_ext_v2.tsv (one row per population x predictor x stratum) and
results/eval_ext_per_gene_v2.tsv (one row per population x predictor x stratum x gene
with at least MIN_N variants: n and rho; the pooled values are computed from these rows,
less the flagged ones).

Usage (PYTHONPATH=src):  python -m atlas.evaluate_ext [--out results/]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from atlas import paths, population, predictors
from atlas.evaluate import dl_pool, fisher_z

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

MODELS = predictors.PREDICTORS
MIN_N = 30

TERRITORIES = ["all", "coding_or_utr", "splice_region",
               "splice_1_2", "splice_3_10", "splice_11_50", "splice_deep"]
CONSEQUENCE = ["missense", "synonymous", "nonsense", "utr", "coding_snv_nontruncating"]
CLINVAR = ["clinvar_classified", "clinvar_recorded_unclassified", "clinvar_unrecorded",
           "clinvar_no_assertion"]
CLASSES = ["deletion", "complex"]

GRID_COLUMNS = ["population", "model", "stratum", "k_genes", "n", "pooled_rho", "ci_lo",
                "ci_hi", "i2_pct", "excluded_genes"]
PER_GENE_COLUMNS = ["population", "model", "stratum", "gene", "n", "rho", "selection_overlap"]


def territory_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    region = df["region"]
    return {"all": pd.Series(True, index=df.index),
            "coding_or_utr": region.eq("coding_or_utr"),
            "splice_region": region.str.startswith("splice_"),
            "splice_1_2": region.eq("splice_1_2"),
            "splice_3_10": region.eq("splice_3_10"),
            "splice_11_50": region.eq("splice_11_50"),
            "splice_deep": region.eq("splice_deep")}


def stratum_masks(df: pd.DataFrame, pop: str = population.SNV) -> dict[str, pd.Series]:
    """The strata of one population (df holds that population only)."""
    m = territory_masks(df)
    if pop == population.SNV:
        cons = df["consequence"]
        m |= {"missense": cons.eq("missense"), "synonymous": cons.eq("synonymous"),
              "nonsense": cons.eq("nonsense"), "utr": cons.isin(["utr5", "utr3"]),
              "coding_snv_nontruncating": cons.isin(["missense", "synonymous"])}
        arm, strict = df["clinvar_arm"], df["clinvar_arm_strict"]
        m |= {"clinvar_classified": arm.eq("classified"),
              "clinvar_recorded_unclassified": arm.eq("recorded_unclassified"),
              "clinvar_unrecorded": arm.eq("unrecorded"),
              "clinvar_no_assertion": strict.eq("recorded_no_assertion")}
    else:
        m |= {c: df["variant_class"].eq(c) for c in CLASSES}
    return m


def per_gene(sub: pd.DataFrame, model: str) -> list[dict]:
    """Per gene with at least MIN_N variants: n, rho, Fisher z and its variance."""
    out = []
    for gene, g in sub.groupby("gene"):
        if len(g) < MIN_N:
            continue
        r = float(g[model].corr(g["functional_pathogenicity"], method="spearman"))
        z, v = fisher_z(r, len(g))
        out.append({"gene": gene, "n": len(g), "rho": r, "z": z, "var": v,
                    "selection_overlap": predictors.is_overlap(model, gene)})
    return out


def pooled_row(pop: str, model: str, stratum: str, sub: pd.DataFrame, pg: list[dict]) -> dict:
    excluded = sorted(predictors.overlap_genes(model))
    use = [p for p in pg if not p["selection_overlap"]]
    n = int((~sub["gene"].isin(excluded)).sum())
    row = {"population": pop, "model": model, "stratum": stratum, "k_genes": len(use),
           "n": n, "pooled_rho": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
           "i2_pct": np.nan, "excluded_genes": ";".join(excluded)}
    if len(use) >= 2:
        p = dl_pool([u["z"] for u in use], [u["var"] for u in use])
        row |= {"pooled_rho": math.tanh(p["pooled"]), "ci_lo": math.tanh(p["ci_lo"]),
                "ci_hi": math.tanh(p["ci_hi"]), "i2_pct": p["i2_pct"]}
    return row


def evaluate_population(df: pd.DataFrame, pop: str) -> tuple[list[dict], list[dict]]:
    d = population.select(df, pop)
    masks = stratum_masks(d, pop)
    models = [m for m in MODELS if m in d.columns]
    if pop == population.NON_SNV:
        models = [m for m in models if d[m].notna().any()]
    cells = {}
    for model in models:
        for stratum, mask in masks.items():
            sub = d[mask].dropna(subset=[model, "functional_pathogenicity"])
            cells[(model, stratum)] = (sub, per_gene(sub, model))
    if pop == population.NON_SNV:
        # a territory enters when some predictor pools in it; the classes always do
        keep = set(CLASSES) | {s for (m, s), (_, pg) in cells.items()
                               if sum(not p["selection_overlap"] for p in pg) >= 2}
        cells = {k: v for k, v in cells.items() if k[1] in keep}
    rows, genes = [], []
    for (model, stratum), (sub, pg) in cells.items():
        rows.append(pooled_row(pop, model, stratum, sub, pg))
        genes += [{"population": pop, "model": model, "stratum": stratum, "gene": p["gene"],
                   "n": p["n"], "rho": p["rho"], "selection_overlap": p["selection_overlap"]}
                  for p in pg]
    return rows, genes


def evaluate_grid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # a declared predictor absent from the matrix is said so, not skipped in silence
    absent = [m for m in MODELS if m not in df.columns]
    if absent:
        print(f"[evaluate_ext] WARNING: {len(absent)} declared models are not in "
              f"this matrix and are omitted: {', '.join(absent)}", file=sys.stderr)
    rows, genes = [], []
    for pop in population.POPULATIONS:
        r, g = evaluate_population(df, pop)
        rows += r
        genes += g
    return (pd.DataFrame(rows, columns=GRID_COLUMNS),
            pd.DataFrame(genes, columns=PER_GENE_COLUMNS))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default=str(paths.path("score_matrix")))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    df = population.load(None, Path(args.matrix))
    grid, genes = evaluate_grid(df)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / paths.name("eval_ext"), sep="\t", index=False)
    genes.to_csv(out / paths.name("eval_ext_per_gene"), sep="\t", index=False)

    snv = grid[grid["population"] == population.SNV]
    wide = snv.pivot(index="model", columns="stratum", values="pooled_rho")
    print(f"{len(grid)} rows ({len(snv)} snv, {len(grid) - len(snv)} non_snv); "
          f"{len(genes)} per-gene rows")
    print(wide.reindex([m for m in MODELS if m in wide.index])[TERRITORIES]
          .to_string(float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
