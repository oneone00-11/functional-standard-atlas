"""Cross-readout replication: predictors against the assays' RNA-level scores.

Four of the seven deposits publish a second, mechanistically distinct readout of
the same variants — RNA abundance of the variant allele — alongside the cell
fitness score the atlas freezes. The RNA readout exists only for coding/UTR
variants (verified below and reported), so this is not a splice-territory
analysis; it is a test of whether predictor performance is a property of the
variant or a property of the chosen readout.

It also supplies something a benchmark rarely has: an empirical, same-experiment
reference point for how large a ρ can reasonably be expected to be. If two
readouts of one assay agree at ρ = r, then a predictor reaching ρ = r against
one of them is performing at the level of the experiment's own internal
agreement.

Usage (PYTHONPATH=src):  python -m atlas.rna_readout --out results/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.evaluate import classify_region, dl_pool, fisher_z
from atlas.reliability import spearman_brown

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

MODELS = [
    "cadd", "alphamissense", "gpn_msa", "evo2", "phylop100way", "alphagenome",
    "nucleotide_transformer", "phastcons100way", "spliceai_ds", "pangolin_score",
    "gnomad_af_global", "gnomad_af_popmax",
    # dbNSFP meta-predictors (missense-only; atlas.metapredictors)
    "revel", "bayesdel_addaf", "clinpred", "metarnn", "primateai", "vest4", "esm1b",
]
MIN_N = 30


def _pool(per_gene: list[dict]) -> dict | None:
    if len(per_gene) < 2:
        return None
    pooled = dl_pool([p["z"] for p in per_gene], [p["var"] for p in per_gene])
    return {
        "k": pooled["k"],
        "rho": math.tanh(pooled["pooled"]),
        "ci_lo": math.tanh(pooled["ci_lo"]),
        "ci_hi": math.tanh(pooled["ci_hi"]),
        "i2_pct": pooled["i2_pct"],
    }


def _rho_row(x: pd.Series, y: pd.Series) -> dict | None:
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) < MIN_N:
        return None
    r = float(s.iloc[:, 0].corr(s.iloc[:, 1], method="spearman"))
    z, v = fisher_z(r, len(s))
    return {"n": len(s), "rho": r, "z": z, "var": v}


def analyse(mat: pd.DataFrame, aux: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = mat.merge(aux[["variant_id", "rna", "rna_rep1", "rna_rep2"]],
                   on="variant_id", how="left")
    df["region"] = df["hgvs_c"].map(classify_region)
    has_rna = df.dropna(subset=["rna"])

    prov = {
        "n_with_rna": int(len(has_rna)),
        "genes": sorted(has_rna["gene"].unique().tolist()),
        "region_coverage": has_rna["region"].value_counts().to_dict(),
    }
    # the RNA readout is coding-only; state it rather than assume it
    coding = has_rna[has_rna["region"] == "coding_or_utr"].copy()
    prov["n_coding"] = int(len(coding))

    # atlas orientation rule: pathogenicity = -score (validated in atlas.assay_aux)
    coding["rna_pathogenicity"] = -coding["rna"]

    rows: list[dict] = []
    cross_pg, agree_pg = [], []
    for gene, g in coding.groupby("gene"):
        r = _rho_row(g["rna_pathogenicity"], g["functional_pathogenicity"])
        if r:
            rows.append({"gene": gene, "comparison": "rna_vs_fitness",
                         "model": "(assay itself)", **r})
            agree_pg.append(r)

    for model in MODELS:
        if model not in coding.columns:
            continue
        pg_fit, pg_rna = [], []
        for gene, g in coding.groupby("gene"):
            rf = _rho_row(g[model], g["functional_pathogenicity"])
            rr = _rho_row(g[model], g["rna_pathogenicity"])
            if rf:
                rows.append({"gene": gene, "comparison": "model_vs_fitness",
                             "model": model, **rf})
                pg_fit.append(rf)
            if rr:
                rows.append({"gene": gene, "comparison": "model_vs_rna",
                             "model": model, **rr})
                pg_rna.append(rr)
        cross_pg.append({"model": model, "fit": _pool(pg_fit), "rna": _pool(pg_rna)})

    # RNA readout reliability, where replicates exist (BRCA1)
    rep = coding.dropna(subset=["rna_rep1", "rna_rep2"])
    if len(rep) >= MIN_N:
        rr = float(rep["rna_rep1"].corr(rep["rna_rep2"], method="spearman"))
        rel = spearman_brown(rr)
        prov["rna_reliability"] = {
            "gene": rep["gene"].iloc[0], "n": int(len(rep)),
            "rep_rho": rr, "reliability": rel, "ceiling": math.sqrt(max(rel, 0.0)),
        }

    prov["pooled_rna_vs_fitness"] = _pool(agree_pg)
    prov["pooled_by_model"] = [
        {"model": c["model"],
         "vs_fitness": c["fit"], "vs_rna": c["rna"],
         "drop": (c["fit"]["rho"] - c["rna"]["rho"]) if c["fit"] and c["rna"] else None}
        for c in cross_pg
    ]
    return pd.DataFrame(rows), prov


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # v2 carries the seven dbNSFP meta-predictors; v1 does not. The default was
    # v1, so a default run silently omitted those models from the delivered
    # table. v1 is kept only as the auditable pre-rescore matrix
    # (see atlas.matrix_v2).
    ap.add_argument("--matrix", default=str(RESULTS / "score_matrix_atlas_v2.parquet"))
    ap.add_argument("--aux", default=str(RESULTS / "assay_aux_v1.parquet"))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    mat = pd.read_parquet(args.matrix)
    # A declared model absent from the matrix used to be skipped in silence.
    absent = [m for m in MODELS if m not in mat.columns]
    if absent:
        print(f"[rna_readout] WARNING: {len(absent)} declared models are not in "
              f"this matrix and are omitted: {', '.join(absent)}", file=sys.stderr)
    aux = pd.read_parquet(args.aux)
    per_gene, prov = analyse(mat, aux)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_gene.to_csv(out / "rna_readout_v1.tsv", sep="\t", index=False)
    (out / "rna_readout_v1.json").write_text(json.dumps(prov, indent=2, default=str) + "\n")

    print(f"RNA readout available for {prov['n_with_rna']:,} variants "
          f"across {len(prov['genes'])} genes: {', '.join(prov['genes'])}")
    print("region coverage:", prov["region_coverage"])
    a = prov["pooled_rna_vs_fitness"]
    print(f"\nAssay-internal agreement (RNA vs fitness, coding only, k={a['k']}): "
          f"ρ = {a['rho']:.3f} ({a['ci_lo']:.3f}–{a['ci_hi']:.3f}), I² = {a['i2_pct']:.0f}%")
    if "rna_reliability" in prov:
        r = prov["rna_reliability"]
        print(f"RNA readout reliability ({r['gene']}, n={r['n']:,}): "
              f"{r['reliability']:.3f} → attenuation ceiling {r['ceiling']:.3f}")

    print(f"\n{'model':<24}{'vs fitness':>22}{'vs RNA':>22}{'drop':>8}")
    for e in sorted(prov["pooled_by_model"],
                    key=lambda x: -(x["vs_fitness"]["rho"] if x["vs_fitness"] else -9)):
        f, r = e["vs_fitness"], e["vs_rna"]
        fs = f"{f['rho']:.3f} ({f['ci_lo']:.3f}–{f['ci_hi']:.3f})" if f else "-"
        rs = f"{r['rho']:.3f} ({r['ci_lo']:.3f}–{r['ci_hi']:.3f})" if r else "-"
        ds = f"{e['drop']:+.3f}" if e["drop"] is not None else "-"
        print(f"{e['model']:<24}{fs:>22}{rs:>22}{ds:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
