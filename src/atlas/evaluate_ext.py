"""Extended stratification of the territory map.

``atlas.evaluate`` scores eight strata. Three questions it cannot answer, each
of which a reader will ask:

*ClinVar ascertainment.* The ClinVar-recorded stratum was compared against
``all``, but ``all`` *contains* the recorded variants, so the comparison is
diluted. The informative contrast is recorded versus **absent**.

*Variant type.* 17,786 indels sit in the atlas and are never evaluated as their
own stratum, even though four predictors score them.

*What drives the coding signal.* A predictor can look strong in coding/UTR
simply by separating nonsense from missense variants. Stratifying by protein
consequence (from ``atlas.consequence``) tests whether the signal survives
inside a single consequence class.

Everything else — per-gene Spearman, minimum n, DerSimonian–Laird pooling on
Fisher-z — is unchanged and imported from ``atlas.evaluate``, so the extended
numbers are directly comparable to the originals.

Usage (PYTHONPATH=src):  python -m atlas.evaluate_ext --out results/
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.evaluate import classify_region, dl_pool, fisher_z

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

MODELS = [
    "alphagenome", "spliceai_ds", "pangolin_score", "cadd", "alphamissense",
    "evo2", "gpn_msa", "nucleotide_transformer", "phylop100way",
    "phastcons100way", "gnomad_af_global", "gnomad_af_popmax",
    # dbNSFP meta-predictors (missense-only; atlas.metapredictors)
    "revel", "bayesdel_addaf", "clinpred", "metarnn", "primateai", "vest4", "esm1b",
]

MIN_N = 30


def stratum_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    region = df["region"]
    cons = df["consequence"]
    is_snv = df["ref"].str.len().eq(1) & df["alt"].str.len().eq(1)
    m: dict[str, pd.Series] = {
        # original eight, recomputed so the extended table is self-contained
        "all": pd.Series(True, index=df.index),
        "coding_or_utr": region.eq("coding_or_utr"),
        "splice_region": region.str.startswith("splice_"),
        "splice_1_2": region.eq("splice_1_2"),
        "splice_3_10": region.eq("splice_3_10"),
        "splice_11_50": region.eq("splice_11_50"),
        "splice_deep": region.eq("splice_deep"),
        "clinvar_recorded": df["clinvar"],
        # new
        "clinvar_absent": ~df["clinvar"],
        "snv": is_snv,
        "indel": ~is_snv,
        "missense": cons.eq("missense"),
        "synonymous": cons.eq("synonymous"),
        "nonsense": cons.eq("nonsense"),
        "utr": cons.isin(["utr5", "utr3"]),
        # protein-truncating vs not, within coding SNVs
        "coding_snv_nontruncating": cons.isin(["missense", "synonymous"]),
    }
    return m


def evaluate_grid(df: pd.DataFrame) -> pd.DataFrame:
    masks = stratum_masks(df)
    rows = []
    for model in MODELS:
        if model not in df.columns:
            continue
        for stratum, mask in masks.items():
            sub = df[mask].dropna(subset=[model, "functional_pathogenicity"])
            per_gene = []
            for gene, g in sub.groupby("gene"):
                if len(g) < MIN_N:
                    continue
                r = float(g[model].corr(g["functional_pathogenicity"], method="spearman"))
                z, v = fisher_z(r, len(g))
                per_gene.append({"gene": gene, "n": len(g), "rho": r, "z": z, "var": v})
            if len(per_gene) < 2:
                rows.append({"model": model, "stratum": stratum, "k_genes": len(per_gene),
                             "n": int(len(sub)), "pooled_rho": np.nan, "ci_lo": np.nan,
                             "ci_hi": np.nan, "i2_pct": np.nan})
                continue
            pg = pd.DataFrame(per_gene)
            p = dl_pool(pg["z"].tolist(), pg["var"].tolist())
            rows.append({
                "model": model, "stratum": stratum, "k_genes": len(pg),
                "n": int(len(sub)),
                "pooled_rho": math.tanh(p["pooled"]),
                "ci_lo": math.tanh(p["ci_lo"]),
                "ci_hi": math.tanh(p["ci_hi"]),
                "i2_pct": p["i2_pct"],
            })
    return pd.DataFrame(rows)


def load(matrix: Path, consequence: Path, clinvar: Path) -> pd.DataFrame:
    df = pd.read_parquet(matrix)
    cv = set(pd.read_parquet(clinvar)["variant_id"])
    # matrix v2 already carries the consequence column; merging again would
    # produce consequence_x/consequence_y and silently break every
    # consequence-based stratum
    if "consequence" not in df.columns:
        cons = pd.read_parquet(consequence)[["variant_id", "consequence"]]
        df = df.merge(cons, on="variant_id", how="left")
    df["region"] = df["hgvs_c"].map(classify_region)
    df["clinvar"] = df["variant_id"].isin(cv)
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=str(RESULTS / "score_matrix_atlas_v1.parquet"))
    ap.add_argument("--consequence", default=str(RESULTS / "consequence_v1.parquet"))
    ap.add_argument("--clinvar",
                    default=str(REPO / "data/external/clinvar_recorded_ids.parquet"))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    df = load(Path(args.matrix), Path(args.consequence), Path(args.clinvar))
    grid = evaluate_grid(df)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "eval_ext_v1.tsv", sep="\t", index=False)

    wide = grid.pivot(index="model", columns="stratum", values="pooled_rho")
    order = [m for m in MODELS if m in wide.index]

    print("Stratum sizes:")
    for s, mask in stratum_masks(df).items():
        print(f"  {s:<26} {int(mask.sum()):>7,}")

    print("\nClinVar ascertainment — recorded vs absent (the diluted 'all' comparison "
          "is shown for reference):")
    cv = wide.reindex(order)[["clinvar_recorded", "clinvar_absent", "all"]].copy()
    cv["uplift_vs_absent"] = cv["clinvar_recorded"] - cv["clinvar_absent"]
    cv["uplift_vs_all"] = cv["clinvar_recorded"] - cv["all"]
    print(cv.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n  median uplift vs absent = {cv['uplift_vs_absent'].median():.3f}   "
          f"vs all = {cv['uplift_vs_all'].median():.3f}")

    print("\nWithin-consequence-class signal (does coding performance survive "
          "removing the nonsense/missense contrast?):")
    print(wide.reindex(order)[["coding_or_utr", "missense", "synonymous", "nonsense", "utr"]]
          .to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nSNV vs indel:")
    print(wide.reindex(order)[["snv", "indel"]].to_string(float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
