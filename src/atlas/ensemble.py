"""Do the predictors carry complementary information, and is a territory-aware
choice worth making?

"No model everywhere" invites the obvious follow-up: then what should be used?
Two things are needed to answer it honestly.

*Complementarity.* If the predictors were merely noisy copies of one another,
combining them could not help. The between-model rank correlation matrix, per
territory, says how much independent information is on the table.

*An ensemble that is not fitted on its own test set.* Picking "the best model
per territory" from the full table and then reporting that model's number is
circular. Here the selection is made **leave-one-gene-out**: for each held-out
gene the winner per stratum is chosen using the other six genes only, and scored
on the held-out gene. That is the number a user would actually get by following
the paper's selection guide on a new gene.

Three strategies are compared on identical variants:
  ``single_global``     one model everywhere (chosen LOGO)
  ``territory_logo``    best model per territory (chosen LOGO)
  ``rank_average``      mean within-gene percentile rank over a fixed panel

Usage (PYTHONPATH=src):  python -m atlas.ensemble --out results/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.evaluate import classify_region, dl_pool, fisher_z

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

STRATA = ["all", "coding_or_utr", "splice_region",
          "splice_1_2", "splice_3_10", "splice_11_50", "splice_deep"]

# Panels are fixed a priori, not chosen by performance: broad-coverage SNV
# scorers, and the three splice-aware models.
PANEL_BROAD = ["cadd", "evo2", "gpn_msa", "phylop100way"]
PANEL_SPLICE = ["alphagenome", "spliceai_ds", "pangolin_score"]

CANDIDATES = PANEL_BROAD + PANEL_SPLICE + ["phastcons100way", "nucleotide_transformer"]

MIN_N = 30


def _mask(region: pd.Series, stratum: str) -> pd.Series:
    if stratum == "all":
        return pd.Series(True, index=region.index)
    if stratum == "splice_region":
        return region.str.startswith("splice_")
    return region == stratum


def add_rank_ensembles(df: pd.DataFrame) -> pd.DataFrame:
    """Within-gene percentile rank averages over the two fixed panels."""
    df = df.copy()
    for name, panel in (("ens_broad", PANEL_BROAD), ("ens_splice", PANEL_SPLICE)):
        pct = pd.DataFrame(index=df.index)
        for m in panel:
            pct[m] = df.groupby("gene")[m].rank(pct=True)
        # require every panel member, so the ensemble is never a different
        # variant set from its own members
        df[name] = pct.mean(axis=1).where(pct.notna().all(axis=1))
    return df


def per_gene_rho(df: pd.DataFrame, col: str, stratum: str) -> pd.DataFrame:
    s = df[_mask(df["region"], stratum)].dropna(subset=[col, "functional_pathogenicity"])
    rows = []
    for gene, g in s.groupby("gene"):
        if len(g) < MIN_N:
            continue
        r = float(g[col].corr(g["functional_pathogenicity"], method="spearman"))
        z, v = fisher_z(r, len(g))
        rows.append({"gene": gene, "n": len(g), "rho": r, "z": z, "var": v})
    return pd.DataFrame(rows)


def pooled(pg: pd.DataFrame) -> float:
    if len(pg) < 2:
        return float("nan")
    return math.tanh(dl_pool(pg["z"].tolist(), pg["var"].tolist())["pooled"])


def complementarity(df: pd.DataFrame, stratum: str) -> pd.DataFrame:
    s = df[_mask(df["region"], stratum)]
    cols = [c for c in CANDIDATES if c in s.columns]
    return s[cols].corr(method="spearman")


def logo_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """Held-out-gene performance of each strategy, per stratum."""
    genes = sorted(df["gene"].unique())
    rows = []
    for stratum in STRATA:
        # per-gene ρ for every candidate, computed once
        pg = {c: per_gene_rho(df, c, stratum).set_index("gene")
              for c in CANDIDATES if c in df.columns}
        pg = {c: t for c, t in pg.items() if len(t) >= 3}
        if not pg:
            continue
        ens = {name: per_gene_rho(df, name, stratum).set_index("gene")
               for name in ("ens_broad", "ens_splice")}

        for held in genes:
            train = {c: t.drop(index=held, errors="ignore") for c, t in pg.items()}
            train = {c: t for c, t in train.items() if len(t) >= 2}
            if not train:
                continue
            pooled_train = {c: pooled(t.reset_index()) for c, t in train.items()}
            pooled_train = {c: v for c, v in pooled_train.items() if np.isfinite(v)}
            if not pooled_train:
                continue
            winner = max(pooled_train, key=pooled_train.get)

            def held_rho(col: str) -> float:
                t = pg.get(col) if col in pg else ens.get(col)
                if t is None or held not in t.index:
                    return float("nan")
                return float(t.loc[held, "rho"])

            rows.append({
                "stratum": stratum, "held_out_gene": held,
                "territory_logo_model": winner,
                "territory_logo": held_rho(winner),
                "single_global_cadd": held_rho("cadd"),
                "rank_average_broad": (float(ens["ens_broad"].loc[held, "rho"])
                                       if held in ens["ens_broad"].index else np.nan),
                "rank_average_splice": (float(ens["ens_splice"].loc[held, "rho"])
                                        if held in ens["ens_splice"].index else np.nan),
                "best_possible_oracle": max(
                    (held_rho(c) for c in pg if held in pg[c].index), default=np.nan),
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=str(RESULTS / "score_matrix_atlas_v1.parquet"))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.matrix)
    df["region"] = df["hgvs_c"].map(classify_region)
    df = add_rank_ensembles(df)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # pooled ρ for the ensembles alongside their members
    pooled_rows = []
    for stratum in STRATA:
        for col in CANDIDATES + ["ens_broad", "ens_splice"]:
            if col not in df.columns:
                continue
            pg = per_gene_rho(df, col, stratum)
            if len(pg) < 2:
                continue
            pooled_rows.append({"stratum": stratum, "score": col, "k_genes": len(pg),
                                "n": int(pg["n"].sum()), "pooled_rho": pooled(pg)})
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(out / "ensemble_pooled_v1.tsv", sep="\t", index=False)

    logo = logo_strategies(df)
    logo.to_csv(out / "ensemble_logo_v1.tsv", sep="\t", index=False)

    for stratum in ("all", "coding_or_utr", "splice_region"):
        cm = complementarity(df, stratum)
        cm.to_csv(out / f"model_correlation_{stratum}_v1.tsv", sep="\t")

    print("Pooled ρ — fixed-panel rank-average ensembles vs their members:")
    piv = pooled_df.pivot(index="score", columns="stratum", values="pooled_rho")
    piv = piv.reindex([c for c in CANDIDATES + ["ens_broad", "ens_splice"] if c in piv.index])
    print(piv[[s for s in STRATA if s in piv.columns]]
          .to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nBetween-model rank correlation, coding/UTR (complementarity):")
    cm = complementarity(df, "coding_or_utr")
    print(cm.round(2).to_string())

    print("\nHeld-out-gene performance of each selection strategy "
          "(median over the 7 held-out genes):")
    summ = (logo.groupby("stratum")[["single_global_cadd", "territory_logo",
                                     "rank_average_broad", "rank_average_splice",
                                     "best_possible_oracle"]]
                .median().reindex([s for s in STRATA if s in set(logo["stratum"])]))
    print(summ.to_string(float_format=lambda x: f"{x:.3f}"))

    picks = (logo.groupby(["stratum", "territory_logo_model"]).size()
                 .rename("times_chosen").reset_index())
    print("\nWhich model the LOGO selector picks per stratum:")
    print(picks.to_string(index=False))

    (out / "ensemble_v1.json").write_text(json.dumps({
        "panel_broad": PANEL_BROAD, "panel_splice": PANEL_SPLICE,
        "strategy_medians": summ.to_dict("index"),
    }, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
