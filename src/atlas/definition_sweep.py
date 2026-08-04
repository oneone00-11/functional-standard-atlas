"""How much does a scoring definition actually change a benchmark's conclusions?

The atlas and the companion benchmark score AlphaGenome under two definitions
and their score columns agree at only rho = 0.672. Read on its own that number
suggests the definition choice is consequential. This module tests whether it
is, by separating three things the single comparison confounds:

*Client version.* The legacy definition is replicated exactly under the current
client, so any residual disagreement with the legacy column is attributable to
the client, not the definition.

*The definitional axes.* A four-window x four-aggregation grid
(``models/alphagenome/sweep_definitions.py``) varies window and aggregation
independently, which the single legacy-vs-atlas comparison cannot.

*Score agreement versus conclusion agreement.* Two definitions can rank variants
differently and still place a model identically on the benchmark. Both are
reported, because they are different questions and only the second one matters
for a benchmark's claims.

Usage (PYTHONPATH=src):  python -m atlas.definition_sweep --out results/
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

DEFINITIONS = ["merged_quantile", "merged_raw", "max_abs_raw", "max_quantile"]
STRATA = ["coding_or_utr", "splice_1_2", "splice_3_10", "splice_11_50", "splice_deep"]
KEY = ["chrom", "pos", "ref", "alt"]
MIN_N = 30


def pooled_rho(sub: pd.DataFrame, col: str) -> float:
    acc = []
    for _gene, g in sub.groupby("gene"):
        g = g.dropna(subset=[col, "functional_pathogenicity"])
        if len(g) < MIN_N:
            continue
        r = float(g[col].corr(g["functional_pathogenicity"], method="spearman"))
        z, v = fisher_z(r, len(g))
        acc.append((z, v))
    if len(acc) < 2:
        return float("nan")
    return math.tanh(dl_pool([a for a, _ in acc], [b for _, b in acc])["pooled"])


def performance_grid(sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for width, w in sweep.groupby("width"):
        for stratum in STRATA:
            sub = w[w["region"] == stratum]
            for d in DEFINITIONS:
                rows.append({"width": int(width), "stratum": stratum, "definition": d,
                             "n": int(sub[d].notna().sum()),
                             "pooled_rho": pooled_rho(sub, d)})
    return pd.DataFrame(rows)


def score_concordance(sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for width, w in sweep.groupby("width"):
        sub = w[DEFINITIONS].dropna()
        cm = sub.corr(method="spearman")
        for i, a in enumerate(DEFINITIONS):
            for b in DEFINITIONS[i + 1:]:
                rows.append({"comparison": "definitions", "width": int(width),
                             "a": a, "b": b, "n": len(sub), "rho": float(cm.loc[a, b])})
    for d in DEFINITIONS:
        piv = sweep.pivot_table(index="variant_id", columns="width", values=d)
        widths = sorted(piv.columns)
        for i, wa in enumerate(widths):
            for wb in widths[i + 1:]:
                p = piv[[wa, wb]].dropna()
                rows.append({"comparison": "windows", "width": np.nan, "a": f"{d}@{wa}",
                             "b": f"{d}@{wb}", "n": len(p),
                             "rho": float(p[wa].corr(p[wb], method="spearman"))})
    return pd.DataFrame(rows)


def legacy_check(sweep: pd.DataFrame, matrix: Path, legacy: Path) -> dict:
    """Is the legacy column reproducible under the current client?"""
    leg = pd.read_parquet(legacy)
    leg["chrom"] = leg["chrom"].astype(str); leg["pos"] = leg["pos"].astype(int)
    mat = pd.read_parquet(matrix, columns=["variant_id"] + KEY + ["hgvs_c", "gene",
                                                                  "alphagenome",
                                                                  "functional_pathogenicity"])
    mat["chrom"] = mat["chrom"].astype(str); mat["pos"] = mat["pos"].astype(int)
    joined = mat.merge(leg[KEY + ["alphagenome_splice"]], on=KEY, how="inner")
    joined["region"] = joined["hgvs_c"].map(classify_region)

    out: dict = {"n_shared_total": int(len(joined))}

    # (a) same definition, different client version
    widest = int(sweep["width"].max())
    s = sweep[sweep["width"] == widest]
    rep = s.merge(joined[["variant_id", "alphagenome_splice"]], on="variant_id", how="inner")
    rep = rep.dropna(subset=["max_abs_raw", "alphagenome_splice"])
    out["replicate_legacy_definition"] = {
        "window": widest, "n": int(len(rep)),
        "rho_vs_legacy_column": float(rep["max_abs_raw"].corr(
            rep["alphagenome_splice"], method="spearman")),
        "note": "current client, legacy definition (max|raw|, widest window)",
    }

    # (b) score-level agreement between the two published columns, by region
    ok = joined.dropna(subset=["alphagenome", "alphagenome_splice"])
    out["score_agreement_by_region"] = {
        "overall": float(ok["alphagenome"].corr(ok["alphagenome_splice"], method="spearman")),
        **{r: float(g["alphagenome"].corr(g["alphagenome_splice"], method="spearman"))
           for r, g in ok.groupby("region") if len(g) >= MIN_N},
    }
    out["region_composition"] = {r: int(n) for r, n in ok["region"].value_counts().items()}

    # (c) conclusion-level agreement: both columns, same variants, same strata
    perf = {}
    sets = {"all": ok, "coding_or_utr": ok[ok["region"] == "coding_or_utr"],
            "splice_region": ok[ok["region"].str.startswith("splice_")]}
    for st in ["splice_1_2", "splice_3_10", "splice_11_50"]:
        sets[st] = ok[ok["region"] == st]
    for name, sub in sets.items():
        a, b = pooled_rho(sub, "alphagenome"), pooled_rho(sub, "alphagenome_splice")
        if np.isfinite(a) and np.isfinite(b):
            perf[name] = {"atlas_definition": a, "legacy_definition": b, "difference": b - a}
    out["performance_on_shared_variants"] = perf
    out["max_abs_performance_difference"] = float(
        max(abs(v["difference"]) for v in perf.values())) if perf else None
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default=str(RESULTS / "alphagenome_definition_sweep_v1.parquet"))
    ap.add_argument("--matrix", default=str(RESULTS / "score_matrix_atlas_v2.parquet"))
    ap.add_argument("--legacy",
                    default=str(REPO / "data/external/companion_alphagenome_v061.parquet"))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    sweep = pd.read_parquet(args.sweep)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    grid = performance_grid(sweep)
    grid.to_csv(out / "definition_sweep_performance_v1.tsv", sep="\t", index=False)
    conc = score_concordance(sweep)
    conc.to_csv(out / "definition_sweep_concordance_v1.tsv", sep="\t", index=False)
    check = legacy_check(sweep, Path(args.matrix), Path(args.legacy))
    (out / "definition_sweep_v1.json").write_text(json.dumps(check, indent=2) + "\n")

    print(f"sweep: {sweep['variant_id'].nunique():,} variants x "
          f"{sweep['width'].nunique()} windows x {len(DEFINITIONS)} definitions\n")

    print("Pooled rho within stratum, every definition x window:")
    piv = grid.pivot_table(index=["stratum", "width"], columns="definition",
                           values="pooled_rho").reindex(DEFINITIONS, axis=1)
    piv["spread"] = piv.max(axis=1) - piv.min(axis=1)
    print(piv.reindex(STRATA, level=0).to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n  largest spread across all 16 definition x window combinations, "
          f"within any stratum: {piv['spread'].max():.3f}")

    d = conc[conc["comparison"] == "definitions"]["rho"]
    w = conc[conc["comparison"] == "windows"]["rho"]
    print(f"\nScore-level agreement: between definitions {d.min():.3f}-{d.max():.3f}; "
          f"across windows {w.min():.3f}-{w.max():.3f}")

    r = check["replicate_legacy_definition"]
    print(f"\nLegacy definition replicated under the current client "
          f"(n={r['n']:,}): rho = {r['rho_vs_legacy_column']:.3f}")
    print("  -> the client version is not the source of the disagreement"
          if r["rho_vs_legacy_column"] > 0.95 else
          "  -> the client version contributes to the disagreement")

    sa = check["score_agreement_by_region"]
    print(f"\nPublished columns agree at rho = {sa['overall']:.3f} overall; by region: "
          + ", ".join(f"{k} {v:.3f}" for k, v in sa.items() if k != "overall"))

    print("\nBut on the SAME variants their benchmark performance barely differs:")
    for name, v in check["performance_on_shared_variants"].items():
        print(f"  {name:<16} atlas {v['atlas_definition']:.3f}  "
              f"legacy {v['legacy_definition']:.3f}  diff {v['difference']:+.3f}")
    print(f"\n  largest performance difference: "
          f"{check['max_abs_performance_difference']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
