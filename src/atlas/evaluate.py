"""Evaluation: per-assay Spearman correlation + DerSimonian–Laird meta-analysis.

For each model score column, compute the per-gene Spearman ρ between the model
score and functional_pathogenicity, then pool across genes with a
DerSimonian–Laird random-effects meta-analysis on Fisher-z transformed
correlations (matching the estimator of the companion benchmark).

Strata:
  - all: every scored variant
  - splice_region: all intron-side splice variants (aggregate)
  - splice_1_2 / splice_3_10 / splice_11_50 / splice_deep: intronic offset
    buckets — offset composition drives apparent performance, and canonical
    dinucleotides saturate functionally (see mapping reports)
  - coding_or_utr
  - clinvar_recorded: variants carrying a ClinVar record (requires --clinvar),
    matching the companion benchmark's evaluation-set construction

Usage (PYTHONPATH=src):

    python -m atlas.evaluate --scores results/alphagenome_scores.parquet \
        --model alphagenome --out results/ [--clinvar labels.parquet]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Intronic offset notation: c.5278-12C>G (offset 12), c.5467+20C>A (offset 20)
_OFFSET = re.compile(r"c\.\*?\d+[+-](\d+)")


def splice_offset(hgvs_c: str) -> float:
    """Intronic offset in bp, or NaN for non-intronic-notation variants."""
    if not isinstance(hgvs_c, str):
        return float("nan")
    m = _OFFSET.search(hgvs_c)
    return float(m.group(1)) if m else float("nan")


def classify_region(hgvs_c: str) -> str:
    """Fine region bucket from the c. HGVS string."""
    off = splice_offset(hgvs_c)
    if math.isnan(off):
        return "coding_or_utr"
    if off <= 2:
        return "splice_1_2"
    if off <= 10:
        return "splice_3_10"
    if off <= 50:
        return "splice_11_50"
    return "splice_deep"


def fisher_z(r: float, n: int) -> tuple[float, float]:
    """Fisher z transform and its variance. Guards |r| -> 1."""
    r = float(min(max(r, -0.999999), 0.999999))
    return math.atanh(r), 1.0 / (n - 3)


def dl_pool(ests: list[float], variances: list[float]) -> dict:
    """DerSimonian–Laird random-effects pooling (and fixed-effect for reference)."""
    k = len(ests)
    if k == 0:
        raise ValueError("nothing to pool")
    e = np.asarray(ests, dtype=float)
    v = np.asarray(variances, dtype=float)
    w = 1.0 / v
    fixed = float(np.sum(w * e) / np.sum(w))
    q = float(np.sum(w * (e - fixed) ** 2))
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    wr = 1.0 / (v + tau2)
    pooled = float(np.sum(wr * e) / np.sum(wr))
    se = math.sqrt(1.0 / np.sum(wr))
    i2 = max(0.0, (q - (k - 1)) / q) * 100 if q > 0 else 0.0
    return {
        "k": k,
        "pooled": pooled,
        "se": se,
        "ci_lo": pooled - 1.96 * se,
        "ci_hi": pooled + 1.96 * se,
        "tau2": tau2,
        "q": q,
        "i2_pct": i2,
    }


def per_gene_spearman(df: pd.DataFrame, model: str, min_n: int = 30) -> pd.DataFrame:
    """Per-gene Spearman ρ of model score vs functional_pathogenicity."""
    rows = []
    for gene, sub in df.groupby("gene"):
        sub = sub.dropna(subset=[model, "functional_pathogenicity"])
        n = len(sub)
        if n < min_n:
            rows.append({"gene": gene, "n": n, "rho": np.nan, "z": np.nan, "var": np.nan, "note": "below min_n"})
            continue
        rho = float(sub[model].corr(sub["functional_pathogenicity"], method="spearman"))
        z, var = fisher_z(rho, n)
        rows.append({"gene": gene, "n": n, "rho": rho, "z": z, "var": var, "note": ""})
    if not rows:
        return pd.DataFrame(columns=["gene", "n", "rho", "z", "var", "note"])
    return pd.DataFrame(rows)


def _pool_to_rho(pooled: dict | None) -> dict | None:
    if pooled is None:
        return None
    pooled["pooled_rho"] = math.tanh(pooled["pooled"])
    pooled["rho_ci_lo"] = math.tanh(pooled["ci_lo"])
    pooled["rho_ci_hi"] = math.tanh(pooled["ci_hi"])
    return pooled


def evaluate(
    df: pd.DataFrame,
    model: str,
    min_n: int = 30,
    clinvar_ids: set | None = None,
) -> dict:
    """Overall + region-stratified (+ optional ClinVar-subset) evaluation."""
    out = {"model": model, "min_n": min_n, "strata": {}}
    df = df.copy()
    df["region"] = df["hgvs_c"].map(classify_region)

    strata = {
        "all": df,
        "splice_region": df[df["region"].str.startswith("splice_")],
        "coding_or_utr": df[df["region"] == "coding_or_utr"],
    }
    for bucket in ("splice_1_2", "splice_3_10", "splice_11_50", "splice_deep"):
        strata[bucket] = df[df["region"] == bucket]
    if clinvar_ids is not None:
        strata["clinvar_recorded"] = df[df["variant_id"].isin(clinvar_ids)]

    for name, sub in strata.items():
        per_gene = per_gene_spearman(sub, model, min_n=min_n)
        ok = per_gene.dropna(subset=["rho"])
        pooled = dl_pool(ok["z"].tolist(), ok["var"].tolist()) if len(ok) else None
        out["strata"][name] = {
            "per_gene": per_gene.drop(columns=["z", "var"]).to_dict("records"),
            "pooled": _pool_to_rho(pooled),
            "n_variants": int(len(sub)),
            "n_genes_used": int(len(ok)),
        }
    return out


def forest_plot(result: dict, path: Path, stratum: str = "all") -> None:
    """Per-gene ρ forest plot with pooled estimate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = result["strata"][stratum]
    rows = [r for r in st["per_gene"] if r.get("rho") is not None and not math.isnan(r["rho"])]
    if not rows or not st["pooled"]:
        return
    rows = sorted(rows, key=lambda r: r["rho"])
    genes = [r["gene"] for r in rows] + ["Pooled (DL)"]
    rhos = [r["rho"] for r in rows] + [st["pooled"]["pooled_rho"]]
    los = [r["rho"] - 1.96 * math.sqrt(1 / (r["n"] - 3)) for r in rows] + [st["pooled"]["rho_ci_lo"]]
    his = [r["rho"] + 1.96 * math.sqrt(1 / (r["n"] - 3)) for r in rows] + [st["pooled"]["rho_ci_hi"]]

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(genes) + 1.5))
    y = np.arange(len(genes))
    colors = ["#78909C"] * len(rows) + ["#2F4B5C"]
    ax.errorbar(rhos, y, xerr=[np.array(rhos) - np.array(los), np.array(his) - np.array(rhos)],
                fmt="none", ecolor="#B9C6CE", zorder=1)
    ax.scatter(rhos, y, c=colors, s=45, zorder=2)
    ax.axvline(0, color="#999999", lw=0.8, ls="--")
    ax.set_yticks(y, genes)
    ax.set_xlabel(f"Spearman ρ vs functional pathogenicity ({stratum})")
    ax.set_title(f"{result['model']} — per-gene correlation with functional scores")
    i2 = st["pooled"]["i2_pct"]
    ax.text(0.99, 0.02, f"I² = {i2:.0f}%", transform=ax.transAxes, ha="right", color="#6B7B8C")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, help="model scores parquet")
    ap.add_argument("--model", required=True, help="score column name")
    ap.add_argument("--out", default="results")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument(
        "--clinvar",
        help="parquet/csv with a variant_id column of ClinVar-recorded variants "
        "(adds the clinvar_recorded stratum, matching the companion benchmark)",
    )
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.scores)
    clinvar_ids = None
    if args.clinvar:
        cv = (
            pd.read_parquet(args.clinvar)
            if args.clinvar.endswith(".parquet")
            else pd.read_csv(args.clinvar)
        )
        if "variant_id" not in cv.columns:
            raise SystemExit("--clinvar file must contain a variant_id column")
        clinvar_ids = set(cv["variant_id"])

    result = evaluate(df, args.model, min_n=args.min_n, clinvar_ids=clinvar_ids)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{args.model}.json").write_text(json.dumps(result, indent=2) + "\n")
    for stratum, st in result["strata"].items():
        if st["pooled"]:
            forest_plot(result, out_dir / f"forest_{args.model}_{stratum}.png", stratum)

    for name, st in result["strata"].items():
        p = st["pooled"]
        if p:
            print(
                f"[{name:16s}] genes={st['n_genes_used']} variants={st['n_variants']:>6} "
                f"pooled ρ={p['pooled_rho']:.3f} (95% CI {p['rho_ci_lo']:.3f}–{p['rho_ci_hi']:.3f}) "
                f"I²={p['i2_pct']:.0f}%"
            )
        else:
            print(f"[{name:16s}] variants={st['n_variants']:>6} (insufficient per-gene n)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
