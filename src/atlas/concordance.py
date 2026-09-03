"""Concordance between two scoring runs of the same variants.

Used to quantify agreement between the companion benchmark's AlphaGenome
definition (client v0.6.1, max|raw_score|, 1-Mb interval) and the atlas
standard (client 0.7.0, merged quantile score, 16-kb interval) — see
models/alphagenome/NOTES.md, DECISION section.

Merges on genomic identity (chrom, pos, ref, alt), reports overall and
per-gene Spearman ρ between the two score columns, and writes a scatter plot.

Usage (PYTHONPATH=src):

    python -m atlas.concordance --old old_scores.parquet --old-col alphagenome \
        --new results/alphagenome_scores.parquet --new-col alphagenome \
        --out results/concordance_alphagenome
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

KEY = ["chrom", "pos", "ref", "alt"]


def concordance(old: pd.DataFrame, old_col: str, new: pd.DataFrame, new_col: str) -> dict:
    for col in KEY:
        old[col] = old[col].astype(str) if col == "chrom" else old[col]
        new[col] = new[col].astype(str) if col == "chrom" else new[col]
    merged = old[KEY + [old_col, "gene"]].merge(
        new[KEY + [new_col]], on=KEY, how="inner", suffixes=("_old", "_new")
    )
    if merged.empty:
        raise ValueError("no shared variants between old and new score files")
    old_name = f"{old_col}_old" if old_col == new_col else old_col
    new_name = f"{new_col}_new" if old_col == new_col else new_col

    def _rho(sub: pd.DataFrame) -> float:
        return float(sub[old_name].corr(sub[new_name], method="spearman"))

    per_gene = {g: _rho(s) for g, s in merged.groupby("gene") if len(s) >= 30}
    return {
        "n_shared": int(len(merged)),
        "overall_rho": _rho(merged),
        "per_gene_rho": per_gene,
        "merged": merged,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", required=True)
    ap.add_argument("--old-col", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--new-col", required=True)
    ap.add_argument("--out", required=True, help="output prefix (json + png)")
    args = ap.parse_args(argv)

    read = lambda p: pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    res = concordance(read(args.old), args.old_col, read(args.new), args.new_col)
    merged = res.pop("merged")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    xcol = f"{args.old_col}_old" if args.old_col == args.new_col else args.old_col
    ycol = f"{args.new_col}_new" if args.old_col == args.new_col else args.new_col
    ax.scatter(merged[xcol], merged[ycol], s=3, alpha=0.3, c="#2F4B5C")
    ax.set_xlabel(f"old: {args.old_col}")
    ax.set_ylabel(f"new: {args.new_col}")
    # Four places, not three: the Nucleotide Transformer concordance is 0.99969,
    # which three places round to 1.000 -- a figure panel claiming a perfect
    # correlation while the text and Note S2 say 0.9997. figures/reproducibility.py
    # already prints four; this is the panel that feeds Supplemental Fig. S13.
    ax.set_title(f"score concordance (Spearman ρ = {res['overall_rho']:.4f}, n = {res['n_shared']})")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)

    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
