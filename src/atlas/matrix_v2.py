"""Assemble score matrix v2: full-precision splice scores + meta-predictors.

Three changes relative to ``score_matrix_atlas_v1.parquet``, which is left
untouched (project rule 1 — produce a new versioned file, never edit in place):

1. ``spliceai_ds`` and ``pangolin_score`` are replaced by the full-precision
   re-scores. These are the *same* score definitions: both tools compute in
   float and round only when formatting output, and both re-scores reproduce the
   v1 columns exactly when re-rounded to two decimals. The v1 values are kept as
   ``*_cli_rounded`` so the comparison stays auditable.
2. Seven dbNSFP meta-predictors are added (missense-only; ``atlas.metapredictors``).
3. The locally derived protein consequence is carried alongside, so downstream
   modules need not re-merge it.

Usage (PYTHONPATH=src):  python -m atlas.matrix_v2 --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

REPLACEMENTS = {
    "spliceai_ds": ("spliceai_scores_fullprec.parquet", "spliceai_ds_fullprec"),
    "pangolin_score": ("pangolin_scores_fullprec.parquet", "pangolin_fullprec"),
}
META = ["revel", "bayesdel_addaf", "clinpred", "metarnn", "primateai", "vest4", "esm1b"]


def build(results: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(results / "score_matrix_atlas_v1.parquet")
    prov: dict = {"source": "score_matrix_atlas_v1.parquet", "replacements": {}}

    for col, (fname, src_col) in REPLACEMENTS.items():
        fp = pd.read_parquet(results / fname)[["variant_id", src_col]]
        merged = df[["variant_id", col]].merge(fp, on="variant_id", how="left")
        both = merged.dropna(subset=[col, src_col])
        max_dev = float((both[src_col].round(2) - both[col]).abs().max())
        if max_dev > 5e-3:
            raise ValueError(
                f"{col}: full-precision re-score does not reproduce v1 when rounded "
                f"(max deviation {max_dev:.4f}); refusing to substitute")
        df[f"{col}_cli_rounded"] = df[col]
        df[col] = merged[src_col].to_numpy()
        prov["replacements"][col] = {
            "source_file": fname,
            "n_scored": int(merged[src_col].notna().sum()),
            "max_rounding_deviation": max_dev,
            "distinct_before": int(merged[col].nunique()),
            "distinct_after": int(merged[src_col].nunique()),
        }

    meta = pd.read_parquet(results / "metapredictor_scores_v1.parquet")
    df = df.merge(meta[["variant_id"] + META], on="variant_id", how="left")
    prov["metapredictors"] = {c: int(df[c].notna().sum()) for c in META}

    cons = pd.read_parquet(results / "consequence_v1.parquet")[["variant_id", "consequence"]]
    df = df.merge(cons, on="variant_id", how="left")
    prov["shape"] = list(df.shape)
    return df, prov


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out)

    df, prov = build(out)
    df.to_parquet(out / "score_matrix_atlas_v2.parquet", index=False)
    (out / "score_matrix_atlas_v2.provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n")

    print(f"score_matrix_atlas_v2.parquet: {df.shape[0]:,} x {df.shape[1]}")
    for col, rec in prov["replacements"].items():
        print(f"  {col}: {rec['n_scored']:,} full-precision values, "
              f"{rec['distinct_before']} -> {rec['distinct_after']:,} distinct, "
              f"max rounding deviation {rec['max_rounding_deviation']:.4f}")
    print("  meta-predictors:", ", ".join(f"{k} {v:,}" for k, v in prov["metapredictors"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
