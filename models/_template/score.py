"""Model scorer skeleton.

Contract (see README.md in this directory):
    python score.py --input <variant_table.parquet> --output scores.parquet

Input : variant_id, chrom, pos, ref, alt, transcript, hgvs_c  (GRCh38)
Output: all input columns + one score column named after this model.
        Larger score = more damaging. Unscoreable variants are dropped.
"""

from __future__ import annotations

import argparse

import pandas as pd

MODEL_NAME = "template"  # TODO: rename to your model; this becomes the score column


def score_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Return df plus the model's score column. Implement the real call here.

    TODO: replace the placeholder below with the actual model invocation
    (local checkpoint, API client, or subprocess). Keep it deterministic:
    fix seeds, pin versions, record the model version string.
    """
    raise NotImplementedError("implement the scorer before running")
    # Example shape:
    # df = df.copy()
    # df[MODEL_NAME] = <computed scores>
    # return df.dropna(subset=[MODEL_NAME])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="input parquet of variants")
    ap.add_argument("--output", required=True, help="output parquet with scores")
    args = ap.parse_args()

    variants = pd.read_parquet(args.input)
    required = {"variant_id", "chrom", "pos", "ref", "alt"}
    missing = required - set(variants.columns)
    if missing:
        raise SystemExit(f"input missing columns: {sorted(missing)}")

    scored = score_variants(variants)
    scored.to_parquet(args.output, index=False)
    print(f"{MODEL_NAME}: scored {len(scored)}/{len(variants)} variants -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
