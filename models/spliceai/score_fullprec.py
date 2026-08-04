"""Re-score SpliceAI at full float precision.

The SpliceAI command-line tool computes its four delta scores as float32 and then
formats them with ``"{:.2f}"``. The rounding is therefore purely a property of the
text output, not of the model or of the score definition: 47% of coding variants
and 67% of deep-intronic variants land on exactly 0.00, which bounds Spearman ρ
below 1 for arithmetic reasons alone (see ``atlas.robustness``).

This script recovers the unrounded values by taking ``get_delta_scores`` verbatim
from the installed package and rewriting the four ``{:.2f}`` fields to ``{:.17g}``. Nothing
else about the computation changes, and ``--validate`` checks exactly that: the
full-precision output, re-rounded to two decimals, must reproduce the stock
function bit for bit.

Run with the SpliceAI environment:
    models/spliceai/.venv/bin/python models/spliceai/score_fullprec.py --validate
    models/spliceai/.venv/bin/python models/spliceai/score_fullprec.py --run
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
REPO = MODEL_DIR.parents[1]
MATRIX = REPO / "results" / "score_matrix_atlas_v1.parquet"
REF_FASTA = REPO / "data" / "refs" / "grch38_subset.fa"
OUT = REPO / "results" / "spliceai_scores_fullprec.parquet"
ANNOTATION = "grch38"
DISTANCE = 50
MASK = 0


def build_fullprec():
    """`get_delta_scores` with only the output precision changed."""
    from spliceai import utils as U

    src = inspect.getsource(U.get_delta_scores)
    patched, n = re.subn(r"\{:\.2f\}", "{:.17g}", src)
    if n != 4:
        raise RuntimeError(f"expected 4 '{{:.2f}}' fields to patch, found {n}")
    patched = patched.replace("def get_delta_scores(", "def get_delta_scores_fullprec(", 1)
    ns = dict(vars(U))
    exec(compile(patched, "<spliceai-fullprec>", "exec"), ns)
    return ns["get_delta_scores_fullprec"]


class Rec:
    """Minimal stand-in for the pysam VCF record get_delta_scores expects."""

    def __init__(self, chrom, pos, ref, alt):
        self.chrom, self.pos, self.ref, self.alts = chrom, pos, ref, [alt]


def aggregate(fields: list[str]) -> float:
    """max over the four delta scores — the atlas definition, unchanged."""
    best = np.nan
    for f in fields:
        parts = f.split("|")
        if len(parts) < 6 or parts[2] == ".":
            continue
        vals = [float(x) for x in parts[2:6]]
        m = max(vals)
        best = m if np.isnan(best) else max(best, m)
    return best


def load_variants() -> pd.DataFrame:
    df = pd.read_parquet(MATRIX, columns=["variant_id", "chrom", "pos", "ref", "alt",
                                          "gene", "spliceai_ds"])
    snv = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)].copy()
    snv["chrom_s"] = snv["chrom"].astype(str)
    return snv.sort_values(["chrom_s", "pos"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true",
                    help="check the patched function against the stock one and exit")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-every", type=int, default=2000)
    args = ap.parse_args()

    from spliceai.utils import Annotator, get_delta_scores

    ann = Annotator(str(REF_FASTA), ANNOTATION)
    fullprec = build_fullprec()
    variants = load_variants()

    if args.validate:
        sample = variants.sample(n=min(300, len(variants)), random_state=20260803)
        checked = mismatched = 0
        for r in sample.itertuples():
            rec = Rec(r.chrom_s, int(r.pos), r.ref, r.alt)
            stock = get_delta_scores(rec, ann, DISTANCE, MASK)
            fine = fullprec(rec, ann, DISTANCE, MASK)
            if len(stock) != len(fine):
                mismatched += 1
                continue
            for a, b in zip(stock, fine):
                pa, pb = a.split("|"), b.split("|")
                if pa[2] == "." or pb[2] == ".":
                    continue
                checked += 1
                rounded = [f"{float(x):.2f}" for x in pb[2:6]]
                if rounded != pa[2:6]:
                    mismatched += 1
                    print("MISMATCH", pa[2:6], rounded)
        print(f"validated {checked} score fields on {len(sample)} variants; "
              f"{mismatched} mismatches")
        return 1 if mismatched else 0

    if not args.run:
        ap.error("pass --validate or --run")

    if args.limit:
        variants = variants.head(args.limit)
    scores, t0 = [], time.time()
    for i, r in enumerate(variants.itertuples(), 1):
        rec = Rec(r.chrom_s, int(r.pos), r.ref, r.alt)
        try:
            scores.append(aggregate(fullprec(rec, ann, DISTANCE, MASK)))
        except Exception:
            scores.append(np.nan)
        if i % args.report_every == 0:
            el = time.time() - t0
            print(f"{i:>6}/{len(variants)}  {el/60:5.1f} min elapsed, "
                  f"{(len(variants)-i)*el/i/60:5.1f} min remaining", flush=True)
    out = variants[["variant_id", "gene", "chrom", "pos", "ref", "alt"]].copy()
    out["spliceai_ds_fullprec"] = scores
    out["spliceai_ds_rounded"] = variants["spliceai_ds"].to_numpy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    ok = out.dropna(subset=["spliceai_ds_fullprec", "spliceai_ds_rounded"])
    agree = (ok["spliceai_ds_fullprec"].round(2) - ok["spliceai_ds_rounded"]).abs().max()
    print(f"\nwrote {OUT} ({len(out):,} variants, "
          f"{out['spliceai_ds_fullprec'].notna().sum():,} scored)")
    print(f"max |round(fullprec,2) - stored rounded| = {agree:.4f}")
    print(f"distinct values: {out['spliceai_ds_fullprec'].nunique():,} "
          f"(was {out['spliceai_ds_rounded'].nunique():,})")
    print(f"total {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
