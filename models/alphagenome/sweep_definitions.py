"""Sweep AlphaGenome scoring definitions: which choice moves the benchmark?

The atlas and the companion benchmark score the same model with two different
definitions and agree at only rho = 0.672. That single comparison confounds two
changes at once — the window shrank from 1 Mb to 16 kb *and* the aggregation
changed from max|raw| to a merged quantile score — so it cannot say which
choice mattered, only that definitions matter somewhere.

This sweep separates them. Each API response already carries both ``raw_score``
and ``quantile_score`` for all three splicing output types, so every aggregation
is obtained from one call; only the window costs extra calls. The grid is
therefore four windows x four aggregations for the price of four windows.

Definitions scored
------------------
``merged_quantile``  max(SPLICE_SITES) + max(SPLICE_SITE_USAGE)
                     + max(SPLICE_JUNCTIONS)/5, on quantile_score  [the atlas]
``merged_raw``       the same combination on raw_score
``max_abs_raw``      max |raw_score| over all rows      [the legacy shape]
``max_quantile``     max quantile_score over all rows

Variants are sampled stratified by gene and region, oversampling splice strata,
because that is where the definition is expected to bite. Results are cached per
(variant, window) so the sweep is resumable and the API is never re-billed for
work already done.

The API key is read from ALPHAGENOME_API_KEY and is never written to disk.

Usage (PYTHONPATH=src):
    ALPHAGENOME_API_KEY=... .venv/bin/python \\
        models/alphagenome/sweep_definitions.py --per-cell 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
REPO = MODEL_DIR.parents[1]
sys.path.insert(0, str(REPO / "src"))

MATRIX = REPO / "results" / "score_matrix_atlas_v2.parquet"
CACHE = MODEL_DIR / "cache" / "sweep"
OUT = REPO / "results" / "alphagenome_definition_sweep_v1.parquet"

WIDTHS = [16384, 131072, 524288, 1048576]
DEFINITIONS = ["merged_quantile", "merged_raw", "max_abs_raw", "max_quantile"]
# splice strata are oversampled: that is where a splice score definition matters
STRATUM_WEIGHT = {"coding_or_utr": 1, "splice_1_2": 3, "splice_3_10": 3,
                  "splice_11_50": 3, "splice_deep": 3}


def derive_all(tidy: pd.DataFrame) -> dict[str, float]:
    """Every definition, from one tidy_scores frame."""
    tcol = "output_type"
    out: dict[str, float] = {}
    for name, col in (("merged_quantile", "quantile_score"), ("merged_raw", "raw_score")):
        def _max_of(t: str) -> float:
            sub = tidy[tidy[tcol].astype(str).str.contains(t, na=False)]
            return float(sub[col].max()) if len(sub) else np.nan
        out[name] = (_max_of("SPLICE_SITES") + _max_of("SPLICE_SITE_USAGE")
                     + _max_of("SPLICE_JUNCTIONS") / 5.0)
    out["max_abs_raw"] = float(tidy["raw_score"].abs().max())
    out["max_quantile"] = float(tidy["quantile_score"].max())
    return out


def sample_variants(per_cell: int, seed: int) -> pd.DataFrame:
    from atlas.evaluate import classify_region
    df = pd.read_parquet(MATRIX, columns=["variant_id", "gene", "chrom", "pos",
                                          "ref", "alt", "hgvs_c",
                                          "functional_pathogenicity"])
    df = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)]
    df = df.dropna(subset=["functional_pathogenicity"])
    df["region"] = df["hgvs_c"].map(classify_region)
    rng = np.random.default_rng(seed)
    picks = []
    for (gene, region), g in df.groupby(["gene", "region"]):
        k = min(len(g), per_cell * STRATUM_WEIGHT.get(region, 1))
        picks.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    return pd.concat(picks, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-cell", type=int, default=40,
                    help="variants per gene x coding stratum (splice strata x3)")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--limit", type=int, default=None, help="pilot: cap total variants")
    args = ap.parse_args()

    key = os.environ.get("ALPHAGENOME_API_KEY")
    if not key:
        raise SystemExit("set ALPHAGENOME_API_KEY (never commit it)")

    from alphagenome.data import genome
    from alphagenome.models import dna_client, variant_scorers

    model = dna_client.create(key)
    scorers = [s for k, s in variant_scorers.RECOMMENDED_VARIANT_SCORERS.items()
               if "splice" in k.lower()]
    if not scorers:
        raise SystemExit("no splice scorers exposed by this client version")

    variants = sample_variants(args.per_cell, args.seed)
    if args.limit:
        variants = variants.head(args.limit)
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"{len(variants):,} variants x {len(WIDTHS)} windows = "
          f"{len(variants) * len(WIDTHS):,} API calls")

    rows, t0, n_api = [], time.time(), 0
    for wi, width in enumerate(WIDTHS, 1):
        half = width // 2
        for i, r in enumerate(variants.itertuples(), 1):
            cf = CACHE / f"{r.variant_id.replace(':', '_').replace('#', '_')}__{width}.json"
            if cf.exists():
                rows.append({**json.loads(cf.read_text()), "variant_id": r.variant_id,
                             "width": width})
                continue
            chrom = f"chr{r.chrom}"
            pos0 = int(r.pos) - 1
            try:
                scored = model.score_variant(
                    interval=genome.Interval(chromosome=chrom,
                                             start=max(0, pos0 - half), end=pos0 + half),
                    variant=genome.Variant(chromosome=chrom, position=int(r.pos),
                                           reference_bases=str(r.ref),
                                           alternate_bases=str(r.alt)),
                    variant_scorers=scorers)
                vals = derive_all(variant_scorers.tidy_scores(scored))
            except Exception as exc:
                vals = {d: np.nan for d in DEFINITIONS}
                vals["error"] = f"{type(exc).__name__}: {exc}"[:200]
            n_api += 1
            cf.write_text(json.dumps(vals))
            rows.append({**vals, "variant_id": r.variant_id, "width": width})
            if i % 200 == 0:
                el = time.time() - t0
                done = (wi - 1) * len(variants) + i
                tot = len(WIDTHS) * len(variants)
                print(f"  width {width:>7} {i:>5}/{len(variants)}  "
                      f"[{done}/{tot}]  {el/60:.1f} min, "
                      f"{(tot-done)*el/max(done,1)/60:.0f} min left", flush=True)

    res = pd.DataFrame(rows)
    res = res.merge(variants[["variant_id", "gene", "region",
                              "functional_pathogenicity"]], on="variant_id", how="left")
    res.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(res):,} rows ({n_api:,} new API calls, "
          f"{time.time()-t0:.0f}s)")
    if "error" in res.columns:
        n_err = int(res["error"].notna().sum())
        print(f"  errors: {n_err:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
