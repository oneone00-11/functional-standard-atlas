"""Fetch precomputed meta-predictor scores from MyVariant.info (dbNSFP, hg38).

The atlas represents the integrative-score class with CADD and AlphaMissense
only, which is a fair objection to a twelve-model benchmark: the meta-predictors
most used in clinical practice are absent. dbNSFP carries them, but the full
release is a ~30 GB download for scores on 46,392 variants in seven genes.
MyVariant.info serves the same dbNSFP fields over a batch REST API, so the
columns can be fetched in a few dozen requests and re-fetched by anyone.

Orientation follows the atlas rule (larger = more damaging). Where dbNSFP
returns one value per affected transcript, they are aggregated by taking the
most damaging, matching how CADD is aggregated over consequence records.

Raw responses are cached under ``models/metapredictors/cache/`` so the fetch is
resumable and the exact payload is auditable.

Usage (PYTHONPATH=src):  python -m atlas.metapredictors --out results/
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
CACHE = REPO / "models" / "metapredictors" / "cache"
ENDPOINT = "https://myvariant.info/v1/variant"
UA = "functional-standard-atlas/1.0 (mailto:cliffzhang@u.nus.edu)"

# dbNSFP path -> (atlas column, flip?) ; flip=True where the source scores
# higher = more tolerated and the atlas needs higher = more damaging.
FIELDS: dict[str, tuple[str, bool]] = {
    "dbnsfp.revel.score": ("revel", False),
    "dbnsfp.bayesdel.add_af.score": ("bayesdel_addaf", False),
    "dbnsfp.clinpred.score": ("clinpred", False),
    "dbnsfp.metarnn.score": ("metarnn", False),
    "dbnsfp.primateai.score": ("primateai", False),
    "dbnsfp.vest4.score": ("vest4", False),
    # ESM-1b reports a log-likelihood ratio: more negative = more damaging.
    "dbnsfp.esm1b.score": ("esm1b", True),
}
BATCH = 500


def _dig(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _most_damaging(value, flip: bool) -> float:
    """Collapse a scalar or per-transcript list to one atlas-oriented score."""
    if value is None:
        return np.nan
    vals = value if isinstance(value, list) else [value]
    out = []
    for v in vals:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    if not out:
        return np.nan
    return -min(out) if flip else max(out)


def hgvs_ids(df: pd.DataFrame) -> list[str]:
    return [f"chr{c}:g.{int(p)}{r}>{a}"
            for c, p, r, a in zip(df["chrom"], df["pos"], df["ref"], df["alt"])]


def post(ids: list[str], retries: int = 4) -> list[dict]:
    body = json.dumps({"ids": ids, "fields": "dbnsfp", "assembly": "hg38"}).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                ENDPOINT, data=body,
                headers={"Content-Type": "application/json", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as fh:
                return json.loads(fh.read().decode())
        except Exception as exc:  # transient 5xx / rate limit / timeout
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"MyVariant.info failed after {retries} attempts: {last}")


def fetch(df: pd.DataFrame, sleep: float = 0.2) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    ids = hgvs_ids(df)
    rows: list[dict] = []
    n_batches = (len(ids) + BATCH - 1) // BATCH

    for b in range(n_batches):
        chunk_ids = ids[b * BATCH:(b + 1) * BATCH]
        chunk_df = df.iloc[b * BATCH:(b + 1) * BATCH]
        cache_file = CACHE / f"batch_{b:04d}.json.gz"
        if cache_file.exists():
            with gzip.open(cache_file, "rt") as fh:
                payload = json.load(fh)
        else:
            payload = post(chunk_ids)
            with gzip.open(cache_file, "wt") as fh:
                json.dump(payload, fh)
            time.sleep(sleep)

        by_query: dict[str, dict] = {}
        for rec in payload:
            q = rec.get("query")
            if q and not rec.get("notfound"):
                by_query.setdefault(q, rec)

        for vid, q in zip(chunk_df["variant_id"], chunk_ids):
            rec = by_query.get(q)
            row = {"variant_id": vid}
            for path, (col, flip) in FIELDS.items():
                row[col] = _most_damaging(_dig(rec, path), flip) if rec else np.nan
            rows.append(row)

        if (b + 1) % 10 == 0 or b + 1 == n_batches:
            print(f"  batch {b + 1}/{n_batches} ({len(rows):,} variants)", flush=True)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=str(RESULTS / "score_matrix_atlas_v1.parquet"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args(argv)

    mat = pd.read_parquet(args.matrix, columns=["variant_id", "gene", "chrom",
                                                "pos", "ref", "alt"])
    snv = mat[(mat["ref"].str.len() == 1) & (mat["alt"].str.len() == 1)].reset_index(drop=True)
    print(f"fetching dbNSFP meta-predictors for {len(snv):,} SNVs "
          f"in {(len(snv) + BATCH - 1) // BATCH} batches")

    scores = fetch(snv, sleep=args.sleep)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out / "metapredictor_scores_v1.parquet", index=False)

    cols = [c for c, _ in FIELDS.values()]
    cov = {c: int(scores[c].notna().sum()) for c in cols}
    summary = {"n_snv": int(len(snv)), "coverage": cov,
               "orientation": {c: ("flipped" if f else "as published")
                               for c, f in FIELDS.values()},
               "endpoint": ENDPOINT, "assembly": "hg38"}
    (out / "metapredictor_scores_v1.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nwrote {out / 'metapredictor_scores_v1.parquet'}")
    for c in cols:
        print(f"  {c:<16} {cov[c]:>7,} scored ({cov[c] / len(snv):5.1%} of SNVs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
