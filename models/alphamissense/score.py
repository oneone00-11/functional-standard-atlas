"""AlphaMissense (hg38) pathogenicity scorer — missense SNVs only.

Contract (see models/_template/README.md):
    python score.py --input <variant_table.parquet> --output scores.parquet

Score column: `alphamissense` = am_pathogenicity (higher = more likely
pathogenic; no flip). Non-missense variants are NOT scored (dropped, never
imputed) — AlphaMissense covers missense only by design.

Source file: AlphaMissense_hg38.tsv.gz (Zenodo record 8208688), ~643 MB.
Columns: #CHROM POS REF ALT genome uniprot transcript protein_variant
am_pathogenicity am_class. Chromosomes are "chrN"-style.

The scorer looks for the gz in --cache-dir (default models/alphamissense/cache/).
Download once, e.g.:
    curl -L -o models/alphamissense/cache/AlphaMissense_hg38.tsv.gz \
        https://zenodo.org/records/8208688/files/AlphaMissense_hg38.tsv.gz
The stream-filter then takes ~2-4 min for the whole atlas matrix.

Use --mock to run offline (deterministic pseudo-scores, for tests).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MODEL_NAME = "alphamissense"
MODEL_DIR = Path(__file__).resolve().parent
GZ_NAME = "AlphaMissense_hg38.tsv.gz"
ZENODO_URL = f"https://zenodo.org/records/8208688/files/{GZ_NAME}"


def score_mock(df: pd.DataFrame) -> pd.Series:
    """Deterministic pseudo-scores in [0, 1), for offline testing."""
    def one(row) -> float:
        d = hashlib.sha256(
            f"mock-am:{row['chrom']}:{row['pos']}:{row['ref']}:{row['alt']}".encode()
        ).hexdigest()
        return (int(d[:8], 16) % 1000) / 1000.0

    return df.apply(one, axis=1)


def filter_alphamissense(df: pd.DataFrame, gz_path: Path) -> pd.Series:
    """am_pathogenicity for variants present in the AlphaMissense table."""
    want = {
        (str(r.chrom), int(r.pos), str(r.ref), str(r.alt)): None
        for r in df.itertuples()
    }
    bounds = {
        str(c): (int(g.pos.min()), int(g.pos.max())) for c, g in df.groupby("chrom")
    }
    found: dict = {}
    with gzip.open(gz_path, "rt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("CHROM"):
                continue
            p = line.rstrip("\n").split("\t")
            chrom = p[0][3:]  # strip "chr"
            lo_hi = bounds.get(chrom)
            if lo_hi is None:
                continue
            pos = int(p[1])
            if pos < lo_hi[0] or pos > lo_hi[1]:
                continue
            key = (chrom, pos, p[2], p[3])
            if key in want:
                found[key] = float(p[8])
    print(f"alphamissense: matched {len(found)}/{len(want)} variant keys", flush=True)
    return df.apply(
        lambda r: found.get((str(r["chrom"]), int(r["pos"]), str(r["ref"]), str(r["alt"]))),
        axis=1,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None, help="score only first N variants")
    ap.add_argument("--mock", action="store_true", help="offline deterministic scores (tests)")
    ap.add_argument("--cache-dir", default=str(MODEL_DIR / "cache"))
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.input)
    required = {"variant_id", "chrom", "pos", "ref", "alt"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"input missing columns: {sorted(missing)}")
    if args.limit:
        df = df.head(args.limit).copy()

    if args.mock:
        df[MODEL_NAME] = score_mock(df)
        n_matched = len(df)
    else:
        gz_path = Path(args.cache_dir) / GZ_NAME
        if not gz_path.exists():
            raise SystemExit(
                f"{gz_path} not found. Download once with:\n"
                f"  curl -L -o {gz_path} {ZENODO_URL}"
            )
        df[MODEL_NAME] = filter_alphamissense(df, gz_path)
        n_matched = int(df[MODEL_NAME].notna().sum())

    out = df.dropna(subset=[MODEL_NAME])
    out.to_parquet(args.output, index=False)

    run_log = {
        "model": "AlphaMissense (hg38 precomputed table)",
        "source_url": ZENODO_URL,
        "score_definition": "am_pathogenicity; higher = more likely pathogenic; missense only",
        "mock": args.mock,
        "scored": len(out),
        "not_in_table": len(df) - len(out),
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    (MODEL_DIR / "run_log.json").write_text(json.dumps(run_log, indent=2) + "\n")
    print(
        f"{MODEL_NAME}: scored {len(out)}/{len(df)} "
        f"({len(df) - len(out)} not in table / non-missense) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
