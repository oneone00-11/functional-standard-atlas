"""AlphaGenome merged-splicing variant scorer.

Contract (see models/_template/README.md):
    python score.py --input <variant_table.parquet> --output scores.parquet

Implements the "merged splicing" score recommended in the AlphaGenome paper's
variant-scoring tutorial (the same definition used in the companion benchmark):

    alphagenome = max(SPLICE_SITES) + max(SPLICE_SITE_USAGE) + max(SPLICE_JUNCTIONS) / 5.0

Larger = more damaging (splice-disrupting). Variants the API cannot score are
dropped and logged.

Requires: env var ALPHAGENOME_API_KEY (free for non-commercial use;
register at https://deepmind.google.com/science/alphagenome).
Use --mock to run without an API key (deterministic pseudo-scores, for tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MODEL_NAME = "alphagenome"
MODEL_DIR = Path(__file__).resolve().parent
PACKAGE_VERSION = "0.7.0"  # alphagenome client, pinned in requirements.txt
DEFAULT_WIDTH = 2**14  # 16 kb variant-centred interval; 2^15 is NOT supported
# by the API (supported: 16384, 131072, 524288, 1048576 — verified 2026-07-26)

OUTPUT_TYPE_CANDIDATES = ("output_type", "requested_output", "output")
# Client 0.7.0 tidy_scores exposes raw_score/quantile_score. The merged score
# sums maxima across output types, which is only meaningful on the
# scale-normalised quantile_score; raw_score is a last-resort fallback.
SCORE_COL_CANDIDATES = ("score", "value", "effect", "quantile_score", "raw_score")


# ---------------------------------------------------------------- score math
def extract_merged_splicing_score(tidy: pd.DataFrame) -> float:
    """Merged splicing score from a tidy_scores DataFrame for ONE variant.

    max over each splicing output type's score column, combined as
    SPLICE_SITES + SPLICE_SITE_USAGE + SPLICE_JUNCTIONS / 5.
    Raises with the available columns listed if the frame shape is unexpected
    (client versions may rename columns; fix here, not silently).
    """
    type_col = next((c for c in OUTPUT_TYPE_CANDIDATES if c in tidy.columns), None)
    score_col = next((c for c in SCORE_COL_CANDIDATES if c in tidy.columns), None)
    if type_col is None or score_col is None:
        raise ValueError(
            f"unexpected tidy_scores columns: {list(tidy.columns)}; "
            f"expected one of {OUTPUT_TYPE_CANDIDATES} and one of {SCORE_COL_CANDIDATES}"
        )

    def _max_of(output_type: str) -> float:
        sub = tidy[tidy[type_col].astype(str).str.contains(output_type, na=False)]
        if sub.empty:
            raise ValueError(f"no rows for {output_type} in tidy_scores output")
        return float(sub[score_col].max())

    return (
        _max_of("SPLICE_SITES")
        + _max_of("SPLICE_SITE_USAGE")
        + _max_of("SPLICE_JUNCTIONS") / 5.0
    )


# -------------------------------------------------------------------- cache
def cache_key(row: pd.Series, width: int) -> str:
    raw = f"{row['chrom']}:{row['pos']}:{row['ref']}:{row['alt']}:w{width}:v{PACKAGE_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def load_cache(cache_path: Path) -> dict:
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["key"]] = rec["score"]
    return cache


def append_cache(cache_path: Path, key: str, score: float) -> None:
    with open(cache_path, "a") as fh:
        fh.write(json.dumps({"key": key, "score": score}) + "\n")


# ------------------------------------------------------------------- client
def build_client():
    from alphagenome.models import dna_client  # local import: heavy dependency

    api_key = os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        sys.exit(
            "ALPHAGENOME_API_KEY not set. Register free (non-commercial) at "
            "https://deepmind.google.com/science/alphagenome, or use --mock."
        )
    return dna_client.create(api_key)


def score_one_api(model, scorers, row: pd.Series, width: int) -> float:
    from alphagenome.data import genome
    from alphagenome.models import variant_scorers

    chrom = f"chr{row['chrom']}"
    pos0 = int(row["pos"]) - 1  # matrix is 1-based; intervals are 0-based half-open
    half = width // 2
    interval = genome.Interval(chromosome=chrom, start=max(0, pos0 - half), end=pos0 + half)
    variant = genome.Variant(
        chromosome=chrom,
        position=int(row["pos"]),
        reference_bases=str(row["ref"]),
        alternate_bases=str(row["alt"]),
    )
    scored = model.score_variant(
        interval=interval, variant=variant, variant_scorers=scorers
    )
    tidy = variant_scorers.tidy_scores(scored)
    return extract_merged_splicing_score(tidy)


def score_one_mock(row: pd.Series, width: int) -> float:
    """Deterministic pseudo-score in a plausible range, for offline testing."""
    digest = hashlib.sha256(
        f"mock:{row['chrom']}:{row['pos']}:{row['ref']}:{row['alt']}".encode()
    ).hexdigest()
    return (int(digest[:8], 16) % 2000) / 1000.0  # 0.000 - 1.999


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None, help="score only first N variants")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="interval width (bp)")
    ap.add_argument("--mock", action="store_true", help="offline deterministic scores (tests)")
    ap.add_argument("--cache-dir", default=str(MODEL_DIR / "cache"))
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.input)
    required = {"variant_id", "chrom", "pos", "ref", "alt"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"input missing columns: {sorted(missing)}")
    if args.limit:
        df = df.head(args.limit).copy()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "scores.jsonl"
    failed_path = cache_dir / "failed.jsonl"
    cache = load_cache(cache_path)

    model = None
    scorers = None
    if not args.mock:
        from alphagenome.models import variant_scorers

        model = build_client()
        scorers = []
        for key in ("SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS"):
            value = variant_scorers.RECOMMENDED_VARIANT_SCORERS[key]
            # 0.7.0 returns a single scorer per key; older versions a list.
            scorers.extend(value if isinstance(value, (list, tuple)) else [value])

    scores: dict[int, float] = {}
    n_from_cache = 0
    for idx, row in df.iterrows():
        key = cache_key(row, args.width)
        if key in cache:
            scores[idx] = cache[key]
            n_from_cache += 1
            continue
        for attempt in range(args.max_retries):
            try:
                score = (
                    score_one_mock(row, args.width)
                    if args.mock
                    else score_one_api(model, scorers, row, args.width)
                )
                scores[idx] = score
                append_cache(cache_path, key, score)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == args.max_retries - 1:
                    with open(failed_path, "a") as fh:
                        fh.write(json.dumps({"key": key, "error": str(exc)[:300]}) + "\n")
                else:
                    time.sleep(2**attempt)

    df[MODEL_NAME] = pd.Series(scores)
    out = df.dropna(subset=[MODEL_NAME])
    out.to_parquet(args.output, index=False)

    run_log = {
        "model": "AlphaGenome (DeepMind API)",
        "client_package": f"alphagenome=={PACKAGE_VERSION}",
        "score_definition": "max(SPLICE_SITES) + max(SPLICE_SITE_USAGE) + max(SPLICE_JUNCTIONS)/5 on quantile_score",
        "width": args.width,
        "mock": args.mock,
        "scored": len(out),
        "failed_or_cached_out": len(df) - len(out),
        "from_cache": n_from_cache,
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    (MODEL_DIR / "run_log.json").write_text(json.dumps(run_log, indent=2) + "\n")
    print(
        f"{MODEL_NAME}: scored {len(out)}/{len(df)} "
        f"({n_from_cache} from cache, {len(df) - len(out)} failed) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
