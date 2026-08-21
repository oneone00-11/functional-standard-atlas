"""UCSC conservation scorer (phyloP100way / phastCons100way, hg38).

Contract (see models/_template/README.md):
    python score.py --input <variant_table.parquet> --output scores.parquet \
        --track phyloP100way

Score column: named after the track, lowercased (`phylop100way` or
`phastcons100way`) to match the companion benchmark's matrix columns.
Exactly one score column per run; run once per track.

Source: UCSC Genome Browser track API
(https://api.genome.ucsc.edu/getData/track?genome=hg38&track=<track>&...).
One request per genomic block (gene span), so request volume is tiny — no
rate-limit risk. Intervals are 0-based half-open per the API; matrix positions
are 1-based. Orientation: higher = more conserved = more damaging. No flip.

Use --mock to run offline (deterministic pseudo-scores, for tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

TRACKS = {"phyloP100way": "phylop100way", "phastCons100way": "phastcons100way"}
API = "https://api.genome.ucsc.edu/getData/track"
MODEL_DIR = Path(__file__).resolve().parent
PACKAGE_VERSION = "2.32.5"  # requests, pinned in requirements.txt
GAP = 50_000  # split fetch blocks at position gaps larger than this


def variant_blocks(df: pd.DataFrame, gap: int = GAP) -> list[tuple[str, int, int]]:
    """Cluster variant positions into (chrom, start0, end) fetch blocks."""
    blocks = []
    for chrom, sub in df.groupby("chrom"):
        pos = sorted(int(p) for p in sub["pos"].unique())
        start = prev = pos[0]
        for p in pos[1:]:
            if p - prev > gap:
                blocks.append((str(chrom), start - 1, prev + 1))
                start = p
            prev = p
        blocks.append((str(chrom), start - 1, prev + 1))
    return blocks


def fetch_intervals(track: str, chrom: str, start0: int, end: int, retries: int = 3) -> list[dict]:
    """(start, end, value) intervals for one block from the UCSC track API."""
    s = requests.Session()
    s.trust_env = False
    last_exc: Exception | None = None
    for i in range(retries):
        try:
            r = s.get(
                API,
                params={"genome": "hg38", "track": track,
                        "chrom": f"chr{chrom}", "start": start0, "end": end},
                timeout=120,
            )
            if r.status_code == 200:
                data = r.json().get(track)
                if isinstance(data, dict):
                    data = data.get(f"chr{chrom}", [])
                if data is None:
                    raise ValueError(f"track {track} absent from API response for chr{chrom}")
                return data
            last_exc = RuntimeError(f"HTTP {r.status_code}")
            time.sleep(2 * (i + 1))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"UCSC API failed for {track} chr{chrom}:{start0}-{end}") from last_exc


def block_positions(track: str, block: tuple[str, int, int], cache_dir: Path, retries: int) -> dict[int, float]:
    """pos(1-based) -> value map for one block, cached as raw JSON per block."""
    chrom, start0, end = block
    block_path = cache_dir / "regions" / track / f"{chrom}_{start0}_{end}.json"
    block_path.parent.mkdir(parents=True, exist_ok=True)
    if block_path.exists():
        intervals = json.loads(block_path.read_text())
    else:
        intervals = fetch_intervals(track, chrom, start0, end, retries)
        block_path.write_text(json.dumps(intervals))
    pos2val: dict[int, float] = {}
    for it in intervals:
        v, s, e = it.get("value"), it.get("start"), it.get("end")
        if v is None or s is None or e is None:
            continue
        for p in range(int(s) + 1, int(e) + 1):  # 0-based half-open -> 1-based
            pos2val[(str(chrom), p)] = float(v)
    return pos2val


def score_mock(df: pd.DataFrame, col: str) -> pd.Series:
    """Deterministic pseudo-scores in a plausible range, for offline tests."""
    def one(row) -> float:
        d = hashlib.sha256(
            f"mock-{col}:{row['chrom']}:{row['pos']}".encode()
        ).hexdigest()
        return (int(d[:8], 16) % 2000) / 100.0 - 10.0  # -10.00 .. 9.99

    return df.apply(one, axis=1)



def _run_log_dir(args):
    """Where the run log belongs.

    A real run records provenance next to the scorer. A mock run — which the
    test suite performs on three variants — writes beside its own output
    instead, so it cannot overwrite the record of the run that produced the
    deposited scores.
    """
    if getattr(args, "mock", False):
        return Path(args.output).resolve().parent
    return MODEL_DIR

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--track", required=True, choices=sorted(TRACKS))
    ap.add_argument("--limit", type=int, default=None, help="score only first N variants")
    ap.add_argument("--mock", action="store_true", help="offline deterministic scores (tests)")
    ap.add_argument("--cache-dir", default=str(MODEL_DIR / "cache"))
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args(argv)

    col = TRACKS[args.track]
    df = pd.read_parquet(args.input)
    required = {"variant_id", "chrom", "pos", "ref", "alt"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"input missing columns: {sorted(missing)}")
    if args.limit:
        df = df.head(args.limit).copy()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.mock:
        df[col] = score_mock(df, col)
        n_blocks = 0
    else:
        blocks = variant_blocks(df)
        print(f"{args.track}: {len(blocks)} genomic block(s)", flush=True)
        pos2val: dict = {}
        for i, b in enumerate(blocks, 1):
            pos2val.update(block_positions(args.track, b, cache_dir, args.max_retries))
            print(f"  block {i}/{len(blocks)} chr{b[0]}:{b[1]}-{b[2]} done", flush=True)
            time.sleep(0.3)
        df[col] = df.apply(lambda r: pos2val.get((str(r["chrom"]), int(r["pos"]))), axis=1)
        n_blocks = len(blocks)

    out = df.dropna(subset=[col])
    out.to_parquet(args.output, index=False)

    run_log = {
        "model": f"UCSC {args.track} (track API, hg38)",
        "track": args.track,
        "api": API,
        "client_package": f"requests=={PACKAGE_VERSION}",
        "score_definition": "per-base track value; higher = more conserved = more damaging",
        "mock": args.mock,
        "blocks": n_blocks,
        "scored": len(out),
        "unscored_or_failed": len(df) - len(out),
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    (_run_log_dir(args) / f"run_log_{col}.json").write_text(json.dumps(run_log, indent=2) + "\n")
    print(
        f"{col}: scored {len(out)}/{len(df)} "
        f"({len(df) - len(out)} without track value) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
