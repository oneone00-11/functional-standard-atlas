"""CADD (GRCh38-v1.7) variant scorer via the CADD web API.

Contract (see models/_template/README.md):
    python score.py --input <variant_table.parquet> --output scores.parquet

Score column: `cadd` = CADD PHRED (higher = more deleterious). When the API
returns multiple annotations for one allele, the max PHRED is taken (same rule
as the companion benchmark's 75_cadd.py).

Requires network access to cadd.gs.washington.edu.
Use --mock to run offline (deterministic pseudo-scores, for tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

MODEL_NAME = "cadd"
MODEL_DIR = Path(__file__).resolve().parent
API_VERSION = "GRCh38-v1.7"
API = f"https://cadd.gs.washington.edu/api/v1.0/{API_VERSION}/{{c}}:{{p}}_{{r}}_{{a}}"
PACKAGE_VERSION = "2.32.5"  # requests, pinned in requirements.txt
N_WORKERS = 5

_local = threading.local()
_lock = threading.Lock()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.trust_env = False  # ignore ambient proxy env vars (old repo: flaky proxy)
        _local.s = s
    return _local.s


def fetch_phred(chrom: str, pos: int, ref: str, alt: str, retries: int = 3) -> float | None:
    """Max PHRED over the API's annotation records.

    Returns None ONLY for a genuine "not scored" (HTTP 200 with an empty
    record list). Any other response shape, status, or network error is
    retried and then raised — transient failures must never be cached as
    nulls (they would never be refetched).
    """
    url = API.format(c=chrom, p=pos, r=ref, a=alt)
    last_exc: Exception | None = None
    for i in range(retries):
        try:
            resp = _session().get(url, timeout=40)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    if not data:
                        return None  # genuinely unscored by CADD
                    return max(float(d["PHRED"]) for d in data if d.get("PHRED") is not None)
                raise ValueError(f"unexpected 200 body shape: {str(data)[:200]}")
            if resp.status_code in (429, 500, 502, 503):
                time.sleep(1.5 * (i + 1))
                continue
            raise RuntimeError(f"HTTP {resp.status_code} for {url}")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"CADD API failed after {retries} retries: {url}") from last_exc


def score_one_mock(chrom: str, pos: int, ref: str, alt: str) -> float:
    """Deterministic pseudo-PHRED in a plausible range, for offline testing."""
    digest = hashlib.sha256(f"mock-cadd:{chrom}:{pos}:{ref}:{alt}".encode()).hexdigest()
    return (int(digest[:8], 16) % 4000) / 100.0  # 0.00 - 39.99


# -------------------------------------------------------------------- cache
def cache_key(input_sha: str, chrom: str, pos: int, ref: str, alt: str) -> str:
    raw = f"{input_sha}:{chrom}:{pos}:{ref}:{alt}:{API_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def load_cache(cache_path: Path) -> dict:
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["key"]] = rec["score"]  # float, or null = unscored
    return cache


def append_cache(cache_path: Path, key: str, score: float | None) -> None:
    with _lock, open(cache_path, "a") as fh:
        fh.write(json.dumps({"key": key, "score": score}) + "\n")


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None, help="score only first N variants")
    ap.add_argument("--mock", action="store_true", help="offline deterministic scores (tests)")
    ap.add_argument("--cache-dir", default=str(MODEL_DIR / "cache"))
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.input)
    required = {"variant_id", "chrom", "pos", "ref", "alt"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"input missing columns: {sorted(missing)}")
    if args.limit:
        df = df.head(args.limit).copy()

    input_sha = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()[:16]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "scores.jsonl"
    failed_path = cache_dir / "failed.jsonl"
    cache = load_cache(cache_path)

    scores: dict[int, float | None] = {}
    n_from_cache = 0
    todo = []
    for idx, row in df.iterrows():
        key = cache_key(input_sha, str(row["chrom"]), int(row["pos"]), str(row["ref"]), str(row["alt"]))
        if key in cache:
            scores[idx] = cache[key]
            n_from_cache += 1
        else:
            todo.append((idx, key, str(row["chrom"]), int(row["pos"]), str(row["ref"]), str(row["alt"])))

    def task(item):
        idx, key, chrom, pos, ref, alt = item
        try:
            if args.mock:
                score = score_one_mock(chrom, pos, ref, alt)
            else:
                score = fetch_phred(chrom, pos, ref, alt, retries=args.max_retries)
            append_cache(cache_path, key, score)
            return idx, score
        except Exception as exc:  # noqa: BLE001
            with _lock, open(failed_path, "a") as fh:
                fh.write(json.dumps({"key": key, "error": str(exc)[:300]}) + "\n")
            return idx, None

    workers = 1 if args.mock else N_WORKERS
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(task, t) for t in todo]
        for done, f in enumerate(as_completed(futs), 1):
            idx, score = f.result()
            scores[idx] = score
            if done % 500 == 0:
                print(f"  {done}/{len(todo)} fetched", flush=True)

    df[MODEL_NAME] = pd.Series(scores)
    out = df.dropna(subset=[MODEL_NAME])
    out.to_parquet(args.output, index=False)

    run_log = {
        "model": "CADD (web API)",
        "api_version": API_VERSION,
        "client_package": f"requests=={PACKAGE_VERSION}",
        "score_definition": "max PHRED over API annotation records; higher = more deleterious",
        "input_sha256_16": input_sha,
        "mock": args.mock,
        "scored": len(out),
        "unscored_or_failed": len(df) - len(out),
        "from_cache": n_from_cache,
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    (MODEL_DIR / "run_log.json").write_text(json.dumps(run_log, indent=2) + "\n")
    print(
        f"{MODEL_NAME}: scored {len(out)}/{len(df)} "
        f"({n_from_cache} from cache, {len(df) - len(out)} unscored/failed) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
