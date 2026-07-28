"""CADD (GRCh38-v1.7) variant scorer.

Contract (see models/_template/README.md):
    python score.py --input <variant_table.parquet> --output scores.parquet

Score column: `cadd` = CADD PHRED (higher = more deleterious).

Two sources (--source):
  - precomputed (default): remote tabix slices of the official whole-genome
    SNV file (81 GB, both mirrors verified to serve Range requests
    2026-07-28). Fast, no rate limits, exact same PHRED values.
  - api: the CADD web API. Bulk scoring triggered load-shedding and an IP ban
    (200 + [] even for BRAF V600E, 2026-07-28 incident); kept for small jobs.
    When the API returns multiple annotations for one allele, max PHRED is
    taken (same rule as the companion benchmark's 75_cadd.py).

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
N_WORKERS = 2  # the API load-sheds under concurrency; keep it gentle
POLITE_DELAY = 0.2  # seconds between requests per worker (<= ~10 req/s total)

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

    Returns None ONLY for a genuine "not scored": an HTTP 200 empty record
    list for an INDEL (CADD precomputes all possible SNVs, so an empty list
    for an SNV is always a rate-limiting artifact — 2026-07-28 incident: the
    API shed load by returning 200 [] even for BRAF V600E, and ~61k bogus
    nulls got cached). Any other response shape, status, or network error is
    retried and then raised — transient failures must never be cached as
    nulls (they would never be refetched).
    """
    is_snv = len(ref) == 1 and len(alt) == 1
    url = API.format(c=chrom, p=pos, r=ref, a=alt)
    last_exc: Exception | None = None
    for i in range(retries):
        try:
            resp = _session().get(url, timeout=40)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    if not data:
                        if is_snv:
                            raise RuntimeError(
                                "empty record list for SNV (API load-shedding, retryable)"
                            )
                        return None  # indel genuinely absent from CADD's precomputed set
                    return max(float(d["PHRED"]) for d in data if d.get("PHRED") is not None)
                raise ValueError(f"unexpected 200 body shape: {str(data)[:200]}")
            if resp.status_code in (429, 500, 502, 503):
                time.sleep(2.0 * (i + 1))
                continue
            raise RuntimeError(f"HTTP {resp.status_code} for {url}")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2.0 * (i + 1))
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


# ------------------------------------------------------- precomputed source
MIRRORS = {
    "US": "https://krishna.gs.washington.edu/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz",
    "DE": "https://kircherlab.bihealth.org/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz",
}
# Verified 2026-07-28: both mirrors serve Range requests; contigs are plain
# ("7", not "chr7"); columns are chrom, pos, ref, alt, RawScore, PHRED.


def variant_blocks(df: pd.DataFrame, gap: int = 50_000) -> list[tuple[str, int, int]]:
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


def fetch_block(url: str, block: tuple[str, int, int], cache_dir: Path, retries: int = 3) -> pd.DataFrame:
    """All CADD rows for one genomic block, cached as a raw tsv per block."""
    import pysam  # local import: heavy dependency, only needed for precomputed

    chrom, start0, end = block
    block_path = cache_dir / "regions" / f"{chrom}_{start0}_{end}.tsv"
    block_path.parent.mkdir(parents=True, exist_ok=True)
    if not block_path.exists():
        tbi_path = cache_dir / "regions" / (url.rsplit("/", 1)[-1] + ".tbi")
        if not tbi_path.exists():
            resp = requests.get(url + ".tbi", timeout=120)
            resp.raise_for_status()
            tbi_path.write_bytes(resp.content)
        last_exc: Exception | None = None
        for i in range(retries):
            try:
                tb = pysam.TabixFile(url, index=str(tbi_path))
                lines = list(tb.fetch(chrom, start0, end))
                tb.close()
                block_path.write_text("\n".join(lines) + ("\n" if lines else ""))
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(2.0 * (i + 1))
        else:
            raise RuntimeError(f"tabix fetch failed for {block}") from last_exc
    rows = []
    for line in block_path.read_text().splitlines():
        f = line.split("\t")
        rows.append((f[0], int(f[1]), f[2], f[3], float(f[5])))  # PHRED col
    return pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt", "cadd"])


def score_precomputed(df: pd.DataFrame, mirror: str, cache_dir: Path, retries: int) -> pd.Series:
    """PHRED per variant from the precomputed whole-genome file via remote tabix."""
    url = MIRRORS[mirror]
    blocks = variant_blocks(df)
    print(f"precomputed source: {len(blocks)} genomic block(s) via {mirror} mirror", flush=True)
    lut = pd.concat(
        [fetch_block(url, b, cache_dir, retries) for b in blocks], ignore_index=True
    ).drop_duplicates(subset=["chrom", "pos", "ref", "alt"])
    merged = df.merge(lut, on=["chrom", "pos", "ref", "alt"], how="left")
    return merged["cadd"]


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None, help="score only first N variants")
    ap.add_argument("--mock", action="store_true", help="offline deterministic scores (tests)")
    ap.add_argument("--cache-dir", default=str(MODEL_DIR / "cache"))
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument(
        "--source",
        choices=["precomputed", "api"],
        default="precomputed",
        help="precomputed = remote tabix on the whole-genome SNV file (recommended; "
        "the API load-sheds and even IP-bans under bulk scoring, 2026-07-28 incident)",
    )
    ap.add_argument("--mirror", choices=sorted(MIRRORS), default="US")
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

    if args.source == "precomputed" and not args.mock:
        df[MODEL_NAME] = score_precomputed(df, args.mirror, cache_dir, args.max_retries)
        out = df.dropna(subset=[MODEL_NAME])
        out.to_parquet(args.output, index=False)
        run_log = {
            "model": "CADD (precomputed whole-genome SNV file, remote tabix)",
            "api_version": API_VERSION,
            "source_url": MIRRORS[args.mirror],
            "client_package": "pysam (see requirements.txt)",
            "score_definition": "PHRED of the exact allele row; higher = more deleterious",
            "input_sha256_16": input_sha,
            "mock": False,
            "scored": len(out),
            "unscored_or_failed": len(df) - len(out),
            "from_cache": "per-block region tsvs in cache/regions/",
            "run_at": datetime.now(timezone.utc).isoformat(),
        }
        (MODEL_DIR / "run_log.json").write_text(json.dumps(run_log, indent=2) + "\n")
        print(
            f"{MODEL_NAME}: scored {len(out)}/{len(df)} "
            f"({len(df) - len(out)} unmatched) -> {args.output}"
        )
        return 0

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
                time.sleep(POLITE_DELAY)
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
