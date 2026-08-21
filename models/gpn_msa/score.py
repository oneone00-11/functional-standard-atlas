"""GPN-MSA (hg38 precomputed scores) variant scorer.

Contract (see models/_template/README.md):
    python score.py --input <variant_table.parquet> --output scores.parquet

Score column: `gpn_msa`. GPN-MSA raw scores are MORE NEGATIVE = more
deleterious (constrained), so the raw value is FLIPPED (gpn_msa = -raw) to
satisfy the contract's "larger = more damaging" rule. The flip is also
recorded in NOTES.md. Concordance vs the companion benchmark's raw column is
therefore expected at rho = -1.0.

Source: HuggingFace dataset songlab/gpn-msa-hg38-scores, scores.tsv.bgz
(34.7 GiB). The xet-backed CDN breaks htslib remote seeking ("Illegal seek",
2026-07-28), so instead of pysam-over-HTTP we:

  1. parse the LOCAL tabix index (gpn_scores.tsv.bgz.tbi, pinned alongside
     this scorer — 2.8 MB, copied from the companion repo refs/) to find the
     exact compressed byte ranges covering the input's genomic blocks;
  2. download only those byte ranges (HTTP 206 verified) plus the bgzf
     header and EOF blocks into a SPARSE file in cache/;
  3. open the sparse file with pysam + the local index and fetch normally.

Whole-matrix run downloads ~10 MB total. Filled ranges are tracked in
cache/filled_ranges.json so reruns are incremental.

Use --mock to run offline (deterministic pseudo-scores, for tests).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

MODEL_NAME = "gpn_msa"
MODEL_DIR = Path(__file__).resolve().parent
RESOLVE_URL = (
    "https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores/"
    "resolve/main/scores.tsv.bgz"
)
TBI_PATH = MODEL_DIR / "gpn_scores.tsv.bgz.tbi"
TOTAL_SIZE = 37_274_961_495  # from CDN Content-Range, verified 2026-07-28
BGZF_HEAD = 65_536
BGZF_EOF = 28
PACKAGE_VERSION = "0.24.0"  # pysam, pinned in requirements.txt


# ------------------------------------------------------------ tbi parsing
def read_tbi(path: Path) -> dict[str, dict[int, list[tuple[int, int]]]]:
    """contig -> {bin_id: [(virtual_beg, virtual_end), ...]} from a .tbi."""
    data = gzip.open(path, "rb").read()
    assert data[:4] == b"TBI\x01", "not a tabix index"
    off = 4
    n_ref, _fmt, _cs, _cb, _ce, _meta, _skip, l_nm = struct.unpack_from("<8i", data, off)
    off += 32
    names = data[off : off + l_nm].decode().rstrip("\x00").split("\x00")
    off += l_nm
    refs: dict[str, dict[int, list[tuple[int, int]]]] = {}
    for rid in range(n_ref):
        (n_bin,) = struct.unpack_from("<i", data, off)
        off += 4
        bins: dict[int, list[tuple[int, int]]] = {}
        for _ in range(n_bin):
            bin_id, n_chunk = struct.unpack_from("<Ii", data, off)
            off += 8
            chunks = struct.unpack_from(f"<{2 * n_chunk}Q", data, off)
            off += 16 * n_chunk
            bins[bin_id] = [(chunks[2 * i], chunks[2 * i + 1]) for i in range(n_chunk)]
        (n_intv,) = struct.unpack_from("<i", data, off)
        off += 4 + 8 * n_intv
        refs[names[rid]] = bins
    return refs


def reg2bins(beg: int, end: int) -> list[int]:
    """Candidate bin ids for [beg, end) (0-based), per the SAMtools spec."""
    end -= 1
    bins = [0]
    for shift, base in ((26, 1), (23, 9), (20, 73), (17, 585), (14, 4681)):
        bins += list(range(base + (beg >> shift), base + (end >> shift) + 1))
    return bins


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


def byte_ranges_for(
    refs: dict, blocks: list[tuple[str, int, int]]
) -> list[list[int]]:
    """Merged compressed-byte ranges covering all blocks (+ bgzf head/EOF)."""
    ranges = [[0, BGZF_HEAD], [TOTAL_SIZE - BGZF_EOF, TOTAL_SIZE]]
    for chrom, start0, end in blocks:
        bins = refs.get(chrom) or refs.get(f"chr{chrom}")
        if bins is None:
            raise ValueError(f"contig {chrom} not in tabix index")
        for b in reg2bins(start0, end):
            for vbeg, vend in bins.get(b, []):
                ranges.append([vbeg >> 16, (vend >> 16) + 1])
    ranges.sort()
    merged: list[list[int]] = []
    for s, e in ranges:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


# --------------------------------------------------------- sparse download
def fill_sparse(url: str, sparse_path: Path, ranges: list[list[int]], state_path: Path) -> None:
    filled = json.loads(state_path.read_text()) if state_path.exists() else []
    done = [(s, e) for s, e in filled]
    if not sparse_path.exists():
        with open(sparse_path, "wb") as f:
            f.truncate(TOTAL_SIZE)
    for s, e in ranges:
        if any(s >= ds and e <= de for ds, de in done):
            continue
        for attempt in range(4):
            try:
                r = requests.get(
                    url, headers={"Range": f"bytes={s}-{e - 1}"}, timeout=300
                )
                if r.status_code != 206:
                    raise RuntimeError(f"HTTP {r.status_code} for range {s}-{e}")
                if len(r.content) != e - s:
                    raise RuntimeError(f"short read: {len(r.content)} != {e - s}")
                with open(sparse_path, "r+b") as f:
                    f.seek(s)
                    f.write(r.content)
                done.append((s, e))
                state_path.write_text(json.dumps(done))
                print(f"  filled {s}-{e} ({(e - s) / 1e6:.2f} MB)", flush=True)
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2.0 * (attempt + 1))
                url = resolve_url()


def resolve_url() -> str:
    return requests.head(RESOLVE_URL, allow_redirects=True, timeout=60).url


def score_mock(df: pd.DataFrame) -> pd.Series:
    """Deterministic pseudo-scores in a plausible range, for offline tests."""
    def one(row) -> float:
        d = hashlib.sha256(
            f"mock-gpn:{row['chrom']}:{row['pos']}:{row['ref']}:{row['alt']}".encode()
        ).hexdigest()
        return (int(d[:8], 16) % 1200) / 100.0 - 2.0  # -2.00 .. 9.99 (flipped scale)

    return df.apply(one, axis=1)


# --------------------------------------------------------------------- main

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
    else:
        import pysam  # local import: heavy dependency

        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        sparse_path = cache_dir / "scores.sparse.bgz"
        state_path = cache_dir / "filled_ranges.json"

        blocks = variant_blocks(df)
        refs = read_tbi(TBI_PATH)
        ranges = byte_ranges_for(refs, blocks)
        total_mb = sum(e - s for s, e in ranges) / 1e6
        print(f"gpn_msa: {len(blocks)} block(s), {len(ranges)} byte range(s), "
              f"{total_mb:.1f} MB to ensure", flush=True)
        fill_sparse(resolve_url(), sparse_path, ranges, state_path)

        tb = pysam.TabixFile(str(sparse_path), index=str(TBI_PATH))
        lut: dict = {}
        for chrom, start0, end in blocks:
            for line in tb.fetch(chrom, start0, end):
                p = line.split("\t")
                lut[(p[0], int(p[1]), p[2], p[3])] = -float(p[4])  # FLIP
        tb.close()
        df[MODEL_NAME] = df.apply(
            lambda r: lut.get((str(r["chrom"]), int(r["pos"]), str(r["ref"]), str(r["alt"]))),
            axis=1,
        )

    out = df.dropna(subset=[MODEL_NAME])
    out.to_parquet(args.output, index=False)

    run_log = {
        "model": "GPN-MSA (hg38 precomputed, HF songlab/gpn-msa-hg38-scores)",
        "source_url": RESOLVE_URL,
        "client_package": f"pysam=={PACKAGE_VERSION}",
        "score_definition": "gpn_msa = -raw (raw is more negative = more deleterious); flipped so larger = more damaging",
        "mock": args.mock,
        "scored": len(out),
        "not_in_table": len(df) - len(out),
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    (_run_log_dir(args) / "run_log.json").write_text(json.dumps(run_log, indent=2) + "\n")
    print(f"{MODEL_NAME}: scored {len(out)}/{len(df)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
