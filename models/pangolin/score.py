"""Pangolin (git tkzeng/Pangolin, CPU torch) variant scorer — local CSV pipeline.

Contract (see models/_template/README.md):
    models/pangolin/.venv/bin/python score.py --input <variants.parquet> \
        --output scores.parquet

Run with THIS model's own venv (models/pangolin/.venv), which carries
pangolin + torch + gffutils + pyfastx — the repo venv does NOT. --mock works
anywhere (no pangolin import).

Score columns (same summary as the companion benchmark, milestone 6):
  `pangolin_score` = max(splice gain, |splice loss|) over all transcript
  records; `pangolin_gain` / `pangolin_loss` of the winning record retained.
  Higher = more splice-altering = more damaging. No flip.

Pipeline (mirrors companion refs/run_pangolin_full.sh):
  1. emit a CSV (CHROM,POS,REF,ALT) of input SNVs — CSV mode because
     Pangolin's VCF writer is incompatible with PyVCF3 (`_Info` signature)
  2. split into chunks, run `pangolin -d 50 chunk.csv REF DB out` per chunk
     in parallel (5 workers, staggered starts — simultaneous torch model
     loads OOM'd the companion run on 7 GB)
  3. parse chunk outputs (PANG_RE over the 5th CSV field), map back by
     (chrom,pos,ref,alt)

Scope: SNVs only (same as SpliceAI; indels deferred, see NOTES.md). Chunk
outputs are cached in cache/chunks/; completed chunks are skipped on rerun
(.done keyed to chunk CSV md5).

References: data/refs/grch38_subset.fa (Ensembl release-112 FASTA, chroms
2/3/13/16/17) and data/refs/grch38_subset.gtf.db (gffutils db from
Homo_sapiens.GRCh38.112.gtf.gz subset to the same chroms) — build steps in
NOTES.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MODEL_NAME = "pangolin_score"
MODEL_DIR = Path(__file__).resolve().parent
REPO = MODEL_DIR.parents[1]
REF_FASTA = REPO / "data" / "refs" / "grch38_subset.fa"
GTF_DB = REPO / "data" / "refs" / "grch38_subset.gtf.db"
N_CHUNKS = 24  # ~1.9k variants each; .done is keyed to the chunk CSV md5 so
# killed/altered chunks are always redone

# Pangolin CSV field: ...|<gain_pos>:<gain>|<loss_pos>:<loss>|Warnings... per
# transcript record (the field itself can contain commas inside Warnings).
PANG_RE = re.compile(r"\|(-?\d+):(-?[0-9.]+)\|(-?\d+):(-?[0-9.]+)\|Warnings")


def emit_csv(df: pd.DataFrame, csv_path: Path) -> int:
    """SNV-only input CSV, Ensembl contig names, sorted by chrom,pos."""
    snv = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)].copy()
    snv["chrom_s"] = snv["chrom"].astype(str)
    snv = snv.sort_values(["chrom_s", "pos"])
    with open(csv_path, "w") as fh:
        fh.write("CHROM,POS,REF,ALT\n")
        for r in snv.itertuples():
            fh.write(f"{r.chrom_s},{int(r.pos)},{r.ref},{r.alt}\n")
    return len(snv)


def parse_pangolin_csv(path: Path) -> dict:
    """(chrom,pos,ref,alt) -> (score, gain, loss), from one Pangolin CSV.

    Manual parse: the Pangolin field can contain commas (Warnings text), so
    split on only the first 4 commas (companion 82_assemble_v2.py).
    """
    scores: dict = {}
    with open(path, encoding="utf-8") as fh:
        fh.readline()  # header
        for line in fh:
            parts = line.rstrip("\n").split(",", 4)
            if len(parts) < 5:
                continue
            chrom, pos, ref, alt, val = parts
            best = gain_b = loss_b = None
            for _g_pos, gain, _l_pos, loss in PANG_RE.findall(val):
                gn, ls = float(gain), float(loss)
                agg = max(gn, -ls)
                if best is None or agg > best:
                    best, gain_b, loss_b = agg, gn, ls
            if best is not None:
                scores[(chrom, int(pos), ref, alt)] = (best, gain_b, loss_b)
    return scores


def run_chunks(csv_path: Path, chunks_dir: Path, n_chunks: int, distance: int) -> dict:
    """Split CSV into chunks, run pangolin per chunk (parallel), parse all."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    lines = csv_path.read_text().splitlines()
    header, body = lines[0], lines[1:]
    chunk_size = math.ceil(len(body) / n_chunks)
    chunk_csvs = []
    for i in range(n_chunks):
        part = body[i * chunk_size : (i + 1) * chunk_size]
        if not part:
            continue
        cc = chunks_dir / f"chunk_{i:02d}.csv"
        cc.write_text("\n".join([header] + part) + "\n")
        chunk_csvs.append((i, cc))

    exe = Path(sys.executable).parent / "pangolin"
    import os
    env = dict(os.environ)
    # cap BLAS threads: 5 torch processes on 10 cores
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})

    def one(item) -> Path:
        i, cc = item
        out = cc.with_suffix(".out.csv")  # pangolin appends ".csv" to the prefix
        done_marker = cc.with_suffix(".done")
        want_md5 = hashlib.md5(cc.read_bytes()).hexdigest()
        if done_marker.exists() and out.exists() and done_marker.read_text() == want_md5:
            return out
        done_marker.unlink(missing_ok=True)  # stale marker from a different input
        time.sleep((i % 5) * 15)  # stagger torch model loads (companion OOM lesson)
        cmd = [str(exe), "-d", str(distance), str(cc), str(REF_FASTA),
               str(GTF_DB), str(cc.with_suffix(".out"))]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"pangolin failed on {cc.name}: {proc.stderr[-400:]}")
        done_marker.write_text(want_md5)
        return out

    with ThreadPoolExecutor(max_workers=min(len(chunk_csvs), 5)) as ex:
        outs = list(ex.map(one, chunk_csvs))
    scores: dict = {}
    for out in outs:
        scores.update(parse_pangolin_csv(out))
    return scores


def score_mock(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic pseudo scores in [0, 1), for offline tests."""
    def one(row):
        d = hashlib.sha256(
            f"mock-pangolin:{row['chrom']}:{row['pos']}:{row['ref']}:{row['alt']}".encode()
        ).hexdigest()
        s = (int(d[:8], 16) % 1000) / 1000.0
        return pd.Series({MODEL_NAME: s, "pangolin_gain": s, "pangolin_loss": -s / 2})

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
    ap.add_argument("--limit", type=int, default=None, help="score only first N variants")
    ap.add_argument("--mock", action="store_true", help="offline deterministic scores (tests)")
    ap.add_argument("--cache-dir", default=str(MODEL_DIR / "cache"))
    ap.add_argument("--chunks", type=int, default=N_CHUNKS)
    ap.add_argument("--distance", type=int, default=50, help="Pangolin -d flanking distance")
    args = ap.parse_args(argv)

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
        df = pd.concat([df, score_mock(df)], axis=1)
        n_snv = int(((df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)).sum())
    else:
        for ref in (REF_FASTA, GTF_DB):
            if not ref.exists():
                raise SystemExit(f"reference missing: {ref} (see NOTES.md build steps)")
        csv_path = cache_dir / "variants.csv"
        n_snv = emit_csv(df, csv_path)
        print(f"pangolin: {n_snv} SNVs -> CSV; running {args.chunks} chunks", flush=True)
        scores = run_chunks(csv_path, cache_dir / "chunks", args.chunks, args.distance)
        hit = df.apply(
            lambda r: scores.get((str(r["chrom"]), int(r["pos"]), str(r["ref"]), str(r["alt"]))),
            axis=1,
        )
        df[MODEL_NAME] = hit.apply(lambda t: t[0] if t else None)
        df["pangolin_gain"] = hit.apply(lambda t: t[1] if t else None)
        df["pangolin_loss"] = hit.apply(lambda t: t[2] if t else None)

    out = df.dropna(subset=[MODEL_NAME])
    out.to_parquet(args.output, index=False)

    run_log = {
        "model": "Pangolin (git tkzeng/Pangolin, local, CPU torch)",
        "reference": "data/refs/grch38_subset.fa (Ensembl release-112, chr 2/3/13/16/17)",
        "annotation": "data/refs/grch38_subset.gtf.db (gffutils, Ensembl GRCh38.112 subset)",
        "distance": args.distance,
        "mask": "default (True)",
        "score_definition": "max(splice gain, |splice loss|) over transcript records; higher = more splice-altering",
        "scope": "SNVs only (indels deferred)",
        "mock": args.mock,
        "snv_in_input": n_snv,
        "scored": len(out),
        "unscored": len(df) - len(out),
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    (_run_log_dir(args) / "run_log.json").write_text(json.dumps(run_log, indent=2) + "\n")
    print(f"{MODEL_NAME}: scored {len(out)}/{len(df)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
