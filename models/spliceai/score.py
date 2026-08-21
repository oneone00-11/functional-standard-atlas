"""SpliceAI (1.3.1, CPU) variant scorer — local VCF pipeline.

Contract (see models/_template/README.md):
    models/spliceai/.venv/bin/python score.py --input <variants.parquet> \
        --output scores.parquet

Run with THIS model's own venv (models/spliceai/.venv), which pins
spliceai==1.3.1 + tensorflow==2.19.1 + setuptools<80 — the repo venv does NOT
carry tensorflow. --mock works anywhere (no spliceai import).

Score column: `spliceai_ds` = max(DS_AG, DS_AL, DS_DG, DS_DL) from the
SpliceAI VCF INFO (same summary as the companion benchmark, milestone 6).
Higher = more splice-altering = more damaging. No flip.

Pipeline (mirrors companion refs/run_spliceai_full.sh):
  1. emit a GRCh38 VCF of input SNVs (Ensembl contig naming, no "chr")
  2. split into chunks, run `spliceai -R data/refs/grch38_subset.fa
     -A grch38 -D 50` per chunk in parallel
  3. parse chunk outputs, map back by (chrom,pos,ref,alt)

Scope: SNVs only for now (the frozen matrix's 17,786 indels need VCF
normalisation checks — deferred, recorded in NOTES.md). Chunk outputs are
cached in cache/chunks/; completed chunks are skipped on rerun.

Reference: data/refs/grch38_subset.fa (+ .fai), Ensembl release-112,
chromosomes 2/3/13/16/17 — built by data/refs build steps (see NOTES.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MODEL_NAME = "spliceai_ds"
MODEL_DIR = Path(__file__).resolve().parent
REPO = MODEL_DIR.parents[1]
REF_FASTA = REPO / "data" / "refs" / "grch38_subset.fa"
# GRCh38 chromosome lengths (Ensembl release-112), for the VCF header.
CHR_LEN = {"2": 242193529, "3": 198295559, "13": 114364328, "16": 90338345, "17": 83257441}
N_CHUNKS = 24  # ~1.9k variants each so a chunk finishes in ~2 min; .done is
# keyed to the chunk VCF md5 so killed/altered chunks are always redone


def emit_vcf(df: pd.DataFrame, vcf_path: Path) -> int:
    """SNV-only VCF, Ensembl contig names, sorted by chrom,pos."""
    snv = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)].copy()
    snv["chrom_s"] = snv["chrom"].astype(str)
    snv = snv.sort_values(["chrom_s", "pos"])
    with open(vcf_path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        for c, ln in CHR_LEN.items():
            fh.write(f"##contig=<ID={c},length={ln}>\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for r in snv.itertuples():
            fh.write(f"{r.chrom_s}\t{int(r.pos)}\t.\t{r.ref}\t{r.alt}\t.\t.\t.\n")
    return len(snv)


def parse_spliceai_vcf(path: Path) -> dict:
    """(chrom,pos,ref,alt) -> max delta score, from one SpliceAI output VCF."""
    scores: dict = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            chrom, pos, ref, alt = f[0], int(f[1]), f[3], f[4]
            info = f[7] if len(f) > 7 else ""
            best = None
            for entry in info.split(";"):
                if entry.startswith("SpliceAI="):
                    for rec in entry[len("SpliceAI="):].split(","):
                        p = rec.split("|")
                        deltas = [float(x) for x in p[2:6]]
                        best = max(deltas) if best is None else max(best, max(deltas))
            if best is not None:
                scores[(chrom, pos, ref, alt)] = best
    return scores


def run_chunks(vcf_path: Path, chunks_dir: Path, n_chunks: int, distance: int) -> dict:
    """Split VCF into chunks, run spliceai per chunk (parallel), parse all."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    lines = vcf_path.read_text().splitlines()
    header = [l for l in lines if l.startswith("#")]
    body = [l for l in lines if not l.startswith("#")]
    chunk_size = math.ceil(len(body) / n_chunks)
    chunk_vcfs = []
    for i in range(n_chunks):
        part = body[i * chunk_size : (i + 1) * chunk_size]
        if not part:
            continue
        cv = chunks_dir / f"chunk_{i:02d}.vcf"
        cv.write_text("\n".join(header + part) + "\n")
        chunk_vcfs.append(cv)

    exe = Path(sys.executable).parent / "spliceai"
    import os
    env = dict(os.environ)
    # 6 TF processes on 10 cores: stop thread-pool thrashing.
    env.update({"TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1",
                "TF_CPP_MIN_LOG_LEVEL": "3"})

    def one(cv: Path) -> Path:
        out = cv.with_suffix(".out.vcf")
        done_marker = cv.with_suffix(".done")
        want_md5 = hashlib.md5(cv.read_bytes()).hexdigest()
        if done_marker.exists() and out.exists() and done_marker.read_text() == want_md5:
            return out
        done_marker.unlink(missing_ok=True)  # stale marker from a different input
        cmd = [str(exe), "-I", str(cv), "-O", str(out), "-R", str(REF_FASTA),
               "-A", "grch38", "-D", str(distance)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"spliceai failed on {cv.name}: {proc.stderr[-400:]}")
        done_marker.write_text(want_md5)
        return out

    with ThreadPoolExecutor(max_workers=min(len(chunk_vcfs), 5)) as ex:
        outs = list(ex.map(one, chunk_vcfs))
    scores: dict = {}
    for out in outs:
        scores.update(parse_spliceai_vcf(out))
    return scores


def score_mock(df: pd.DataFrame) -> pd.Series:
    """Deterministic pseudo delta-scores in [0, 1), for offline tests."""
    def one(row) -> float:
        d = hashlib.sha256(
            f"mock-spliceai:{row['chrom']}:{row['pos']}:{row['ref']}:{row['alt']}".encode()
        ).hexdigest()
        return (int(d[:8], 16) % 1000) / 1000.0

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
    ap.add_argument("--distance", type=int, default=50, help="SpliceAI -D flanking distance")
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
        df[MODEL_NAME] = score_mock(df)
        n_snv = int(((df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)).sum())
    else:
        if not REF_FASTA.exists():
            raise SystemExit(f"reference missing: {REF_FASTA} (see NOTES.md build steps)")
        vcf_path = cache_dir / "variants.vcf"
        n_snv = emit_vcf(df, vcf_path)
        print(f"spliceai: {n_snv} SNVs -> VCF; running {args.chunks} chunks", flush=True)
        scores = run_chunks(vcf_path, cache_dir / "chunks", args.chunks, args.distance)
        df[MODEL_NAME] = df.apply(
            lambda r: scores.get((str(r["chrom"]), int(r["pos"]), str(r["ref"]), str(r["alt"]))),
            axis=1,
        )

    out = df.dropna(subset=[MODEL_NAME])
    out.to_parquet(args.output, index=False)

    run_log = {
        "model": "SpliceAI 1.3.1 (local, CPU, tensorflow 2.19.1)",
        "reference": "data/refs/grch38_subset.fa (Ensembl release-112, chr 2/3/13/16/17)",
        "annotation": "spliceai bundled grch38",
        "distance": args.distance,
        "score_definition": "max(DS_AG, DS_AL, DS_DG, DS_DL); higher = more splice-altering",
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
