"""Re-score Pangolin at full float precision.

Like SpliceAI, Pangolin computes its splice-gain and splice-loss scores as
floats and rounds them only when formatting the output line:

    gain_str = f"{g-d}:{round(gain[g],2)}"
    loss_str = f"{l-d}:{round(loss[l],2)}"

With the atlas settings (``-d 50``, default masking, no score cutoff, exon
scoring off) those two lines are the only place precision is lost — masking
alters the gain/loss arrays but not the formatting. 54% of coding variants and
71% of deep-intronic variants therefore land on exactly 0.00, which bounds
Spearman ρ below 1 for arithmetic reasons alone (see ``atlas.robustness``).

This script reuses Pangolin's own model loading, annotation database and
sequence handling, and patches only those two format strings. ``--validate``
checks that the full-precision output, re-rounded to two decimals, reproduces
the stock function exactly.

Pangolin evaluates twelve models per variant, so one process needs well over a
day for the atlas. The run is therefore split into chunks that workers claim
atomically: start as many workers as there are free cores, stop them at any
time, and re-run to pick up whatever is left. A chunk file is written via a
temporary file and renamed, so it never appears half-complete.

Run with the Pangolin environment:
    models/pangolin/.venv/bin/python models/pangolin/score_fullprec.py --validate
    for i in 1 2 3 4; do models/pangolin/.venv/bin/python \
        models/pangolin/score_fullprec.py --run & done
"""

from __future__ import annotations

import argparse
import inspect
import re
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
REPO = MODEL_DIR.parents[1]
MATRIX = REPO / "results" / "score_matrix_atlas_v1.parquet"
REF_FASTA = REPO / "data" / "refs" / "grch38_subset.fa"
GTF_DB = REPO / "data" / "refs" / "grch38_subset.gtf.db"
OUT = REPO / "results" / "pangolin_scores_fullprec.parquet"
CHUNK_DIR = MODEL_DIR / "cache" / "fullprec_chunks"
DISTANCE = 50

PANG_RE = re.compile(r"\|(-?\d+):(-?[0-9.eE+-]+)\|(-?\d+):(-?[0-9.eE+-]+)\|Warnings")


class Args:
    """Stand-in for the argparse namespace process_variant expects."""

    def __init__(self, distance: int):
        self.distance = distance
        self.mask = "True"          # Pangolin's default, and what the atlas used
        self.score_cutoff = None
        self.score_exons = "False"
        self.reference_file = str(REF_FASTA)


def build_fullprec(PP: types.ModuleType):
    """`process_variant` with only the two output roundings removed."""
    src = inspect.getsource(PP.process_variant)
    patched, n1 = re.subn(r'round\(gain\[g\],\s*2\)', "float(gain[g]):.17g", src)
    patched, n2 = re.subn(r'round\(loss\[l\],\s*2\)', "float(loss[l]):.17g", patched)
    if (n1, n2) != (1, 1):
        raise RuntimeError(f"expected one gain and one loss rounding, found {n1} and {n2}")
    ns = dict(vars(PP))
    exec(compile(patched, "<pangolin-fullprec>", "exec"), ns)
    return ns["process_variant"]


def aggregate(scores: str) -> float:
    """max(splice gain, |splice loss|) — the atlas definition, unchanged."""
    best = np.nan
    if not isinstance(scores, str):
        return best
    for _gp, gain, _lp, loss in PANG_RE.findall(scores):
        agg = max(float(gain), -float(loss))
        best = agg if np.isnan(best) else max(best, agg)
    return best


def load_variants() -> pd.DataFrame:
    df = pd.read_parquet(MATRIX, columns=["variant_id", "chrom", "pos", "ref", "alt",
                                          "gene", "pangolin_score"])
    snv = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)].copy()
    snv["chrom_s"] = snv["chrom"].astype(str)
    return snv.sort_values(["chrom_s", "pos"]).reset_index(drop=True)


def setup(PP: types.ModuleType):
    import gffutils
    import torch
    from pkg_resources import resource_filename

    gtf = gffutils.FeatureDB(str(GTF_DB))
    models = []
    for i in [0, 2, 4, 6]:
        for j in range(1, 4):
            model = PP.Pangolin(PP.L, PP.W, PP.AR)
            weights = torch.load(
                resource_filename("pangolin", "models/final.%s.%s.3.v2" % (j, i)),
                map_location=torch.device("cpu"))
            model.load_state_dict(weights)
            model.eval()
            models.append(model)
    return gtf, models


def _claim_is_stale(claim_f: Path) -> bool:
    """True when the process that wrote this claim is no longer running."""
    try:
        pid = int(claim_f.read_text().strip())
    except (OSError, ValueError):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunks", type=int, default=96)
    args_cli = ap.parse_args()

    import pangolin.pangolin as PP

    args = Args(DISTANCE)
    gtf, models = setup(PP)
    fullprec = build_fullprec(PP)
    variants = load_variants()

    if args_cli.validate:
        sample = variants.sample(n=min(120, len(variants)), random_state=20260803)
        checked = mismatched = 0
        for r in sample.itertuples():
            stock = PP.process_variant(0, r.chrom_s, int(r.pos), r.ref, r.alt,
                                       gtf, models, args)
            fine = fullprec(0, r.chrom_s, int(r.pos), r.ref, r.alt, gtf, models, args)
            if stock == -1 or fine == -1:
                continue
            sp, fp = PANG_RE.findall(stock), PANG_RE.findall(fine)
            if len(sp) != len(fp):
                mismatched += 1
                continue
            for (_a, sg, _b, sl), (_c, fg, _d, fl) in zip(sp, fp):
                checked += 1
                if f"{float(fg):.2f}" != f"{float(sg):.2f}" or \
                   f"{float(fl):.2f}" != f"{float(sl):.2f}":
                    mismatched += 1
                    print("MISMATCH", (sg, sl), "vs", (fg, fl))
        print(f"validated {checked} score pairs on {len(sample)} variants; "
              f"{mismatched} mismatches")
        return 1 if mismatched else 0

    if not args_cli.run:
        ap.error("pass --validate or --run")

    if args_cli.limit:
        variants = variants.head(args_cli.limit)

    # Chunked, claim-based work sharing. Pangolin evaluates twelve models per
    # variant, so a single process needs >24 h for the atlas; workers claim
    # chunks atomically, so any number can be started or stopped at any time and
    # an interrupted chunk is simply re-claimed.
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    chunk_idx = np.array_split(np.arange(len(variants)), args_cli.chunks)
    t0 = time.time()
    n_done_here = 0

    while True:
        claimed = None
        for idx, rows_idx in enumerate(chunk_idx):
            done_f = CHUNK_DIR / f"chunk_{idx:03d}.tsv"
            claim_f = CHUNK_DIR / f"chunk_{idx:03d}.claim"
            if done_f.exists():
                continue
            try:
                fd = os.open(claim_f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                # a claim whose worker has died must not block its chunk forever
                if _claim_is_stale(claim_f):
                    claim_f.unlink(missing_ok=True)
                continue
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            claimed = (idx, variants.iloc[rows_idx], done_f, claim_f)
            break
        if claimed is None:
            break

        idx, chunk, done_f, claim_f = claimed
        rows = []
        for r in chunk.itertuples():
            try:
                sc = fullprec(0, r.chrom_s, int(r.pos), r.ref, r.alt, gtf, models, args)
                val = aggregate(sc) if sc != -1 else np.nan
            except Exception:
                val = np.nan
            rows.append(f"{r.variant_id}\t{val}")
        tmp = done_f.with_suffix(".part")
        tmp.write_text("\n".join(rows) + "\n")
        tmp.rename(done_f)          # atomic: a chunk file only ever appears complete
        claim_f.unlink(missing_ok=True)
        n_done_here += len(chunk)
        el = (time.time() - t0) / 60
        remaining = sum(1 for i in range(len(chunk_idx))
                        if not (CHUNK_DIR / f"chunk_{i:03d}.tsv").exists())
        print(f"chunk {idx:>3} done ({len(chunk)} variants); this worker "
              f"{n_done_here:,} in {el:.1f} min; {remaining} chunks left", flush=True)

    outstanding = [i for i in range(len(chunk_idx))
                   if not (CHUNK_DIR / f"chunk_{i:03d}.tsv").exists()]
    if outstanding:
        print(f"worker exiting; {len(outstanding)} chunks still outstanding "
              "(another worker holds them, or re-run to pick them up)")
        return 0

    scored: dict[str, float] = {}
    for i in range(len(chunk_idx)):
        for line in (CHUNK_DIR / f"chunk_{i:03d}.tsv").read_text().splitlines():
            if not line.strip():
                continue
            vid, val = line.split("\t")
            scored[vid] = float(val)

    out = variants[["variant_id", "gene", "chrom", "pos", "ref", "alt"]].copy()
    out["pangolin_fullprec"] = out["variant_id"].map(scored)
    out["pangolin_rounded"] = variants["pangolin_score"].to_numpy()
    out.to_parquet(OUT, index=False)

    ok = out.dropna(subset=["pangolin_fullprec", "pangolin_rounded"])
    print(f"\nwrote {OUT} ({len(out):,} variants, "
          f"{out['pangolin_fullprec'].notna().sum():,} scored)")
    if len(ok):
        print("max |round(fullprec,2) - stored rounded| = "
              f"{(ok['pangolin_fullprec'].round(2) - ok['pangolin_rounded']).abs().max():.4f}")
    print(f"distinct values: {out['pangolin_fullprec'].nunique():,} "
          f"(was {out['pangolin_rounded'].nunique():,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
