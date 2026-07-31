"""Evo 2 (7B) variant scorer — windowed log-likelihood delta.

Contract (see models/_template/README.md):
    <gpu env>/bin/python score.py --input <variants.parquet> --output scores.parquet

Requires a CUDA GPU with the official evo2 stack (torch + flash-attn + evo2,
see requirements.txt / cluster/RUNBOOK_GPU.md); --mock works anywhere.

Score column: `evo2` = LL(REF window) − LL(ALT window), where LL is the
model's autoregressive sequence log-likelihood from
`Evo2.score_sequences([...])`, over an 8192-bp window centred on the variant.
HIGHER = the alt allele is more disfavoured by the model = MORE pathogenic
(same orientation convention as the NT column; there is no legacy evo2
column to mirror — the companion matrix's `evo2` column is entirely NA — so
this definition is NEW for the atlas and recorded here as the standard).

- Checkpoint: arcinstitute/evo2_7b (StripedHyena2, bf16)
- Window: 8192 bp centred (well inside evo2_7b's native context; keeps
  VRAM and runtime practical — recorded here, do not change silently)
- Skips (→ unscored): FASTA ref-base mismatch; window containing
  N/ambiguous bases; window truncated at chromosome end below 1 kb.
- Scope: SNVs only (alignment-free LL delta would technically support
  indels, but SNV-only keeps the column comparable across models; indels
  deferred, recorded here).

Batching: ref/alt sequence pairs are scored in batches of 2*--batch-size
sequences per forward pass. Resumable: cache/evo2_scores_cache.tsv keyed by
variant_id, checkpointed every --checkpoint-every variants.

Reference: data/refs/grch38_subset.fa (+ .fai), Ensembl release-112,
chromosomes 2/3/13/16/17 (shared with SpliceAI/Pangolin/NT).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MODEL_NAME = "evo2"
MODEL_DIR = Path(__file__).resolve().parent
REPO = MODEL_DIR.parents[1]
REF_FASTA = REPO / "data" / "refs" / "grch38_subset.fa"

CHECKPOINT = "evo2_7b"  # Evo2() registry name; weights from arcinstitute/evo2_7b (HF)
WINDOW_BP = 8192
MIN_WINDOW_BP = 1000
BASES = set("ACGT")


def extract_window(fasta_chrom: str, pos0: int) -> str | None:
    """8192-bp window centred at pos0 (0-based), or None if too short."""
    half = WINDOW_BP // 2
    start = max(0, pos0 - half)
    end = min(len(fasta_chrom), pos0 + half)
    window = fasta_chrom[start:end].upper()
    if len(window) < MIN_WINDOW_BP or set(window) - BASES:
        return None
    return window


def score_mock(df: pd.DataFrame) -> pd.Series:
    """Deterministic pseudo-LLRs, for offline tests."""

    def one(row) -> float:
        d = hashlib.sha256(
            f"mock-evo2:{row['chrom']}:{row['pos']}:{row['ref']}:{row['alt']}".encode()
        ).hexdigest()
        return ((int(d[:8], 16) % 4000) - 2000) / 1000.0  # in [-2, 2)

    return df.apply(one, axis=1)


def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    prev = pd.read_csv(cache_path, sep="\t", dtype={"variant_id": str})
    return dict(zip(prev["variant_id"], prev[MODEL_NAME]))


def write_cache(cache_path: Path, scores: dict) -> None:
    pd.DataFrame(
        [(k, v) for k, v in scores.items()], columns=["variant_id", MODEL_NAME]
    ).to_csv(cache_path, sep="\t", index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None, help="score only first N variants")
    ap.add_argument("--mock", action="store_true", help="offline deterministic scores (tests)")
    ap.add_argument("--cache-dir", default=str(MODEL_DIR / "cache"))
    ap.add_argument("--batch-size", type=int, default=4, help="variants per forward batch")
    ap.add_argument("--checkpoint-every", type=int, default=100)
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
    cache_path = cache_dir / "evo2_scores_cache.tsv"

    if args.mock:
        df[MODEL_NAME] = score_mock(df)
        n_snv = int(((df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)).sum())
    else:
        if not REF_FASTA.exists():
            raise SystemExit(f"reference missing: {REF_FASTA} (see NOTES.md build steps)")
        from evo2 import Evo2
        from pyfaidx import Fasta
        from tqdm import tqdm

        snv_mask = (df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)
        n_snv = int(snv_mask.sum())
        done = load_cache(cache_path)
        print(f"evo2: {n_snv} SNVs in input, {len(done)} cached", flush=True)

        fasta = Fasta(str(REF_FASTA))
        model = Evo2(CHECKPOINT)
        print(f"evo2: {CHECKPOINT} loaded", flush=True)

        scores: dict = {k: v for k, v in done.items()}
        todo = []
        seq_cache: dict = {}
        for r in df.itertuples(index=False):
            if r.variant_id in done:
                continue
            val = None
            if (len(r.ref) == 1) and (len(r.alt) == 1):
                chrom = str(r.chrom)
                if chrom not in seq_cache:
                    seq_cache[chrom] = str(fasta[chrom])
                chrom_seq = seq_cache[chrom]
                pos0 = int(r.pos) - 1
                if 0 <= pos0 < len(chrom_seq) and chrom_seq[pos0].upper() == r.ref.upper():
                    window = extract_window(chrom_seq, pos0)
                    if window is not None:
                        wstart = max(0, pos0 - WINDOW_BP // 2)
                        within = pos0 - wstart
                        alt_window = window[:within] + r.alt.upper() + window[within + 1:]
                        todo.append((r.variant_id, window, alt_window))
                        continue
            scores[r.variant_id] = val  # None -> unscored

        print(f"evo2: {len(todo)} variant windows to score", flush=True)
        new_since_ckpt = 0
        for i in tqdm(range(0, len(todo), args.batch_size)):
            batch = todo[i : i + args.batch_size]
            seqs = []
            for _vid, ref_w, alt_w in batch:
                seqs.extend([ref_w, alt_w])
            try:
                lls = model.score_sequences(seqs)
            except Exception as exc:  # surface the first real error loudly
                raise RuntimeError(
                    f"evo2 score_sequences failed at batch {i}: {exc}. "
                    "If this is an API-name mismatch, check the installed evo2 "
                    "package's README/notebooks for the scoring entry point."
                ) from exc
            for (vid, _r, _a), ll_ref, ll_alt in zip(batch, lls[0::2], lls[1::2]):
                scores[vid] = float(ll_ref - ll_alt)
            new_since_ckpt += len(batch)
            if new_since_ckpt >= args.checkpoint_every:
                write_cache(cache_path, scores)
                new_since_ckpt = 0
        write_cache(cache_path, scores)
        df[MODEL_NAME] = df["variant_id"].map(scores)

    out = df.dropna(subset=[MODEL_NAME])
    out.to_parquet(args.output, index=False)

    run_log = {
        "model": f"Evo 2 ({CHECKPOINT}, arcinstitute/evo2_7b, bf16)",
        "reference": "data/refs/grch38_subset.fa (Ensembl release-112, chr 2/3/13/16/17)",
        "window_bp": WINDOW_BP,
        "score_definition": "LL(ref window) - LL(alt window) via Evo2.score_sequences; higher = more pathogenic",
        "scope": "SNVs only (indels deferred)",
        "concordance": "no legacy evo2 scores exist; this definition is new for the atlas",
        "mock": args.mock,
        "snv_in_input": n_snv,
        "scored": len(out),
        "unscored": len(df) - len(out),
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    (MODEL_DIR / "run_log.json").write_text(json.dumps(run_log, indent=2) + "\n")
    print(f"{MODEL_NAME}: scored {len(out)}/{len(df)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
