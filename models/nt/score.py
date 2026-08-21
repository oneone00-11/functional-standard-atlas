"""Nucleotide Transformer (NT v2 500M) variant scorer — masked-token LLR.

Contract (see models/_template/README.md):
    <gpu env>/bin/python score.py --input <variants.parquet> --output scores.parquet

Runs on a CUDA GPU (rented pod / cluster) with torch + transformers; --mock
works anywhere (no torch import).

Score column: `nucleotide_transformer` = log P(REF 6-mer | masked context)
- log P(ALT 6-mer | masked context) at the 6-mer token holding the variant.
HIGHER = alternate allele more disfavoured = MORE pathogenic. This mirrors the
companion benchmark's milestone 8b/11 definition exactly (scripts/92_score_nt.py,
verified there: splice P/LP mean +2.9 vs B/LB +0.5), so concordance vs the
legacy `nucleotide_transformer` column is expected.

- Checkpoint: InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
  (non-overlapping 6-mer tokenizer, masked-LM head)
- Window: 6000 bp centred on the variant, start shifted so the variant
  offset is a multiple of 6 (variant inside exactly one token).
  NOTE: the companion's original cloud run did NOT record its window
  (legacy docs/OPEN_ITEMS.md); 6000 is the documented default. If
  concordance rho < 0.99, window mismatch is the prime suspect.
- Skips (score = None): FASTA ref-base mismatch, N/ambiguous base in window,
  ref/alt 6-mer maps to UNK. Counted as unscored in run_log.json.

Scope: SNVs only (masked-token LLR is undefined for length change; indels
deferred). Resumable: cache/nt_scores_cache.tsv is keyed by variant_id and
checkpointed every --checkpoint-every variants; reruns skip cached ids.

Reference: data/refs/grch38_subset.fa (+ .fai), Ensembl release-112,
chromosomes 2/3/13/16/17 (shared with SpliceAI/Pangolin).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MODEL_NAME = "nucleotide_transformer"
MODEL_DIR = Path(__file__).resolve().parent
REPO = MODEL_DIR.parents[1]
REF_FASTA = REPO / "data" / "refs" / "grch38_subset.fa"

CHECKPOINT = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
KMER = 6
WINDOW_BP = 6000
BASES = set("ACGT")


def masked_llr(tok, model, device, chrom_seq: str, pos0: int, ref: str, alt: str):
    """log P(ref 6-mer) - log P(alt 6-mer) at the masked variant token.

    chrom_seq: full chromosome sequence (str), pos0: 0-based variant index.
    Returns None on ref mismatch / ambiguous window / UNK token.
    """
    import torch

    if pos0 < 0 or pos0 >= len(chrom_seq) or chrom_seq[pos0].upper() != ref.upper():
        return None
    half = WINDOW_BP // 2
    start = max(0, pos0 - half)
    end = min(len(chrom_seq), pos0 + half)
    var_off = pos0 - start
    start += var_off % KMER  # align so the variant sits inside one token
    var_off = pos0 - start
    window = chrom_seq[start:end].upper()
    if set(window) - BASES:
        return None

    tok_idx = var_off // KMER
    within = var_off % KMER
    ref_kmer = window[tok_idx * KMER:(tok_idx + 1) * KMER]
    if len(ref_kmer) < KMER or ref_kmer[within] != ref.upper():
        return None
    alt_kmer = ref_kmer[:within] + alt.upper() + ref_kmer[within + 1:]

    enc = tok(window, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    mask_pos = tok_idx + 1  # +1 for the leading CLS token
    ref_id = tok.convert_tokens_to_ids(ref_kmer)
    alt_id = tok.convert_tokens_to_ids(alt_kmer)
    if tok.unk_token_id in (ref_id, alt_id):
        return None
    input_ids[0, mask_pos] = tok.mask_token_id
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, mask_pos]
        logp = torch.log_softmax(logits, dim=-1)
    return float(logp[ref_id] - logp[alt_id])


def score_mock(df: pd.DataFrame) -> pd.Series:
    """Deterministic pseudo-LLRs, for offline tests."""

    def one(row) -> float:
        d = hashlib.sha256(
            f"mock-nt:{row['chrom']}:{row['pos']}:{row['ref']}:{row['alt']}".encode()
        ).hexdigest()
        return ((int(d[:8], 16) % 6000) - 3000) / 1000.0  # in [-3, 3)

    return df.apply(one, axis=1)


def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    prev = pd.read_csv(cache_path, sep="\t", dtype={"variant_id": str})
    return dict(zip(prev["variant_id"], prev[MODEL_NAME]))



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
    ap.add_argument("--checkpoint-every", type=int, default=200)
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
    cache_path = cache_dir / "nt_scores_cache.tsv"

    if args.mock:
        df[MODEL_NAME] = score_mock(df)
        n_snv = int(((df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)).sum())
    else:
        if not REF_FASTA.exists():
            raise SystemExit(f"reference missing: {REF_FASTA} (see NOTES.md build steps)")
        import torch
        from pyfaidx import Fasta
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        from tqdm import tqdm

        snv_mask = (df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)
        n_snv = int(snv_mask.sum())
        done = load_cache(cache_path)
        print(f"nt: {n_snv} SNVs in input, {len(done)} cached", flush=True)

        fasta = Fasta(str(REF_FASTA))
        tok = AutoTokenizer.from_pretrained(CHECKPOINT)
        model = AutoModelForMaskedLM.from_pretrained(CHECKPOINT)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        print(f"nt: {CHECKPOINT} on {device}", flush=True)

        scores: dict = {}
        seq_cache: dict = {}
        new_since_ckpt = 0
        for r in tqdm(df.itertuples(index=False), total=len(df)):
            if r.variant_id in done:
                scores[r.variant_id] = done[r.variant_id]
                continue
            val = None
            if (len(r.ref) == 1) and (len(r.alt) == 1):
                chrom = str(r.chrom)
                if chrom not in seq_cache:
                    seq_cache[chrom] = str(fasta[chrom])
                try:
                    val = masked_llr(tok, model, device, seq_cache[chrom],
                                     int(r.pos) - 1, r.ref, r.alt)
                except Exception:
                    val = None
            scores[r.variant_id] = val
            new_since_ckpt += 1
            if new_since_ckpt >= args.checkpoint_every:
                pd.DataFrame(
                    [(k, v) for k, v in scores.items()], columns=["variant_id", MODEL_NAME]
                ).to_csv(cache_path, sep="\t", index=False)
                new_since_ckpt = 0
        pd.DataFrame(
            [(k, v) for k, v in scores.items()], columns=["variant_id", MODEL_NAME]
        ).to_csv(cache_path, sep="\t", index=False)
        df[MODEL_NAME] = df["variant_id"].map(scores)

    out = df.dropna(subset=[MODEL_NAME])
    out.to_parquet(args.output, index=False)

    run_log = {
        "model": f"Nucleotide Transformer v2 500M ({CHECKPOINT})",
        "reference": "data/refs/grch38_subset.fa (Ensembl release-112, chr 2/3/13/16/17)",
        "window_bp": WINDOW_BP,
        "score_definition": "logP(ref 6-mer|masked) - logP(alt 6-mer|masked); higher = more pathogenic",
        "scope": "SNVs only (indels deferred)",
        "window_caveat": "companion original run window unrecorded (legacy OPEN_ITEMS); 6000 bp is documented default",
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
