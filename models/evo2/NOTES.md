# Evo 2 scorer — NOTES

## What

`evo2` = windowed log-likelihood delta from **Evo 2 7B**
(`arcinstitute/evo2_7b`, StripedHyena2, bf16):

```
score = LL(REF 8192-bp window) − LL(ALT 8192-bp window)   # Evo2.score_sequences
```

**HIGHER = alt allele more disfavoured = MORE pathogenic.** There is NO
legacy evo2 column to mirror (the companion matrix's `evo2` column is all
NA) — this definition is new, introduced by the atlas, and recorded here as
the standard. No concordance ρ=1.0 gate; instead the smoke test (GPU setup
step 6) sanity-checks that known pathogenic splice variants score high.

## Parameters

- Window **8192 bp** centred on the variant (inside evo2_7b's native
  context; balances signal vs runtime/VRAM). Do not change silently — the
  value is part of the score definition.
- Skips (→ unscored): FASTA ref-base mismatch; N/ambiguous base in window;
  window < 1 kb at chromosome edge.
- **SNVs only** (LL delta would support indels, but SNV-only keeps the
  column comparable with NT/SpliceAI/Pangolin; indels deferred).

## Environment (official "light install", 7B only)

- CUDA GPU ≥ 24 GB VRAM (L40S 48 GB used); Linux; Python 3.12.
- `torch==2.7.1` (cu128) → `flash-attn==2.8.0.post2 --no-build-isolation`
  (torch first!) → `evo2`. Exact pins frozen post-run (rule 3).
- Weights via HuggingFace `arcinstitute/evo2_7b` — download on the pod
  BEFORE scoring (compute-side internet not guaranteed).

## Run

- Resumable: `cache/evo2_scores_cache.tsv` keyed by variant_id,
  checkpointed every 100 variants; `--batch-size 4` default
  (8 sequences per forward).
- Expected rate on L40S: ~2–6 variants/min is a conservative planning
  figure → 46,392 SNVs ≈ overnight. If OOM: lower `--batch-size`.
- Full run (inside tmux on the pod):
  `<env>/bin/python models/evo2/score.py --input
  data/frozen/frozen-matrix-v1.parquet --output results/evo2_scores.parquet`

## After the run

1. `rsync` `results/evo2_scores.parquet` back to the Mac.
2. Evaluate → assemble → heatmap v3 (no concordance step — new column).
3. Paste pod `pip freeze` versions into `requirements.txt`, commit.
