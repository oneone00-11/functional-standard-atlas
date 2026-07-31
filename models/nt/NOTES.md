# NT (Nucleotide Transformer) scorer — NOTES

## What

`nucleotide_transformer` = masked-token log-likelihood ratio from
**NT v2 500M multi-species** (`InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`):

```
score = log P(REF 6-mer | masked context) − log P(ALT 6-mer | masked context)
```

**HIGHER = alt allele more disfavoured = MORE pathogenic.** Mirrors the
companion benchmark milestone 8b/11 definition (`scripts/92_score_nt.py`;
there verified on ClinVar: splice P/LP mean ≈ +2.9 vs B/LB ≈ +0.5), so the
atlas run must pass concordance vs the legacy `nucleotide_transformer`
column (21,410 shared variants).

## Parameters (same as companion)

- Window **6000 bp** centred on the variant, start shifted so variant offset
  is a multiple of 6 (variant inside exactly one 6-mer token).
- Tokenizer: non-overlapping 6-mer; +1 position offset for the CLS token.
- Skips (→ unscored): FASTA ref-base mismatch; N/ambiguous base in window;
  ref/alt 6-mer → UNK token.
- **SNVs only** (masked-token LLR undefined for length changes; indels deferred).

## Known caveat — window provenance

The companion's original cloud run did **not** record its window
(legacy `docs/OPEN_ITEMS.md`); 6000 bp is the documented default, not a
confirmed original. Concordance ρ=1.0 is hoped but not guaranteed — if
0.95 < ρ < 1.0, window mismatch is the documented prime suspect, and the
finding (definition sensitivity) feeds the paper's reproducibility narrative.

## Environment & run

- GPU required (CUDA). Env `nt` built on the rented pod — see
  `cluster/RUNBOOK_GPU.md` for the exact build; `requirements.txt` pins are
  frozen from the pod after the run (rule 3).
- Reference: `data/refs/grch38_subset.fa` + `.fai` (shared; rebuilt on the pod).
- Resumable: `cache/nt_scores_cache.tsv` keyed by variant_id, checkpointed
  every 200 variants. Expected rate on an L40S: ~10–25 variants/s
  (500M model, single-variant forward passes, definition fidelity) →
  46,392 SNVs ≈ 0.5–1.5 h.
- Full run: `<env>/bin/python models/nt/score.py --input
  data/frozen/frozen-matrix-v1.parquet --output results/nt_scores.parquet`
  (on the pod, inside tmux — see RUNBOOK).

## After the run

1. `rsync` `results/nt_scores.parquet` back to the Mac.
2. Concordance vs legacy column (expected ρ≈1.0, caveat above) → evaluate →
   assemble → heatmap v3 (same flow as SpliceAI/Pangolin).
3. Paste the pod's `pip freeze` versions into `requirements.txt`, delete the
   "pinned post-run" notes, commit.
