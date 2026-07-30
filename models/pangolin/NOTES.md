# Pangolin scorer — NOTES

## What

Local **Pangolin** (git `tkzeng/Pangolin`, pip reports 1.0.2) splice-model
scorer, CPU torch 2.13.0. Mirrors the companion benchmark's milestone 6
recipe so scores are definition-identical (verified by concordance ρ).

- Input: frozen matrix SNVs → CSV (`CHROM,POS,REF,ALT`, Ensembl contig names)
- Command per chunk: `pangolin -d 50 chunk.csv data/refs/grch38_subset.fa
  data/refs/grch38_subset.gtf.db chunk.out` (default mask)
- Output: `chunk.out.csv`; 5th field holds per-transcript records
  `...|<gain_pos>:<gain>|<loss_pos>:<loss>|Warnings...`
- **Score**: `pangolin_score = max(gain, |loss|)` over records; winning
  record's `pangolin_gain` / `pangolin_loss` retained.
  **Higher = more splice-altering = more damaging. No flip.**

## Environment

- `models/pangolin/.venv` — torch CPU wheel + `git+https://github.com/tkzeng/Pangolin.git`
  + gffutils 0.14 + pyfastx 2.3.1 + PyVCF3 1.0.4 (exact pins in requirements.txt)
- **PyVCF3 is imported at pangolin startup** (`import vcf` in pangolin.py)
  even in CSV mode — must be installed, but its VCF writer is never used.
  The companion benchmark hit the PyVCF3 `_Info` signature incompatibility in
  VCF mode → we use **CSV mode** throughout (same decision).

## Reference files (data/refs/, gitignored)

- `grch38_subset.fa` + `.fai` — Ensembl release-112 FASTA, chroms
  2/3/13/16/17 (built for SpliceAI; shared).
- `Homo_sapiens.GRCh38.112.gtf.gz` — Ensembl r112 GTF
  (`https://ftp.ensembl.org/pub/release-112/gtf/homo_sapiens/Homo_sapiens.GRCh38.112.gtf.gz`).
- `grch38_subset.gtf` — subset to chroms 2/3/13/16/17 (911,843 lines):
  `gunzip -c ...gtf.gz | awk '$1=="2"||$1=="3"||$1=="13"||$1=="16"||$1=="17"'`
- `grch38_subset.gtf.db` — gffutils db (762 MB, ~19 s build):
  `gffutils.create_db(subset_gtf, db, force=True, keep_order=True,
  merge_strategy="create_unique", disable_infer_genes=True,
  disable_infer_transcripts=True)`

Chromosome naming: no `chr` prefix everywhere (FASTA, GTF, matrix) — matches
companion verification.

## Scope & known limitations

- **SNVs only** (46,392 of the 64,178 frozen variants). Indels deferred —
  same rationale as SpliceAI (VCF/CSV normalisation unvalidated).
- Variants outside annotated gene windows may be absent from Pangolin output
  → reported as unscored in run_log.json (companion had 0 such cases on the
  same annotation).

## Performance lessons (from companion run)

- 5 parallel workers max; **stagger starts 15 s** — 8 simultaneous torch
  model loads OOM'd the companion's 7 GB box. We cap `OMP_NUM_THREADS=2`,
  `MKL_NUM_THREADS=2` per process.
- Chunked + md5-keyed `.done` markers: reruns skip completed chunks.

## Full run

`models/pangolin/run_full.sh` (user terminal; multi-hour torch CPU job).
Then concordance vs companion `pangolin_score` (expected ρ=1.0), evaluate,
assemble, heatmap — same as SpliceAI flow.
