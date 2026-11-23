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

## External genes (2026-11-22)

- `build_external_refs.py` builds `data/refs/grch38_external.fa` (+ `.fai`),
  `grch38_external.gtf` and `grch38_external.gtf.db`: Ensembl release 112, chromosomes
  1, 7, 16 and 17, by the recipe above; `grch38_subset.*` is untouched. It checks
  chromosomes 1 and 7 (FASTA and GTF rows) byte for byte against what the companion
  release (tag v3.1.0-submission) scored on, reading the hashes from the release's run
  logs, and chromosomes 16 and 17 against `grch38_subset.fa`; record:
  `data/refs/grch38_external.build.json`.
- `score_external.py` runs `score_fullprec.py` unchanged (sha256 checked with
  `--expect-sha256` and recorded) on a table of variants, here the 5,095 intronic SNVs
  of the five external genes, with that reference and database. **One thread per
  process** (`OMP_NUM_THREADS=MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`): the
  companion release found the scorer reproduces its anchors bit for bit on one thread
  and not on 2, 4, 6 or 8. Six worker processes, each with its own resumable cache in
  `cache/external_v1/`; `--collect` writes `cache/external_v1/pangolin.parquet` and its
  `.environment.json`.
- Environment: the Pangolin venv the companion release recorded for its external
  columns (that of the older checkout); pip freeze identical.
- The delivered column is `pangolin_score` of `results/external_scores_v1.parquet`.
- Run 2026-11-22T15:29:57Z to 2026-11-23T09:35:49Z (the machine slept from 2026-11-22T18:14Z to 2026-11-23T09:15Z and the run went on from its cache): 5,095 scored by
  the six workers, 0 unscorable, 0 NaN, 0 exceptions. On the 4,522 rows the release scored
  every value equals the release's (largest absolute difference 0).
