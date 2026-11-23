# SpliceAI scorer notes

- **Model**: SpliceAI 1.3.1 (PyPI `spliceai`), tensorflow 2.19.1 CPU,
  `setuptools<80` (SpliceAI imports `pkg_resources`) — identical versions to
  the companion benchmark's WSL env (milestone 6). Lives in the model-local
  venv `models/spliceai/.venv`; the repo venv deliberately stays TF-free.
- **Invocation**: `models/spliceai/run_full.sh` (or `score.py` with the model
  venv's python). Score column `spliceai_ds` = max(DS_AG, DS_AL, DS_DG,
  DS_DL); higher = more splice-altering. No flip. `-A grch38` bundled
  annotation, `-D 50` (companion-identical).
- **Reference**: `data/refs/grch38_subset.fa` (+ `.fai`) — Ensembl
  release-112 per-chromosome FASTA for chr 2/3/13/16/17, concatenated.
  Download: `curl -O https://ftp.ensembl.org/pub/release-112/fasta/
  homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.<N>.fa.gz` for
  N in {2,3,13,16,17}, then `gunzip -c` in that order > grch38_subset.fa,
  `pysam.faidx`. Contig names have no `chr` prefix, matching our matrix and
  SpliceAI's annotation (verified in the companion repo).
- **Scope**: SNVs only. The frozen matrix's 17,786 indels are deferred —
  SpliceAI scores indels in principle, but VCF indel normalisation
  (anchor base, left-alignment) needs a verification pass first.
- **Chunking & resume**: VCF split into 24 chunks (~1.9k variants), run with
  5-way parallelism, 2 intra-op TF threads per process. Each completed chunk
  writes a `.done` marker containing the MD5 of its input VCF — a chunk is
  redone whenever its input changed (a stale `.done` from the 300-variant
  benchmark initially masked a full-matrix chunk; fixed 2026-07-29).
- **Throughput** (measured 2026-07-29, 10-core M-series, 32 GB):
  300 variants in 2m25s wall / 13m26s CPU. Full matrix ≈ 1–4 h wall.
  NOT runnable inside an agent command window — use `run_full.sh` in a
  terminal; killed runs resume from the chunk cache.
- **Benchmark verification**: first 300 variants all scored; UTR variants
  return 0.0 deltas, matching the companion matrix exactly.
- **Date scored (atlas full matrix)**: see `run_log.json` after the full run.

## External genes (2026-11-22)

- `score_external.py` runs `score_fullprec.py` unchanged (its sha256 checked with
  `--expect-sha256` and recorded) on a table of variants: here the 5,095 intronic SNVs
  of CTCF, SFPQ, RAD51D, XRCC2 and SBDS (`data/raw/external_v1/score_input.parquet`,
  written by `python -m atlas.external_scores check`).
- Reference `data/refs/grch38_external.fa`: Ensembl release 112, chromosomes 1, 7, 16
  and 17, built and checked by `models/pangolin/build_external_refs.py`. SpliceAI's
  bundled `grch38` annotation, `-D 50`, mask 0, full precision: as for the core genes.
- Environment: this folder's `.venv`, the one the companion release (tag
  v3.1.0-submission) recorded for its external columns; pip freeze identical. One
  process, values cached per variant (`cache/external_v1/spliceai.tsv`), resumable.
- Output `cache/external_v1/spliceai.parquet` and its `.environment.json`; the
  delivered column is `spliceai_ds` of `results/external_scores_v1.parquet`, whose
  provenance record carries the dates and counts.
- Run 2026-11-22T15:29:57Z to 2026-11-23T09:34:43Z (the machine slept from 2026-11-22T18:14Z to 2026-11-23T09:15Z and the run went on from its cache): 5,095 scored, 0 NaN,
  0 exceptions. On the 4,522 rows the release scored (3-50 bp) every value equals the
  release's (largest absolute difference 0).
