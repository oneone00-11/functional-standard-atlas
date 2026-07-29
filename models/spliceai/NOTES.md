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
