# GPN-MSA scorer notes

- **Source**: HuggingFace dataset `songlab/gpn-msa-hg38-scores`,
  `scores.tsv.bgz` (37,274,961,495 bytes, from CDN Content-Range header,
  verified 2026-07-28). Contigs are plain "17"-style; columns
  chrom,pos,ref,alt,score. Same file as companion benchmark
  `scripts/90_score_gpn.py`.
- **Index**: `gpn_scores.tsv.bgz.tbi` (2.8 MB) is pinned in this directory,
  copied from the companion repo `refs/` where it was committed.
- **Orientation FLIP**: GPN-MSA raw scores are more negative = more
  deleterious. The score column is `gpn_msa = -raw` so that larger = more
  damaging (contract rule). Concordance vs the companion's raw
  `gpn_msa_score` column is therefore expected at **ρ = -1.0**.
- **2026-07-28 xet-CDN problem**: HuggingFace migrated the dataset to xet
  storage; htslib remote seeking against the xet bridge fails with
  "Illegal seek" even though plain HTTP Range requests (both closed and
  open-ended) return correct 206 responses. Workaround implemented in
  `score.py`: parse the local tabix index to compute exact compressed byte
  ranges for the needed genomic blocks, download only those ranges (plus
  bgzf header/EOF blocks) into a sparse file, then fetch with pysam locally.
  Whole-matrix run downloads ~10 MB. Filled ranges are tracked in
  `cache/filled_ranges.json`; reruns are incremental.
- **Coverage**: the file contains all possible SNVs; companion matrix had
  21,410/21,410. Indels are not in the table (dropped, not imputed).
- **Spot check 2026-07-28**: first BRCA1 variants read −3.79/−3.96/−3.08 raw,
  identical to the companion matrix values.
- **Date scored (atlas full matrix)**: see `run_log.json`.

## External genes (2026-11-22)

`python -m atlas.external_scores lookups` runs this folder's `score.py` on the 5,095
intronic SNVs of CTCF, SFPQ, RAD51D, XRCC2 and SBDS, unchanged, from a byte-identical
copy in `cache/external_v1/` (sha256 checked), so that the run log it writes beside
itself does not overwrite the record of the core-gene run here. The environment is the
one the companion release (tag v3.1.0-submission) recorded for its external lookup
columns; pip freeze identical. The remote source's version is read and recorded before
scoring (`data/raw/external_v1/lookups.sources.json`). Output and run log:
`cache/external_v1/`; the delivered column is in `results/external_scores_v1.parquet`,
whose provenance record carries the dates and counts.

Run 2026-11-22T15:34Z (revision cf1718a9, the pinned one): 5,095 scored, 0 not in the
table; on the 4,522 rows the release scored, equal to the release on every row.
