# CADD scorer notes

- **Sources** (`--source`):
  - `precomputed` (default, added 2026-07-28): remote tabix slices of the
    official whole-genome SNV file `CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz`
    (81 GB; US mirror `krishna.gs.washington.edu`, DE mirror
    `kircherlab.bihealth.org`; both verified to serve HTTP Range on
    2026-07-28; contigs are plain "7"-style; columns
    chrom,pos,ref,alt,RawScore,PHRED). One region tsv per genomic block is
    cached under `cache/regions/`; reruns are instant. Reproduces the API
    values exactly (concordance vs companion matrix: **ρ = 1.0, n = 19,883**).
  - `api`: per-variant web API `https://cadd.gs.washington.edu/api/v1.0/GRCh38-v1.7/`.
    When the API returns several annotation records for one allele, the
    **max PHRED** is taken (identical rule to companion benchmark
    `scripts/75_cadd.py`).
- **API version**: GRCh38-v1.7 (both sources serve the same v1.7 scores).
- **Score column**: `cadd` = PHRED. **Orientation**: higher = more deleterious.
- **Coverage (atlas full matrix)**: 46,392/46,392 SNVs (**100%**). The 17,786
  non-SNVs (indels) in the frozen matrix are absent from the SNV-only
  precomputed file BY DESIGN — dropped, not imputed. Indel scoring would need
  the gnomAD r4.0 indel file (known indels only) or offline scoring; out of
  scope for the SNV-focused evaluation.
- **API cache**: per-variant JSONL keyed by SHA-256(input file) + coordinates
  + API version; `null` = HTTP 200 empty record list for an INDEL (genuinely
  absent from CADD's precomputed set), cached deliberately. For SNVs an empty
  list is ALWAYS a rate-limiting artifact and raises instead of being cached.
  Transient failures go to `failed.jsonl`, never the cache.
- **2026-07-28 incident**: under 5-worker concurrency the API started
  load-shedding — HTTP 200 with `[]` for every variant including BRAF V600E —
  and 61,513 bogus nulls were cached. Nulls purged (backup:
  `cache/scores.jsonl.bak-ratelimited`), workers cut to 2 with a 0.2 s delay,
  SNV-empty responses made retryable, and the precomputed source added as the
  new default. Probe API recovery with `7:140753336_A_T` (BRAF V600E).
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

Run 2026-11-22T15:34Z: 5,095 scored, 0 unscored; on the 4,522 rows the release scored,
equal to the release on every row.
