# AlphaMissense scorer notes

- **Source**: `AlphaMissense_hg38.tsv.gz`, Zenodo record 8208688
  (`https://zenodo.org/records/8208688/files/AlphaMissense_hg38.tsv.gz`,
  643 MB, HEAD-verified 2026-07-28). Same file as companion benchmark
  `scripts/74_alphamissense.py`. No version string beyond "hg38" is published
  with the file; the Zenodo record is the canonical citation.
- **Score column**: `alphamissense` = am_pathogenicity.
  **Orientation**: higher = more likely pathogenic. No flip.
- **Scope**: missense SNVs only, by design of the model. Non-missense
  variants are absent from the table and are dropped, never imputed.
  Companion-matrix coverage was 11,926/21,410; the atlas matrix has a higher
  intronic fraction, so relative coverage will be lower — that is expected,
  not a failure.
- **Matching**: exact on (chrom, pos, ref, alt) after stripping the "chr"
  prefix; per-chrom coordinate windows prune the scan.
- **Cache**: the gz itself is the cache (~643 MB in `cache/`); the
  stream-filter costs ~2-4 min per run.
- **Date scored (atlas full matrix)**: see `run_log.json`.
