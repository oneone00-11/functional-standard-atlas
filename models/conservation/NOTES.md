# Conservation scorer notes (phyloP100way / phastCons100way)

- **Source**: UCSC Genome Browser track API,
  `https://api.genome.ucsc.edu/getData/track?genome=hg38&track=<track>`,
  same source as companion benchmark `scripts/73_conservation.py`.
- **Tracks**: `phyloP100way`, `phastCons100way` (hg38). GERP is not served
  natively for hg38 by the UCSC API (documented in the companion report);
  deferred.
- **Invocation**: one run per track (`--track phyloP100way` /
  `--track phastCons100way`); output column is the lowercased track name
  (`phylop100way` / `phastcons100way`), matching the companion matrix columns.
- **Coordinates**: API intervals are 0-based half-open; matrix positions are
  1-based. Conversion: a position p (1-based) is covered by interval
  [start, end) iff start + 1 <= p <= end.
- **Fetch strategy**: one request per genomic block (variant positions
  clustered at 50 kb gaps → the 7 gene spans). ~14 requests total for the
  whole matrix — no rate-limit risk (contrast with the per-variant CADD API
  incident, 2026-07-28). Raw interval JSON cached per (track, block) under
  `cache/regions/<track>/`; reruns are instant.
- **Orientation**: higher value = more conserved = more damaging. No flip.
- **Coverage expectation**: companion matrix had 21,410/21,410 for both
  tracks. Positions without a track value (gaps in the 100-way alignment)
  are dropped, not imputed.
- **Date scored (atlas full matrix)**: see `run_log_<col>.json`.

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

Run 2026-11-22T15:34Z: phyloP and phastCons 5,095 scored each, 0 unscored; on the 4,522
rows the release scored, each equal to the release on every row.
