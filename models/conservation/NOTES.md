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
