# gnomAD AF baseline scorer notes

- **Source**: gnomAD GraphQL API `https://gnomad.broadinstitute.org/api`,
  dataset `gnomad_r4` (GRCh38), queried per genomic region (one request per
  ~50 kb variant block; ~7 requests for the whole matrix). Raw region
  responses cached under `cache/regions/`; reruns are instant.
- **Metrics** (run once per `--metric`, identical definitions to companion
  benchmark `scripts/72_gnomad.py`):
  - `global` → `gnomad_af_global` = max(genome.af, exome.af)
  - `popmax` → `gnomad_af_popmax` = max over continental populations
    {afr, amr, eas, nfe, sas, fin, mid, asj} of ac/an (genome and exome)
- **Orientation FLIP**: higher AF = more likely benign. Score columns are
  `-AF` so larger = more damaging (contract rule). Concordance vs the
  companion's raw AF columns is therefore expected at **ρ = -1.0**.
- **Absence ≠ 0**: variants not present in gnomAD are left UNSCORED (dropped,
  never imputed as AF = 0), matching the companion benchmark's conservative
  choice. Rationale: absence can reflect low coverage / poor mappability as
  well as true rarity; a sensitivity analysis treating absence as AF = 0 is a
  legitimate downstream experiment but is NOT baked into the score column.
- **Coverage expectation**: companion matrix 7,708/21,410 (36%) for global
  AF. Low absolute coverage is expected — most SGE-saturated variants are
  ultra-rare or unobserved in population databases, which is itself the
  point of the functional-vs-population comparison.
- **Date scored (atlas full matrix)**: see `run_log_<col>.json`.

## External genes (2026-11-22)

`python -m atlas.external_scores lookups` runs this folder's `score.py` on the 5,095
intronic SNVs of CTCF, SFPQ, RAD51D, XRCC2 and SBDS, unchanged, from a byte-identical
copy in `cache/external_v1/` (sha256 checked), so that the run log it writes beside
itself does not overwrite the record of the core-gene run here. The companion release
scored no gnomAD column for these genes, so the definition, scorer and dataset
(gnomad_r4) are the core genes'; the environment is the lookup environment the release
recorded for its CADD, GPN-MSA and conservation columns, which holds this scorer's pins
(requests 2.32.5, pandas 3.0.5, pyarrow 21.0.0). The remote source is recorded before
scoring (`data/raw/external_v1/lookups.sources.json`). Output and run log:
`cache/external_v1/`; the delivered column is in `results/external_scores_v1.parquet`,
whose provenance record carries the dates and counts.

Run 2026-11-22T15:35Z: 1,316 of the 5,095 found in gnomAD and scored (global and
popmax); the other 3,779 are not in gnomAD and stay unscored, not filled. The release has
no column to compare.
