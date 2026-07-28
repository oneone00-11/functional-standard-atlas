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
