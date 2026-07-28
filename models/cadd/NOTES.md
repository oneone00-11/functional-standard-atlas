# CADD scorer notes

- **Source**: CADD web API, `https://cadd.gs.washington.edu/api/v1.0/GRCh38-v1.7/`
- **API version**: GRCh38-v1.7 (from the API URL path; companion benchmark used the same)
- **Score column**: `cadd` = PHRED-scaled score. When the API returns several
  annotation records for one allele, the **max PHRED** is taken (identical rule
  to companion benchmark `scripts/75_cadd.py`).
- **Orientation**: higher = more deleterious. No flip needed.
- **Coverage expectation**: CADD scores essentially all SNVs; companion matrix
  coverage was 19,897/21,410 (93%) — the missing ~7% were API give-up errors
  under a flaky proxy, not unscorable variants. This port retries and caches
  per-variant, so coverage on the frozen 64,178-variant matrix should be higher.
- **Cache**: per-variant JSONL keyed by SHA-256(input file) + coordinates + API
  version; `null` score = CADD returned HTTP 200 with an empty record list
  (genuinely unscored), cached deliberately so reruns do not re-hammer the API.
  Transient failures (rate limiting, non-200 statuses, odd body shapes) raise
  and are logged to `failed.jsonl` but NEVER cached, so reruns refetch them.
  (First smoke run cached transient nulls under 5-worker concurrency and lost
  14/50 variants; fixed 2026-07-28.)
- **Date first scored (atlas full matrix)**: see `run_log.json`.
