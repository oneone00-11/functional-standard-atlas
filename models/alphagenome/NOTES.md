# AlphaGenome scorer — NOTES

- **Model**: AlphaGenome (Google DeepMind), remote API, non-commercial use
- **Client**: `alphagenome==0.7.0` (PyPI), verified 2026-07-27
- **API key**: register free at https://deepmind.google.com/science/alphagenome →
  `export ALPHAGENOME_API_KEY=...` (never commit the key)
- **Score definition** (AlphaGenome paper's recommended splicing variant score,
  same as the companion benchmark): per variant, `score_variant` with the
  recommended scorers for `SPLICE_SITES`, `SPLICE_SITE_USAGE`, `SPLICE_JUNCTIONS`;
  merged as `max(SPLICE_SITES) + max(SPLICE_SITE_USAGE) + max(SPLICE_JUNCTIONS)/5`.
  Larger = more damaging; no orientation flip needed.
- **Interval**: variant-centred, width 2^15 bp (default; `--width` to change);
  intervals 0-based half-open, variant positions 1-based VCF-style
- **Caching**: `cache/scores.jsonl`, key = sha256(chrom:pos:ref:alt:width:client_version).
  Re-runs resume from cache. Failed variants go to `cache/failed.jsonl` after 3
  retries with backoff, and are dropped from output (coverage measured downstream).
- **Run metadata**: every run (re)writes `run_log.json` with client version,
  score definition, timestamp and counts — quote these in the paper, do not
  hand-write version numbers from memory.

## Known limits

- The API is rate-limited and intended for thousands (not millions) of
  predictions; score in batches and rely on the cache for resume.
- Scores depend on `--width`; keep it fixed within a study and report it.
