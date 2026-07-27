# AlphaGenome scorer — NOTES

- **Model**: AlphaGenome (Google DeepMind), remote API, non-commercial use
- **Client**: `alphagenome==0.7.0` (PyPI), verified 2026-07-27
- **API key**: register free at https://deepmind.google.com/science/alphagenome →
  `export ALPHAGENOME_API_KEY=...` (never commit the key)
- **Score definition** (AlphaGenome paper's recommended splicing variant score):
  per variant, `score_variant` with the recommended scorers for `SPLICE_SITES`,
  `SPLICE_SITE_USAGE`, `SPLICE_JUNCTIONS`; merged as
  `max(SPLICE_SITES) + max(SPLICE_SITE_USAGE) + max(SPLICE_JUNCTIONS)/5` on the
  client's **`quantile_score`** column (scale-normalised across tracks; client
  0.7.0 `tidy_scores` exposes `raw_score`/`quantile_score`, verified live
  2026-07-26). Larger = more damaging; no orientation flip needed.
  NOTE: the companion benchmark (variant-fm-benchmark, `scripts/91_score_alphagenome.py`)
  used a different aggregate — `max |raw_score|` across splice tracks with a
  1-Mb interval and client v0.6.1. The two definitions are NOT interchangeable;
  see OPEN DECISION below before scoring the full matrix.
- **Interval**: variant-centred, width 2^14 bp (default; `--width` to change).
  The API only supports sequence lengths 16384 / 131072 / 524288 / 1048576
  (verified live 2026-07-26) — 2^15 is rejected. Intervals are 0-based
  half-open, variant positions 1-based VCF-style
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

## DECISION (resolved 2026-07-27, before full-matrix scoring)

**Atlas standard = the developer-recommended merged splicing score on
`quantile_score`, 16 kb interval, client 0.7.0** (as implemented in `score.py`).

Rationale:
1. An independent benchmark is most defensible when every model is scored with
   its developer's recommended protocol ("even-handed by construction").
2. `quantile_score` is scale-normalised across tracks, which is the only
   meaningful basis for cross-track `max`; `raw_score` tracks are not
   commensurable.
3. 16 kb intervals are ~60x cheaper than 1-Mb per call; 64k variants on 1-Mb
   intervals is infeasible under the free API's rate limits.

Compatibility with the companion benchmark (v0.6.1, `max|raw_score|`, 1-Mb):
evaluation is Spearman rank-based and therefore robust to monotone scorer
changes. As a robustness note for the paper, run an old-vs-new concordance
check on ~200 splice variants (old scores from variant-fm-benchmark vs new
scores from this pipeline) and report their Spearman; target > 0.9.
