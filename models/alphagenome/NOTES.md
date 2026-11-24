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

## Official definition (2026-11-22)

- `score_official.py` scores the official variant-scoring notebook's merged splicing
  score, ported from the companion release's `scripts/stage_E17/score_official.py`
  (tag v3.1.0-submission). The code runs from the release's `NOTEBOOK_SOURCE.json`
  (copied to `data/raw/companion_v3.1.0/` by `python -m atlas.external_scores
  companion`; sha256 checked): cells 5, 9 and 13 as written, and cell 15's lines for one
  variant (a 1 Mb interval centred on it, the three scorers). The merged score is the
  column `alphagenome_official`, apart from this folder's `alphagenome` column (the
  older definition above); the three scorers are kept beside it.
- Environment: this folder's `.venv`, client `alphagenome==0.10.0`, installed from
  `requirements_official.txt` (the release's requirements file, byte for byte) with
  `--require-hashes --no-deps`; pip freeze identical to the release's record.
- Key: only through `read_key()` (the environment variable, or
  `~/.config/alphagenome/key` with permissions 0600). Never printed or written; records
  say only whether a key was found and where.
- Guards: two threads at most; doubling pauses between attempts; stop on an
  authentication error, on three variants in a row out of attempts, on a negative raw
  score. `--check-start` scores the release's ten drift-check variants afresh and
  compares them with the release; `--external` scores the variants; `--check-end`
  scores the ten again and must equal `--check-start` to the last bit. `--mock` runs
  offline for the tests.
- Research use only; the scores train no other model. Outputs in `cache/official_v1/`;
  the delivered columns are in `results/external_scores_v1.parquet`.
- Run: the start's drift check equal to the release on all ten variants, to the last
  bit; the 5,095 variants 2026-11-22T15:31:03Z to 16:21:45Z on two threads, 0 retries,
  0 failures; the end's drift check equal to the start's, to the last bit. On the 4,522
  rows the release scored, the merged score and the three scorers equal the release's
  on every row.

## The frozen matrix by the official definition (2026-11-24)

- `python -m atlas.alphagenome_official` (`src/atlas/alphagenome_official.py`) writes the
  input and assembles what the scorer wrote; `score_official.py` is unchanged (sha256
  3bb934200a35624f…, as in A2) and runs in this folder's `.venv` as above. Input
  `data/raw/alphagenome_official_atlas_v1/score_input.parquet`: the frozen matrix's 64,178
  variants, its 46,392 SNVs first and then the 17,786 others (9,714 deletions, 64
  insertions, 8,008 complex), each part in the matrix's order. Outputs in
  `cache/official_atlas_v1/`; the columns `alphagenome_official`,
  `alphagenome_splice_sites`, `alphagenome_splice_site_usage` and
  `alphagenome_splice_junctions` in `results/alphagenome_official_scores_v1.parquet`, with
  its provenance record.
- Trial: 20 variants drawn with seed 20261124 (10 SNVs; a deletion, an insertion and a
  complex variant at ±1–2 bp; 7 other non-SNVs), into `cache/official_atlas_v1_trial/`:
  every one scored by all three scorers, no raw score below zero, 0 retries.
- Run: the start's drift check equal to the release on all ten variants, to the last bit
  (2026-11-24T03:53Z). The 64,178 variants on two threads, in three segments (stopped
  twice with Ctrl+C and resumed from the cache): 14,932 from 03:54:03Z to 06:18:13Z,
  29,185 from 07:41:22Z to 12:24:17Z, 20,061 from 15:33:49Z to 18:59:52Z. The last
  segment 0 retries and 0 failures; the scorer records calls and retries only for the
  segment that completes a run, so the first two have none recorded. Every variant has a
  row from each scorer and no NaN. The end's drift check (19:00:27Z to 19:00:32Z) equal to
  the start's, to the last bit, and scored after the run.
- Against the release (`results/alphagenome_official_concordance_v1.tsv`): on the release
  analysis set's 8,853 SNVs, the merged score and the three scorers equal the release's
  `all_scores.tsv` (read round-trip) on every row.
- Against this folder's older `alphagenome` column (reported only): Spearman ρ 0.763 over
  the 64,178 variants, 0.752 over the SNVs; by stratum, over all variants, from 0.437 at
  ±1–2 bp to 0.924 beyond 50 bp; by gene × stratum in the same file.
