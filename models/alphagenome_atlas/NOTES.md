# AlphaGenome Atlas scorer — NOTES

- **Product**: AlphaGenome Atlas (Google DeepMind), released 2026-09-08: precomputed
  variant scores behind their own API, a different service from the per-variant scoring
  of `models/alphagenome/`. Client `alphagenome==0.9.0` (`alphagenome.atlas`), the client
  the companion release used, so that the columns are the release's to the bit (0.10.0
  is not used).
- **Columns**: `avi` (scorer `AVI_SCORE`, the Atlas's combined variant impact score),
  `avi_splice_sites`, `avi_splice_site_usage`, `avi_splice_junctions` (scorers
  `SPLICE_SITES`, `SPLICE_SITE_USAGE`, `SPLICE_JUNCTIONS`). Per variant, each is the
  maximum over tracks and over rows (`SPLICE_JUNCTIONS` returns one row per junction) of
  the scorer's `X`, which client 0.9.0 fills with the service's raw scores; the calibrated
  scores in the `quantiles` layer are not read (the release's
  `alphagenome_official/ATLAS_COLUMNS_CHECK.json`).
- **Scorer**: `score.py`, ported from the companion release (tag v3.1.0-submission, read
  with `git show`): `phase1/src/evid_score_avi.py`'s `run()` and `_aggregate()`, and the
  guards of `scripts/stage_E18/run_avi.py`. Batches of 250 variants; six attempts per
  batch with doubling pauses (20 s after a quota error, 5 s after any other, plus 2 s);
  an authentication error, or a batch out of attempts, stops the run; a checkpoint after
  every batch, so a stopped run resumes. The scorers the service offers are listed and
  recorded (`cache/<run>/scorers.json`) before anything is scored. `--mock` runs offline
  for the tests.
- **Variants**: SNVs only, as the Atlas is precomputed for them: the frozen matrix's
  46,392 SNVs and the external genes' 5,095 intronic SNVs
  (`data/raw/alphagenome_atlas_v1/score_input.parquet`, written by
  `python -m atlas.alphagenome_atlas inputs`).
- **Environment**: `~/work/envs/alphagenome-atlas` (Python 3.12.13), the environment the
  release's Atlas columns were scored in: its `pip freeze` equals the release's record
  (`phase1/data/evidence/external/e18_anchors_avi.parquet.environment.json`, 48
  packages) package by package. `requirements.txt` is written after the run by
  `pin_requirements.py` (each package with the sha256 of the file pip picks, and the
  downloaded wheel's RECORD compared with the installed files).
- **Key**: only through `read_key()` (the environment variable, or
  `~/.config/alphagenome/key` with permissions 0600); never printed or written.
- **Licence**: AlphaGenome API outputs and Atlas information are for non-commercial
  research use only and must not be used to train other models.
- **Dependency to keep in mind**: the Atlas's combined score (AVI) was selected on a
  validation set that includes BRCA1, RAD51C and DDX3X experiments (the companion paper's
  reference 25); BRCA1 and RAD51C are core genes here. Recorded in the provenance; how the
  rebuild treats it is decided there.

## Runs

- 2026-11-24, anchors (`--anchors`, `cache/anchors_v1/`): the service's 22 scorers equal
  to the release's list of 2026-09-20 (`avi_scorers_20260920.json`), one by one; the 39
  anchors equal to the release's `external/e18_anchors_avi.parquet` in all four columns,
  to the bit (04:44:51Z to 04:44:53Z, one request, 0 retries).
- 2026-11-24, full (`--run`, `cache/atlas_v1/`): the scorer list again equal to the
  release's; the 51,487 SNVs from 19:06:33Z to 19:26:33Z in one run, 206 calls (batches of
  250), 0 retries, 0 failed batches, no error; every variant has all four columns. On the
  rows the release scored, the four columns equal the release's on every row: the 8,853
  analysis-set rows of `avi.parquet` and the 4,522 of `external/newgenes_avi.parquet`
  (`results/alphagenome_atlas_concordance_v1.tsv`, with the anchors' check).
- Environment after the run: `pip freeze` equal to the release's two records
  (`e18_anchors_avi` and `newgenes_avi`), package by package, 48 packages;
  `requirements.txt` and `requirements.check.json` written then (19:27:14Z): 48 wheels,
  each one's RECORD equal to the installed files.
- The columns: `results/alphagenome_atlas_scores_v1.parquet` (variant_id, gene, set and the
  four columns) with its provenance record (the licence, and the validation-set
  dependency above).
