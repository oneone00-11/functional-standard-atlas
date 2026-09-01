# Project conventions

Rules the analysis code is written against. They are referenced by number in
module docstrings ("CONVENTIONS.md rule 5"), so they are recorded here rather than left
implicit.

## 1. Frozen data is never edited in place

An analysis that changes a frozen input produces a new versioned file
(`*_v2.parquet`) and leaves the original untouched. `data/frozen/` carries
SHA-256 manifests and is treated as immutable once written.

## 2. Every scored column is reproducible from a committed script

No column enters the score matrix by hand. Each has a scorer under `models/`
with its own environment, and its provenance is recorded in that model's
`NOTES.md`.

## 3. Dependencies are pinned after the run, not before

`requirements.txt` lists the direct dependencies; the resolved environment is
captured with `pip freeze` once the run has completed, so the pins describe what
actually produced the numbers.

## 4. A source column's meaning is verified, never assumed

Before a deposited column is used it is checked against something independent —
a published interval, a replicate mean, a documented scale. Columns that fail
the check are recorded as unusable rather than used with a caveat.

## 5. Numbers are read from pipeline outputs, never transcribed

Every figure, table and quoted value is produced by a script reading a file
under `results/`. Nothing is copied by hand from one document to another, so a
rerun cannot leave a stale number behind.

## 6. Development output never lands in `results/`

`results/` is published: it ships in the release archive and is cited from the
manuscript. Smoke tests, probes and one-off checks write to `results/_scratch/`,
which `atlas.package_release` excludes and which nothing downstream may read.

The rule exists because three July smoke-test artefacts —
`ag_test.parquet`, `ag_test_report_v1.md` and `cadd_smoke.parquet` — reached the
v2.3.1 and v2.3.2 archives. One of them carried a section headed OPEN DECISION
recording a scoring choice that the manuscript has since settled, and named a
term the release denylist rejects outside `models/`. Nothing referenced them.
They were invisible because the hygiene gate scanned `git ls-files` while
`results/` is gitignored; the gate now scans the packaging output instead.

A per-file exclusion list would have hidden the next one. The place is the rule.
