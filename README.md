# functional-standard-atlas

A functional-standard atlas for benchmarking genomic variant effect predictors and
foundation models — scaled beyond splice regions and single-gene studies.

Companion project to:

- *Functional-evidence benchmarking of splice- and sequence-based variant effect
  predictors in hereditary cancer genes* (under review, Bioinformatics)
- *Ranking is saturated but calibration is not* (under review, Briefings in Bioinformatics)

## What this repo is

1. **A versioned, content-hashed data asset**: SGE/MAVE functional score sets
   (MaveDB) + saturation-mutagenesis MPRA sets, coordinate-validated to GRCh38,
   frozen into immutable matrices with a SHA-256 manifest.
2. **A uniform scoring harness**: one directory per predictor/foundation model,
   each with a pinned environment, a `score.py` entry point and cached outputs.
3. **An evaluation pipeline**: per-assay Spearman correlation pooled by
   DerSimonian–Laird random-effects meta-analysis, stratified by variant class
   (missense / splice / regulatory / indel), with leave-one-assay-out validation.

## Layout

```
config/assays.yaml        # assay registry (MaveDB URNs, tiers, notes)
data/raw/                 # source downloads, hashed and manifest-tracked (gitignored)
data/frozen/              # frozen analysis matrices + manifest (gitignored, hash-pinned)
models/<model>/           # one folder per model: score.py + env lock + cached scores
models/_template/         # copy this to add a new model
src/atlas/                # package: fetch, manifest, mapping, evaluation
tests/                    # pytest; must pass before any result is trusted
results/                  # generated tables/figures (gitignored, script-produced)
```

## Quick start

```bash
make setup     # install pinned dependencies
make fetch     # download all registered assays + build hash manifest
make test      # run the guardrail tests
```

Add a new assay → edit `config/assays.yaml`, re-run `make fetch`.
Add a new model → `cp -r models/_template models/<name>` and implement `score.py`.

## Reproducibility contract

Everything in `data/` and `results/` is produced by scripts and is reproducible
from tracked sources with `make fetch && make test`. No external identifier
(MaveDB URN, RefSeq transcript, genome build) may be written from memory —
it must be pulled from source by a script and hash-recorded.
