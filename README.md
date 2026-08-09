# functional-standard-atlas

An attenuation-corrected, territory-resolved benchmark of variant effect
predictors against saturation genome editing (SGE) measurements: 64,178
variants, seven cancer susceptibility genes, nineteen predictors.

Accompanies the manuscript listed in `CITATION.cff`. Two companion manuscripts
on overlapping data are under review at *Bioinformatics* and *Briefings in
Bioinformatics*; they benchmark embedding architectures and share no results
section with this work.

## What this repository is

1. **A frozen, content-hashed data asset.** Seven MaveDB SGE score sets mapped
   to GRCh38, orientation-harmonised, frozen into immutable matrices with a
   SHA-256 manifest.
2. **A uniform scoring harness.** One directory per predictor, each with a
   pinned environment, a `score.py` entry point and cached outputs.
3. **An evaluation pipeline.** Per-gene Spearman ρ pooled by DerSimonian–Laird
   random-effects meta-analysis across sixteen strata, with per-stratum
   measurement-reliability estimates, attenuation correction, paired
   dependent-correlation tests and leave-one-gene-out validation.

Every number in the manuscript is read from a file under `results/`; every
figure and table is produced by a committed script.

## Reproduction: what you can run, and what you cannot

**Read this before cloning.** `results/`, `data/raw/`, `data/frozen/`,
`data/refs/` and `models/*/cache/` are gitignored. **A `git clone` gives you
code only and reproduces no figure.** The four tiers below are honest about
where each stops.

| Tier | You have | You can | Wall clock | Disk |
|---|---|---|---|---|
| 1 | git clone | run the 83 guardrail tests | ~25 s | 20 MB |
| 2 | **+ Zenodo archive** | **regenerate every figure and table in the paper** | **< 1 min** | 60 MB |
| 3 | + network | rebuild the frozen matrix from MaveDB | ~20 min | 2.2 GB |
| 4 | + API key + GPU | re-score every predictor from scratch | ~40 h | 6 GB |

**Tier 2 is the one that matters for review.** Download the Zenodo archive
(DOI in `CITATION.cff`), which ships `results/` and `data/frozen/`, then:

```bash
make setup                                     # pinned dependencies, ~2 min
PYTHONPATH=src python figures/atlas_overview.py            # Fig 1
PYTHONPATH=src python figures/hardening_figures.py         # Figs 2-6, Supplemental Fig S15
PYTHONPATH=src python figures/splice_territory_comparison.py   # Supplemental Fig S14
PYTHONPATH=src python figures/reproducibility.py           # Supplemental Fig S13
```

Each script takes under three seconds and writes PNG + PDF into `results/`.
The analysis modules that produce the underlying tables are equally cheap:

```bash
PYTHONPATH=src python -m atlas.reliability          # attenuation ceilings
PYTHONPATH=src python -m atlas.evaluate_ext         # the 19 x 16 grid
PYTHONPATH=src python -m atlas.robustness           # ties, power, paired tests
PYTHONPATH=src python -m atlas.simulate_attenuation # 151,200-trial validation, ~4 min
PYTHONPATH=src python -m atlas.definition_sweep     # reads the cached sweep
```

### What needs credentials, and what a reviewer without them loses

Tier 4 needs two things this repository cannot ship:

- **An AlphaGenome API key** (`ALPHAGENOME_API_KEY`; free for non-commercial
  use). Needed to regenerate the `alphagenome` column and to re-run the
  definition sweep. **A reviewer without a key can still reproduce every
  figure and number**, because the scored columns and the 10,888 cached sweep
  responses are in the archive. What they cannot do is re-derive those scores
  from the API.
- **A GPU** (one NVIDIA A6000, 48 GB) for Evo2-7B and NT-v2-500M. Evo2-7B alone
  took 25 h for 46,392 SNVs. Again, the scored columns are archived; only
  re-derivation needs the hardware.

Everything else — SpliceAI, Pangolin, CADD, AlphaMissense, GPN-MSA, the
conservation tracks, gnomAD and the seven dbNSFP meta-predictors — is
reproducible on CPU with network access.

## Full pipeline (tiers 3–4)

```bash
make setup     # install pinned dependencies
make fetch     # download registered assays + reference genome, build hash manifest
make freeze    # map to GRCh38 (Mutalyzer + Ensembl), build the frozen matrix
make test      # 83 guardrail tests
```

`make fetch` pulls ~2.1 GB of reference sequence, which is why `data/refs/` is
neither committed nor archived.

## Layout

```
config/assays.yaml     assay registry (MaveDB URNs, tiers, notes)
data/raw/              source downloads, hash-tracked          (gitignored; in archive)
data/frozen/           frozen matrices + SHA-256 manifest      (gitignored; in archive)
data/refs/             GRCh38 FASTA + annotation               (gitignored; NOT archived)
models/<name>/         score.py, pinned env, NOTES.md provenance
src/atlas/             fetch, mapping, evaluation, reliability, robustness,
                       clinical evidence, ensembles, simulation, definition sweep
figures/               figure-generating scripts
tests/                 pytest guardrails; must pass before any result is trusted
results/               all pipeline outputs                    (gitignored; in archive)
```

## Adding a predictor or an assay

Copy `models/_template/`, or add the MaveDB URN to `config/assays.yaml` and
re-run `make fetch`. the project conventions state the rules that keep results trustworthy
— frozen data are immutable, dependencies are pinned exactly, statistics are
read from pipeline outputs rather than transcribed, and every figure comes from
a committed script.

## Licence

MIT for code (`LICENSE`); CC BY 4.0 for derived data (`LICENSE-DATA`). The
upstream MaveDB deposits carry CC0 or CC BY 4.0 terms that pass through, and
the `alphagenome` column derives from a non-commercial API tier — both are
detailed in `LICENSE-DATA`.
